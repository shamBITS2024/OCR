from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import torch

from annexure_table_utils import (
    OCRTextPage,
    TableExtraction,
    build_pdf_list,
    extract_tables_from_markup,
    find_annexure_span,
    iter_pdf_pages,
    safe_stem,
    write_manifest,
    write_table_outputs,
    write_text_pages,
)
from script_logging import configure_logger

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "transformers is required. Install it with: pip install -U transformers accelerate pillow pymupdf pandas"
    ) from exc


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEFAULT_PROMPT = "<|grounding|>Convert the document to markdown."
LOGGER = logging.getLogger("deepseek_ocr")


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class DeepSeekOcrRunner:
    def __init__(self, model_id: str, work_dir: Path, prompt: str) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": preferred_dtype(),
        }
        if torch.cuda.is_available():
            load_kwargs["_attn_implementation"] = "flash_attention_2"

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def infer(self, image, page_no: int) -> str:
        image_path = self.work_dir / f"page_{page_no:04d}.png"
        image.save(image_path)

        result = self.model.infer(
            tokenizer=self.tokenizer,
            prompt=self.prompt,
            image_file=str(image_path),
            output_path=str(self.work_dir),
            base_size=1024,
            image_size=1024,
            crop_mode="none",
            save_results=False,
            test_compress=False,
        )

        if isinstance(result, dict):
            for key in ("markdown", "text", "result", "output"):
                if key in result and result[key]:
                    return str(result[key]).strip()
        return str(result).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Annexure 1 table from every PDF in a folder using DeepSeek-OCR.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deepseek_ocr"))
    parser.add_argument("--pdf-glob", default="*.pdf")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--target-annexure", type=int, default=1)
    parser.add_argument("--max-annexure-pages", type=int, default=20)
    parser.add_argument("--stop-after-empty-pages", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def process_pdf(args: argparse.Namespace, runner: DeepSeekOcrRunner, pdf_path: Path) -> None:
    pdf_output_dir = args.output_dir / safe_stem(pdf_path)
    text_output_dir = pdf_output_dir / "ocr_text"
    text_pages: list[OCRTextPage] = []

    LOGGER.info("Converting pages to markdown: %s", pdf_path.name)
    for page_no, image in iter_pdf_pages(pdf_path, dpi=args.dpi):
        markdown = runner.infer(image, page_no)
        text_pages.append(OCRTextPage(page_no=page_no, text=markdown))
        LOGGER.info("page %s: OCR done", page_no)

    write_text_pages(text_pages, text_output_dir)

    span = find_annexure_span(
        text_pages,
        target_annexure=args.target_annexure,
        max_annexure_pages=args.max_annexure_pages,
    )

    manifest = {
        "engine": "deepseek_ocr",
        "model_id": args.model_id,
        "pdf_path": str(pdf_path.resolve()),
        "target_annexure": args.target_annexure,
        "prompt": args.prompt,
        "start_page": span.start_page,
        "end_page": span.end_page,
        "candidate_pages": span.candidate_pages,
        "end_trigger_page": span.end_trigger_page,
        "truncated_by_limit": span.truncated_by_limit,
    }

    if span.start_page is None or span.end_page is None:
        LOGGER.warning("Annexure 1 not found")
        write_manifest(pdf_output_dir, manifest)
        return

    table_pages: list[TableExtraction] = []
    found_any_tables = False
    empty_streak = 0

    LOGGER.info("Collecting tables from pages %s to %s", span.start_page, span.end_page)
    for page in text_pages:
        if not (span.start_page <= page.page_no <= span.end_page):
            continue

        tables = extract_tables_from_markup(page.text)
        table_pages.append(TableExtraction(page_no=page.page_no, raw_output=page.text, tables=tables))

        if tables:
            found_any_tables = True
            empty_streak = 0
        elif found_any_tables:
            empty_streak += 1

        LOGGER.info("page %s: extracted %s table(s)", page.page_no, len(tables))
        if found_any_tables and args.stop_after_empty_pages > 0 and empty_streak >= args.stop_after_empty_pages:
            LOGGER.info("stopping after %s empty page(s) following detected tables", empty_streak)
            break

    manifest.update(write_table_outputs(table_pages, pdf_output_dir / "tables"))
    write_manifest(pdf_output_dir, manifest)


def main() -> None:
    args = parse_args()
    pdf_paths = build_pdf_list(args.data_dir, args.pdf_glob)
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (args.output_dir / "run.log")
    configure_logger("deepseek_ocr", log_file, args.log_level)
    LOGGER.info("Starting DeepSeek-OCR extractor")
    LOGGER.info("Using model: %s", args.model_id)

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as temp_dir:
        runner = DeepSeekOcrRunner(args.model_id, Path(temp_dir), args.prompt)
        for pdf_path in pdf_paths:
            try:
                process_pdf(args, runner, pdf_path)
            except Exception:
                LOGGER.exception("Failed while processing %s", pdf_path)
                raise


if __name__ == "__main__":
    main()
