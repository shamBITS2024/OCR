from __future__ import annotations

import argparse
import logging
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
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "transformers is required. Install it with: pip install -U transformers accelerate pillow pymupdf pandas"
    ) from exc

try:
    from transformers import GlmOcrForConditionalGeneration as _GlmOcrModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForImageTextToText as _GlmOcrModel


DEFAULT_MODEL_ID = "zai-org/GLM-OCR"
LOGGER = logging.getLogger("glm_ocr")


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class GlmOcrRunner:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs = {
            "torch_dtype": preferred_dtype(),
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        self.model = _GlmOcrModel.from_pretrained(model_id, **load_kwargs).eval()
        self.device = next(self.model.parameters()).device

    def infer(self, image, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_tokens = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, prompt_tokens:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        if decoded:
            return decoded
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Annexure 1 table from every PDF in a folder using GLM-OCR.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glm_ocr"))
    parser.add_argument("--pdf-glob", default="*.pdf")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--target-annexure", type=int, default=1)
    parser.add_argument("--max-annexure-pages", type=int, default=20)
    parser.add_argument("--stop-after-empty-pages", type=int, default=2)
    parser.add_argument("--max-text-tokens", type=int, default=768)
    parser.add_argument("--max-table-tokens", type=int, default=1536)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def process_pdf(args: argparse.Namespace, runner: GlmOcrRunner, pdf_path: Path) -> None:
    pdf_output_dir = args.output_dir / safe_stem(pdf_path)
    text_output_dir = pdf_output_dir / "ocr_text"
    text_pages: list[OCRTextPage] = []

    LOGGER.info("Scanning pages for Annexure %s: %s", args.target_annexure, pdf_path.name)
    for page_no, image in iter_pdf_pages(pdf_path, dpi=args.dpi):
        text = runner.infer(image, "Text Recognition:", args.max_text_tokens)
        text_pages.append(OCRTextPage(page_no=page_no, text=text))
        LOGGER.info("page %s: OCR done", page_no)

    write_text_pages(text_pages, text_output_dir)

    span = find_annexure_span(
        text_pages,
        target_annexure=args.target_annexure,
        max_annexure_pages=args.max_annexure_pages,
    )

    manifest = {
        "engine": "glm_ocr",
        "model_id": args.model_id,
        "pdf_path": str(pdf_path.resolve()),
        "target_annexure": args.target_annexure,
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
    page_numbers = list(range(span.start_page, span.end_page + 1))

    LOGGER.info("Extracting tables from pages %s to %s", span.start_page, span.end_page)
    for page_no, image in iter_pdf_pages(pdf_path, dpi=args.dpi, page_numbers=page_numbers):
        raw_output = runner.infer(image, "Table Recognition:", args.max_table_tokens)
        tables = extract_tables_from_markup(raw_output)
        table_pages.append(TableExtraction(page_no=page_no, raw_output=raw_output, tables=tables))

        if tables:
            found_any_tables = True
            empty_streak = 0
        elif found_any_tables:
            empty_streak += 1

        LOGGER.info("page %s: extracted %s table(s)", page_no, len(tables))
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
    configure_logger("glm_ocr", log_file, args.log_level)
    LOGGER.info("Starting GLM-OCR extractor")
    LOGGER.info("Using model: %s", args.model_id)
    runner = GlmOcrRunner(args.model_id)

    for pdf_path in pdf_paths:
        try:
            process_pdf(args, runner, pdf_path)
        except Exception:
            LOGGER.exception("Failed while processing %s", pdf_path)
            raise


if __name__ == "__main__":
    main()
