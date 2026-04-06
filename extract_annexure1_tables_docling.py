from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from annexure_table_utils import (
    OCRTextPage,
    TableExtraction,
    build_pdf_list,
    find_annexure_span,
    safe_stem,
    write_manifest,
    write_table_outputs,
    write_text_pages,
)
from script_logging import configure_logger

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "docling is required. Install it with: pip install -U docling easyocr pandas"
    ) from exc


LOGGER = logging.getLogger("docling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Annexure 1 table from every PDF in a folder using Docling.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/docling"))
    parser.add_argument("--pdf-glob", default="*.pdf")
    parser.add_argument("--target-annexure", type=int, default=1)
    parser.add_argument("--max-annexure-pages", type=int, default=20)
    parser.add_argument(
        "--force-full-page-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ocr-lang", nargs="+", default=["en"])
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_converter(args: argparse.Namespace) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    if getattr(pipeline_options, "table_structure_options", None) is not None:
        pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=args.ocr_lang,
        force_full_page_ocr=args.force_full_page_ocr,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def get_doc_page_numbers(document) -> list[int]:
    pages = getattr(document, "pages", None)
    if isinstance(pages, dict):
        numbers = []
        for key, page in pages.items():
            page_no = getattr(page, "page_no", None)
            numbers.append(int(page_no if page_no is not None else key))
        return sorted(numbers)

    if isinstance(pages, list):
        numbers = []
        for index, page in enumerate(pages, start=1):
            numbers.append(int(getattr(page, "page_no", index)))
        return numbers

    return []


def page_markdown(document, page_no: int) -> str:
    for method_name in ("export_to_markdown", "export_to_text"):
        method = getattr(document, method_name, None)
        if not callable(method):
            continue
        try:
            content = method(page_no=page_no)
        except TypeError:
            continue
        if content:
            return str(content)
    return ""


def table_page_no(table) -> int | None:
    prov = getattr(table, "prov", None) or []
    for item in prov:
        page_no = getattr(item, "page_no", None)
        if page_no is not None:
            return int(page_no)
    return None


def table_raw_output(table, document, table_df: pd.DataFrame) -> str:
    exporter = getattr(table, "export_to_html", None)
    if callable(exporter):
        try:
            return str(exporter(doc=document))
        except TypeError:
            pass
    return table_df.to_csv(index=False)


def process_pdf(args: argparse.Namespace, converter: DocumentConverter, pdf_path: Path) -> None:
    LOGGER.info("Converting %s", pdf_path.name)
    pdf_output_dir = args.output_dir / safe_stem(pdf_path)
    text_output_dir = pdf_output_dir / "ocr_text"

    conv_res = converter.convert(pdf_path)
    document = conv_res.document

    page_numbers = get_doc_page_numbers(document)
    text_pages = [OCRTextPage(page_no=page_no, text=page_markdown(document, page_no)) for page_no in page_numbers]
    write_text_pages(text_pages, text_output_dir)

    span = find_annexure_span(
        text_pages,
        target_annexure=args.target_annexure,
        max_annexure_pages=args.max_annexure_pages,
    )

    manifest = {
        "engine": "docling",
        "pdf_path": str(pdf_path.resolve()),
        "target_annexure": args.target_annexure,
        "start_page": span.start_page,
        "end_page": span.end_page,
        "candidate_pages": span.candidate_pages,
        "end_trigger_page": span.end_trigger_page,
        "truncated_by_limit": span.truncated_by_limit,
        "ocr_lang": args.ocr_lang,
        "force_full_page_ocr": args.force_full_page_ocr,
    }

    if span.start_page is None or span.end_page is None:
        LOGGER.warning("Annexure 1 not found")
        write_manifest(pdf_output_dir, manifest)
        return

    table_pages: list[TableExtraction] = []
    for index, table in enumerate(getattr(document, "tables", []), start=1):
        page_no = table_page_no(table)
        if page_no is None:
            continue
        if not (span.start_page <= page_no <= span.end_page):
            continue

        table_df: pd.DataFrame = table.export_to_dataframe(doc=document)
        raw_output = table_raw_output(table, document, table_df)
        table_pages.append(
            TableExtraction(
                page_no=page_no,
                raw_output=raw_output,
                tables=[table_df.fillna("")],
            )
        )
        LOGGER.info("table %s: page %s", index, page_no)

    manifest.update(write_table_outputs(table_pages, pdf_output_dir / "tables"))
    write_manifest(pdf_output_dir, manifest)


def main() -> None:
    args = parse_args()
    pdf_paths = build_pdf_list(args.data_dir, args.pdf_glob)
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (args.output_dir / "run.log")
    configure_logger("docling", log_file, args.log_level)
    LOGGER.info("Starting Docling extractor")
    converter = build_converter(args)

    for pdf_path in pdf_paths:
        try:
            process_pdf(args, converter, pdf_path)
        except Exception:
            LOGGER.exception("Failed while processing %s", pdf_path)
            raise


if __name__ == "__main__":
    main()
