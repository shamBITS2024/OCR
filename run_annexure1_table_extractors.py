from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from script_logging import configure_logger


ENGINE_CONFIG = {
    "glm_ocr": {
        "script": "extract_annexure1_tables_glm_ocr.py",
        "default_output_subdir": "glm_ocr",
    },
    "deepseek_ocr": {
        "script": "extract_annexure1_tables_deepseek_ocr.py",
        "default_output_subdir": "deepseek_ocr",
    },
    "docling": {
        "script": "extract_annexure1_tables_docling.py",
        "default_output_subdir": "docling",
    },
}

LOGGER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Annexure 1 table extractors against the PDFs in the data folder.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=tuple(ENGINE_CONFIG),
        default=list(ENGINE_CONFIG),
        help="Which backends to run.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--pdf-glob", default="*.pdf")
    parser.add_argument("--target-annexure", type=int, default=1)
    parser.add_argument("--max-annexure-pages", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--stop-after-empty-pages", type=int, default=2)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--glm-python", type=Path)
    parser.add_argument("--deepseek-python", type=Path)
    parser.add_argument("--docling-python", type=Path)
    parser.add_argument("--glm-model-id", default="zai-org/GLM-OCR")
    parser.add_argument("--deepseek-model-id", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--deepseek-prompt", default="<|grounding|>Convert the document to markdown.")
    parser.add_argument("--max-text-tokens", type=int, default=768)
    parser.add_argument("--max-table-tokens", type=int, default=1536)
    parser.add_argument(
        "--force-full-page-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Docling OCR mode.",
    )
    parser.add_argument("--ocr-lang", nargs="+", default=["en"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_python(args: argparse.Namespace, engine: str) -> Path:
    if engine == "glm_ocr" and args.glm_python:
        return args.glm_python
    if engine == "deepseek_ocr" and args.deepseek_python:
        return args.deepseek_python
    if engine == "docling" and args.docling_python:
        return args.docling_python
    return args.python


def build_command(args: argparse.Namespace, engine: str, script_path: Path, output_dir: Path) -> list[str]:
    python_path = resolve_python(args, engine)
    command = [
        str(python_path),
        str(script_path),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--pdf-glob",
        args.pdf_glob,
        "--target-annexure",
        str(args.target_annexure),
        "--max-annexure-pages",
        str(args.max_annexure_pages),
    ]

    if engine == "glm_ocr":
        command.extend(
            [
                "--model-id",
                args.glm_model_id,
                "--dpi",
                str(args.dpi),
                "--stop-after-empty-pages",
                str(args.stop_after_empty_pages),
                "--max-text-tokens",
                str(args.max_text_tokens),
                "--max-table-tokens",
                str(args.max_table_tokens),
            ]
        )
    elif engine == "deepseek_ocr":
        command.extend(
            [
                "--model-id",
                args.deepseek_model_id,
                "--dpi",
                str(args.dpi),
                "--stop-after-empty-pages",
                str(args.stop_after_empty_pages),
                "--prompt",
                args.deepseek_prompt,
            ]
        )
    elif engine == "docling":
        command.extend(["--force-full-page-ocr" if args.force_full_page_ocr else "--no-force-full-page-ocr"])
        if args.ocr_lang:
            command.extend(["--ocr-lang", *args.ocr_lang])

    return command


def main() -> None:
    global LOGGER
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (args.output_root / "logs")
    LOGGER = configure_logger("annexure1_launcher", log_dir / "launcher.log", args.log_level)

    run_summary: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "log_dir": str(log_dir.resolve()),
        "engines": [],
    }

    overall_ok = True
    for engine in args.engines:
        config = ENGINE_CONFIG[engine]
        script_path = base_dir / config["script"]
        output_dir = args.output_root / config["default_output_subdir"]
        engine_log_path = log_dir / f"{engine}.log"
        command = build_command(args, engine, script_path, output_dir)
        command.extend(["--log-file", str(engine_log_path), "--log-level", args.log_level])

        LOGGER.info("Running %s", engine)
        LOGGER.info("%s", " ".join(f'"{part}"' if " " in part else part for part in command))

        engine_summary = {
            "engine": engine,
            "script": str(script_path),
            "python": str(resolve_python(args, engine)),
            "output_dir": str(output_dir),
            "log_file": str(engine_log_path),
            "command": command,
        }

        if args.dry_run:
            engine_summary["status"] = "dry_run"
            run_summary["engines"].append(engine_summary)
            continue

        completed = subprocess.run(command, cwd=base_dir, check=False)
        engine_summary["returncode"] = completed.returncode
        engine_summary["status"] = "ok" if completed.returncode == 0 else "failed"
        run_summary["engines"].append(engine_summary)

        if completed.returncode != 0:
            overall_ok = False
            LOGGER.error("%s failed with return code %s", engine, completed.returncode)
            if args.fail_fast:
                break
        else:
            LOGGER.info("%s completed successfully", engine)

    run_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_summary["status"] = "ok" if overall_ok else "failed"

    summary_path = args.output_root / "annexure1_launcher_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
