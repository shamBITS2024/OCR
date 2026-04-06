from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import fitz
import pandas as pd
from PIL import Image


ANNEXURE_TOKEN_RE = re.compile(r"(?i)\bannexure\b[\s:._-]*([a-z0-9ivxlcdm]+)\b")
TABLE_HINT_RE = re.compile(
    r"(?is)<table|^\s*\|.+\|\s*$|work\s*code|description|amount|rate|duty|"
    r"serial\s*no|s\.?\s*no|sr\.?\s*no|minimum\s+guaranteed",
)


@dataclass
class OCRTextPage:
    page_no: int
    text: str


@dataclass
class TableExtraction:
    page_no: int
    raw_output: str
    tables: list[pd.DataFrame] = field(default_factory=list)


@dataclass
class AnnexureSpan:
    start_page: int | None
    end_page: int | None
    candidate_pages: list[int] = field(default_factory=list)
    end_trigger_page: int | None = None
    truncated_by_limit: bool = False


def iter_pdf_pages(
    pdf_path: Path,
    *,
    dpi: int = 200,
    page_numbers: Sequence[int] | None = None,
) -> Iterator[tuple[int, Image.Image]]:
    selected = set(page_numbers) if page_numbers else None
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            if selected and index not in selected:
                continue
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            yield index, image


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_stem(path: Path | str) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "document"


def parse_annexure_number(token: str) -> int | None:
    token = (token or "").strip().lower()
    if not token:
        return None

    if token.isdigit():
        try:
            return int(token)
        except ValueError:
            return None

    token = token.replace("1", "i").replace("l", "i")
    if set(token) <= {"i"}:
        return len(token)

    roman_map = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    try:
        values = [roman_map[ch] for ch in token]
    except KeyError:
        return None

    total = 0
    prev = 0
    for value in reversed(values):
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total or None


def annexure_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in ANNEXURE_TOKEN_RE.finditer(text or ""):
        number = parse_annexure_number(match.group(1))
        if number is not None:
            numbers.append(number)
    return numbers


def heading_score(text: str, target_annexure: int) -> int:
    score = 0
    matched_annexure = False
    lines = [normalize_text(line) for line in (text or "").splitlines() if line.strip()]
    top_lines = lines[:20]

    for line_index, line in enumerate(top_lines):
        numbers = annexure_numbers(line)
        if target_annexure in numbers:
            matched_annexure = True
            score += max(1, 10 - line_index)
            if re.match(r"(?i)^annexure\b", line):
                score += 5
            break

    first_chunk = normalize_text((text or "")[:500])
    if target_annexure in annexure_numbers(first_chunk):
        matched_annexure = True
        score += 2
    if matched_annexure and TABLE_HINT_RE.search(text or ""):
        score += 1
    return score


def has_other_annexure_heading(text: str, target_annexure: int) -> bool:
    lines = [normalize_text(line) for line in (text or "").splitlines() if line.strip()]
    for line in lines[:20]:
        for number in annexure_numbers(line):
            if number != target_annexure:
                return True
    return False


def find_annexure_span(
    page_texts: Sequence[OCRTextPage],
    *,
    target_annexure: int = 1,
    max_annexure_pages: int | None = 20,
) -> AnnexureSpan:
    candidates: list[tuple[int, int]] = []

    for page in page_texts:
        score = heading_score(page.text, target_annexure)
        if score > 0:
            candidates.append((page.page_no, score))

    if not candidates:
        fallback_pages = [
            page.page_no for page in page_texts if target_annexure in annexure_numbers(page.text)
        ]
        if not fallback_pages:
            return AnnexureSpan(start_page=None, end_page=None)
        start_page = fallback_pages[-1]
        candidate_pages = fallback_pages
    else:
        strong = [page_no for page_no, score in candidates if score >= 8]
        candidate_pages = [page_no for page_no, _score in candidates]
        start_page = (strong or candidate_pages)[-1]

    end_page = page_texts[-1].page_no
    end_trigger_page: int | None = None

    for page in page_texts:
        if page.page_no <= start_page:
            continue
        if has_other_annexure_heading(page.text, target_annexure):
            end_page = page.page_no - 1
            end_trigger_page = page.page_no
            break

    truncated = False
    if max_annexure_pages is not None and end_page >= start_page:
        limited_end = start_page + max_annexure_pages - 1
        if end_page > limited_end:
            end_page = limited_end
            truncated = True

    return AnnexureSpan(
        start_page=start_page,
        end_page=end_page,
        candidate_pages=candidate_pages,
        end_trigger_page=end_trigger_page,
        truncated_by_limit=truncated,
    )


def strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        parts = stripped.splitlines()
        if len(parts) >= 2:
            return "\n".join(parts[1:-1]).strip()
    return text or ""


def _split_markdown_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _split_markdown_row(line)
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{3,}:?", cell or "")) for cell in cells)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.fillna("")
    cleaned.columns = [
        normalize_text(str(col)) or f"col_{index + 1}" for index, col in enumerate(cleaned.columns)
    ]
    cleaned = cleaned.apply(lambda column: column.map(lambda value: normalize_text(str(value))))
    cleaned = cleaned.loc[:, [col for col in cleaned.columns if any(cleaned[col].astype(str).str.len() > 0)]]
    if cleaned.empty:
        return cleaned
    mask = cleaned.apply(lambda row: any(bool(cell) for cell in row), axis=1)
    cleaned = cleaned.loc[mask].reset_index(drop=True)
    return cleaned


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    payload = {
        "columns": list(df.columns),
        "rows": df.astype(str).values.tolist(),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def extract_tables_from_markup(text: str) -> list[pd.DataFrame]:
    cleaned = strip_code_fences(text)
    tables: list[pd.DataFrame] = []
    seen: set[str] = set()

    if "<table" in cleaned.lower():
        try:
            for df in pd.read_html(StringIO(cleaned)):
                normalized = _normalize_dataframe(df)
                if normalized.empty:
                    continue
                fp = _dataframe_fingerprint(normalized)
                if fp not in seen:
                    tables.append(normalized)
                    seen.add(fp)
        except ValueError:
            pass

    block: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        if line.count("|") >= 2:
            block.append(line)
            continue
        if block:
            for df in _parse_markdown_table_block(block):
                fp = _dataframe_fingerprint(df)
                if fp not in seen:
                    tables.append(df)
                    seen.add(fp)
            block = []
    if block:
        for df in _parse_markdown_table_block(block):
            fp = _dataframe_fingerprint(df)
            if fp not in seen:
                tables.append(df)
                seen.add(fp)

    return tables


def _parse_markdown_table_block(lines: Sequence[str]) -> list[pd.DataFrame]:
    if len(lines) < 2:
        return []

    header = _split_markdown_row(lines[0])
    data_lines = list(lines[1:])
    if data_lines and _is_markdown_separator(data_lines[0]):
        data_lines = data_lines[1:]
    if not data_lines:
        return []

    width = max(len(header), *(len(_split_markdown_row(line)) for line in data_lines))
    header = _pad_row(header, width)
    rows = [_pad_row(_split_markdown_row(line), width) for line in data_lines]

    df = pd.DataFrame(rows, columns=header)
    df = _normalize_dataframe(df)
    return [df] if not df.empty else []


def _pad_row(row: list[str], width: int) -> list[str]:
    if len(row) < width:
        row = row + [""] * (width - len(row))
    return row[:width]


def merge_table_extractions(table_pages: Sequence[TableExtraction]) -> pd.DataFrame | None:
    collected: list[pd.DataFrame] = []
    width_groups: dict[int, list[tuple[int, pd.DataFrame]]] = {}

    for page in table_pages:
        for table in page.tables:
            width_groups.setdefault(len(table.columns), []).append((page.page_no, table))

    if not width_groups:
        return None

    best_width = max(width_groups, key=lambda width: sum(len(df) for _page_no, df in width_groups[width]))
    selected = width_groups[best_width]
    base_columns = list(selected[0][1].columns)

    for page_no, df in selected:
        current = df.copy()
        current = current.iloc[:, : len(base_columns)]
        current.columns = base_columns
        current.insert(0, "source_page", page_no)
        if not current.empty:
            header_like = current.apply(
                lambda row: [normalize_text(str(value)).lower() for value in row[1:]]
                == [normalize_text(str(col)).lower() for col in base_columns],
                axis=1,
            )
            current = current.loc[~header_like]
        collected.append(current)

    if not collected:
        return None
    merged = pd.concat(collected, ignore_index=True)
    return _normalize_dataframe(merged)


def write_text_pages(page_texts: Sequence[OCRTextPage], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for page in page_texts:
        page_file = output_dir / f"page_{page.page_no:04d}.txt"
        page_file.write_text(page.text or "", encoding="utf-8")


def write_table_outputs(table_pages: Sequence[TableExtraction], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_model_output"
    page_tables_dir = output_dir / "page_tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_tables_dir.mkdir(parents=True, exist_ok=True)

    table_count = 0
    pages_with_tables = 0
    for page in table_pages:
        (raw_dir / f"page_{page.page_no:04d}.txt").write_text(page.raw_output or "", encoding="utf-8")
        if page.tables:
            pages_with_tables += 1
        for index, table in enumerate(page.tables, start=1):
            table_count += 1
            table.to_csv(
                page_tables_dir / f"page_{page.page_no:04d}_table_{index:02d}.csv",
                index=False,
            )

    merged = merge_table_extractions(table_pages)
    if merged is not None and not merged.empty:
        merged.to_csv(output_dir / "annexure_1_merged.csv", index=False)

    return {
        "page_count_in_span": len(table_pages),
        "pages_with_tables": pages_with_tables,
        "table_count": table_count,
        "merged_row_count": 0 if merged is None else int(len(merged)),
    }


def build_pdf_list(data_dir: Path, pdf_glob: str = "*.pdf") -> list[Path]:
    return sorted(path for path in data_dir.glob(pdf_glob) if path.is_file())


def write_manifest(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
