from __future__ import annotations

import re
from typing import Dict, Any, Iterable


def extract_annexure_ii(text: str) -> str | None:
    """Return the text block for Annexure II.

    - Finds the first occurrence of "Annexure II" (OCR tolerant, e.g., Annexure ll)
    - Returns text from there up to the next annexure heading or end of text
    """
    # Be tolerant to OCR confusions between I and l
    annexure_ii_pat = re.compile(r"Annexure\s*[I1][I1]", re.IGNORECASE)
    m = annexure_ii_pat.search(text)
    if not m:
        return None

    start = m.start()
    # Next annexure heading (Annexure I/II/III/IV/V etc.) beyond the current one
    next_annex_pat = re.compile(r"\n\s*Annexure\s+[IVXLCDM1]+\b", re.IGNORECASE)
    mnext = next_annex_pat.search(text, m.end())
    end = mnext.start() if mnext else len(text)
    return text[start:end].strip()


def extract_annexures(text: str) -> list[dict[str, Any]]:
    """Find all Annexure sections and return list with metadata.

    Each item: {index: 1-based order, heading: 'Annexure II', numeral: 'II', start, end, text}
    OCR tolerant to I/1 confusions.
    """
    # Match lines starting with Annexure <numeral>
    pat = re.compile(r"(?mi)^\s*Annexure\s+([IVXLCDM1]+)\b.*$")
    matches = list(pat.finditer(text))
    results: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        numeral = m.group(1)
        heading = m.group(0).strip()
        results.append({
            "index": i + 1,
            "heading": heading,
            "numeral": numeral,
            "start": start,
            "end": end,
            "text": text[start:end].strip(),
        })
    return results


def _clean_num(s: str) -> float | int | None:
    s2 = re.sub(r"[^0-9.,-]", "", s or "").replace(",", "")
    if not s2:
        return None
    try:
        return int(s2) if "." not in s2 else float(s2)
    except ValueError:
        return None


def _yield_kv_lines(lines: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # Pattern 1: Label: Value
        m = re.match(r"^(?P<label>[^:]{2,}?)\s*:\s*(?P<value>.+)$", s)
        if m:
            label = re.sub(r"\s+", " ", m.group("label").strip())
            value = m.group("value").strip()
            num = _clean_num(value)
            out[label] = num if num is not None else value
            continue
        # Pattern 2: Label ..... amount (amount at the end)
        m2 = re.match(r"^(?P<label>.+?)\s+(?P<amount>[\-\d][\d,]*(?:\.\d{1,2})?)$", s)
        if m2:
            label = re.sub(r"\s+", " ", m2.group("label").strip())
            amount = _clean_num(m2.group("amount"))
            out[label] = amount if amount is not None else m2.group("amount")
            continue
    return out


def parse(text: str) -> Dict[str, Any]:
    """Parse Annexure II into a flat dict of fields.

    Returns a dict with keys:
      - Section: "Annexure II"
      - RawText: the raw annexure block
      - plus key-value pairs detected from the annexure text
    """
    block = extract_annexure_ii(text)
    if not block:
        return {}
    # split by lines and parse common patterns
    lines = block.splitlines()
    kv = _yield_kv_lines(lines)
    result: Dict[str, Any] = {"Section": "Annexure II", "RawText": block}
    # Merge parsed kv pairs
    result.update(kv)
    return result


def parse_block(block_text: str) -> Dict[str, Any]:
    """Parse a given annexure block (no heading search)."""
    lines = (block_text or "").splitlines()
    kv = _yield_kv_lines(lines)
    out: Dict[str, Any] = {"RawText": block_text}
    out.update(kv)
    return out


