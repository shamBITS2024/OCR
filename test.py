import os
import re
import glob
import json
import shutil
import pdf2image
from parser import (
    parse as parse_annex,
    extract_annexure_ii as annex_extract,
    extract_annexures as annex_list,
    parse_block as parse_block,
)
try:
    from PIL import Image  # noqa: F401
except ImportError:
    import Image  # type: ignore  # noqa: F401
import pytesseract


def _find_poppler_bin() -> str | None:
    """Best-effort locate Poppler's bin folder on Windows.

    Common winget install path: C:\\Program Files\\poppler-<ver>\\Library\\bin
    """
    candidates = [
        r"C:\\Program Files\\poppler-*\\Library\\bin",
        r"C:\\Program Files (x86)\\poppler-*\\Library\\bin",
    ]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern), reverse=True)
        for m in matches:
            if os.path.isdir(m):
                return m
    return None


def _find_tesseract_exe() -> str | None:
    """Locate tesseract.exe or return None if not found."""
    candidates = [
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    which = shutil.which("tesseract")
    return which


def _configure_dependencies() -> str | None:
    """Configure PATH and pytesseract if running on Windows.

    Returns the poppler bin path if found (for passing to pdf2image).
    """
    poppler_bin = _find_poppler_bin()
    if poppler_bin and poppler_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + poppler_bin

    tess_exe = _find_tesseract_exe()
    if tess_exe:
        pytesseract.pytesseract.tesseract_cmd = tess_exe

    return poppler_bin


def pdf_to_img(pdf_file: str, poppler_path: str | None = None):
    kwargs = {"poppler_path": poppler_path} if poppler_path else {}
    return pdf2image.convert_from_path(pdf_file, **kwargs)


def ocr_core(img: Image.Image, timeout: int | None = None) -> str:
    return pytesseract.image_to_string(img, timeout=timeout)


def parse_payslip(text: str) -> dict:
    flags = re.IGNORECASE | re.MULTILINE

    designation = _first_match(
        text,
        [
            re.compile(r"Designation\s*:\s*(.+)", flags),
        ],
    )

    period = _first_match(
        text,
        [
            re.compile(r"Payment\s*For\s*The\s*Period\s*:\s*(.+)", flags),
            re.compile(r"For\s*The\s*Period\s*:\s*(.+)", flags),
        ],
    )

    pran = _first_match(
        text,
        [
            re.compile(r"PRAN\s*(?:No\.?|Number)?\s*:\s*([A-Z0-9]+)", flags),
        ],
    )

    basic_pay_raw = _first_match(
        text,
        [
            re.compile(r"Basic\s*Pay\s*:\s*([\d,\.]+)", flags),
            re.compile(r"(?m)^\s*Basic\s*[\/.]*\s*Off\.?\s*Pay\s+([\d,\.]+)\b", flags),
        ],
    )
    basic_pay = _clean_amount(basic_pay_raw)

    bank_acct = _first_match(
        text,
        [
            re.compile(r"Bank\s*A\s*[\\/cC]\s*No\s*:\s*([A-Z*Xx\d]+)", flags),
            re.compile(r"Bank\s*Account\s*No\s*:\s*([A-Z*Xx\d]+)", flags),
        ],
    )

    da = _clean_amount(
        _first_match(
            text,
            [re.compile(r"(?m)^\s*DA\s+([\d,\.]+)\b", flags)],
        )
    )

    hra = _clean_amount(
        _first_match(
            text,
            [re.compile(r"(?m)^\s*HRA\s+([\d,\.]+)\b", flags)],
        )
    )

    cghs = _clean_amount(
        _first_match(
            text,
            [re.compile(r"(?m)^\s*CGHS\s+([\d,\.]+)\b", flags)],
        )
    )

    tax = _clean_amount(
        _first_match(
            text,
            [
                re.compile(r"(?m)^\s*(?:Income\s*Tax|ITax|Tax)\s+([\d,\.]+)\b", flags),
            ],
        )
    )

    net_pay = _clean_amount(
        _first_match(
            text,
            [re.compile(r"Net\s*Pay\s*:\s*([\d,\.]+)", flags)],
        )
    )

    return {
        "Designation": designation,
        "Period": period,
        "PRAN": pran,
        "BasicPay": basic_pay,
        "BankAccount": bank_acct,
        "DA": da,
        "HRA": hra,
        "CGHS": cghs,
        "Tax": tax,
        "NetPay": net_pay,
    }


def _clean_amount(val: str | None) -> int | float | None:
    if not val:
        return None
    s = re.sub(r"[^0-9.,-]", "", val)
    s = s.replace(",", "")
    if s == "":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


def _first_match(text: str, patterns: list[re.Pattern], group: int = 1) -> str | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(group).strip()
    return None




def print_pages(pdf_file: str):
    poppler_bin = _configure_dependencies()
    try:
        images = pdf_to_img(pdf_file, poppler_path=poppler_bin)
    except Exception as e:
        # Provide a clearer hint if Poppler isn't available
        msg = str(e)
        if "PDFInfoNotInstalledError" in msg or "poppler" in msg.lower():
            raise SystemExit(
                "Poppler not found. Install it (winget install oschwartz10612.Poppler) "
                "and restart terminal, or set poppler_path manually."
            ) from e
        raise
    full_text = []
    for pg, img in enumerate(images, start=1):
        page_text = ocr_core(img)
        print(f"--- Page {pg} ---")
        print(page_text)
        full_text.append(page_text)

    parsed = parse_payslip("\n".join(full_text))
    print("\n== Parsed fields ==")
    # Save to CSV and Excel
    import pandas as pd
    df = pd.DataFrame([parsed])
    df.to_csv("parsed.csv", index=False)
    df.to_excel("parsed.xlsx", index=False)

    print(json.dumps(parsed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Mode A: Process a folder of PDFs to extract Annexure II
    folder = r"D:\Office\ocr_jobs\2023-24 Transfer from"
    if os.path.isdir(folder):
        poppler_bin = _configure_dependencies()
        results = []
        pdf_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder, f))
        ]
        for path in pdf_files:
            print(f"\n=== FILE: {os.path.basename(path)} ===")
            try:
                images = pdf2image.convert_from_path(path, poppler_path=poppler_bin, dpi=200)
            except Exception as e:
                print(f"Failed to open {path}: {e}")
                continue
            # OCR all pages, collect annexure blocks, then pick the 2nd one
            all_text = []
            for idx, img in enumerate(images, start=1):
                try:
                    text = ocr_core(img, timeout=20)
                    print(f"Page {idx}: {len(text)} chars")
                    all_text.append(text)
                except Exception as ocr_err:
                    print(f"OCR failed on page {idx}: {ocr_err}")
            full_text = "\n".join(all_text)
            annexes = annex_list(full_text)
            if not annexes:
                print(f"No Annexure headings found in: {os.path.basename(path)}")
                continue
            print(f"Found {len(annexes)} annexure section(s)")
            # Choose 2nd annexure if available
            target = annexes[1] if len(annexes) >= 2 else annexes[0]
            print(f"Selecting annexure #{target['index']}: {target['heading']}")
            block = target["text"]
            # Save block preview
            outdir = os.path.join(os.path.dirname(__file__), "annexure_text")
            os.makedirs(outdir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            outpath = os.path.join(outdir, f"{base}.annexure{target['index']}.txt")
            try:
                with open(outpath, "w", encoding="utf-8") as fh:
                    fh.write(block)
                print(f"Saved annexure block -> {outpath}")
            except Exception as werr:
                print(f"Warn: could not save annexure text: {werr}")
            # Parse the selected annexure block
            parsed = parse_block(block)
            if not parsed:
                print(f"Selected annexure block not parsed in: {os.path.basename(path)}")
                continue
            parsed_row = {"File": os.path.basename(path)}
            parsed_row.update(parsed)
            results.append(parsed_row)
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv("annexure_ii.csv", index=False)
            df.to_excel("annexure_ii.xlsx", index=False)
            print(f"Saved {len(results)} rows to annexure_ii.csv and annexure_ii.xlsx")
        else:
            print("No Annexure II data extracted.")
    else:
        # Mode B: Single payslip path (existing behavior)
        pdf_path = r"D:\Office\ocr_jobs\rpt_emp_payslip.php.pdf"
        if not os.path.isfile(pdf_path):
            raise SystemExit(f"PDF not found at: {pdf_path}")
        print_pages(pdf_path)
