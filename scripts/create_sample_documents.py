from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "samples"
PAGE_SIZE = (1600, 2100)
MARGIN = 130
FONT_SIZE = 38
LINE_GAP = 24

DOCUMENTS = {
    "synthetic_mortgage_application.png": [
        "MORTGAGE APPLICATION REVIEW",
        "Case ID: CASE-OCR-201",
        "Document Type: Mortgage Application",
        "Borrower ID: SYNTHETIC-BORROWER-001",
        "Signature Status: MISSING",
        "Exception: The borrower signature is missing from the mortgage application.",
    ],
    "synthetic_income_document.png": [
        "INCOME DOCUMENT CHECKLIST",
        "Case ID: CASE-OCR-202",
        "Document Type: Income Verification",
        "Borrower ID: SYNTHETIC-BORROWER-002",
        "Required Document Status: NOT PROVIDED",
        "Exception: The recent paystub required for income verification is missing from the loan file.",
    ],
    "synthetic_property_address.png": [
        "PROPERTY ADDRESS COMPARISON",
        "Case ID: CASE-OCR-203",
        "Document Type: Appraisal and Title Review",
        "Borrower ID: SYNTHETIC-BORROWER-003",
        "Appraisal Property Address: 123 Example Avenue, Sampletown, CA 90001",
        "Title Record Address: 123 Example Avenue, Sampletown, CA 90010",
        "Exception: The property address on the appraisal conflicts with the title record.",
    ],
}


def load_font() -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    font = load_font()
    for filename, lines in DOCUMENTS.items():
        image = Image.new("RGB", PAGE_SIZE, "white")
        draw = ImageDraw.Draw(image)
        y = MARGIN
        for index, line in enumerate(lines):
            wrapped_lines = textwrap.wrap(line, width=68) or [""]
            for wrapped_line in wrapped_lines:
                draw.text((MARGIN, y), wrapped_line, fill="black", font=font)
                y += FONT_SIZE + LINE_GAP
            if index == 0:
                y += 60
        image.save(OUTPUT_DIR / filename, dpi=(200, 200))


if __name__ == "__main__":
    main()
