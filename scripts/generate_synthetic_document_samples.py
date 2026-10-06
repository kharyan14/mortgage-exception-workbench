from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "samples" / "generated"
PAGE_SIZE = (1600, 2200)
MARGIN = 120
FONT_SIZE = 38
LINE_SPACING = 18
WRAP_WIDTH = 64
FIRST_NAMES = [
    "Avery",
    "Jordan",
    "Morgan",
    "Riley",
    "Casey",
    "Taylor",
    "Quinn",
    "Skyler",
    "Cameron",
    "Rowan",
    "Emerson",
    "Parker",
    "Reese",
    "Dakota",
    "Finley",
    "Sage",
    "Drew",
    "Robin",
    "Kendall",
    "Arden",
]
LAST_NAMES = [
    "Example",
    "Sample",
    "Fiction",
    "Demo",
    "Test",
    "Mock",
    "Placeholder",
    "Synthetic",
    "Practice",
    "Training",
]

APPLICATION_EXCEPTIONS = [
    (
        "missing_document",
        "The borrower signature is missing from the mortgage application.",
        ["Signature Status: MISSING", "Income Verification: PRESENT"],
    ),
    (
        "missing_document",
        "The required recent paystub is missing from the supporting loan file.",
        ["Signature Status: PRESENT", "Required Paystub: NOT PROVIDED"],
    ),
    (
        "conflicting_values",
        "The property address on the application conflicts with the title record.",
        [
            "Application Property Address: 10 SYNTHETIC OAK WAY, SAMPLE CITY",
            "Title Record Address: 11 SYNTHETIC OAK WAY, SAMPLE CITY",
        ],
    ),
    (
        "missing_document",
        "The required employer name field is blank on the mortgage application.",
        ["Current Employer: [BLANK]", "Monthly Income: SYNTHETIC VALUE A"],
    ),
    (
        "conflicting_values",
        "The monthly income on the application conflicts with the paystub summary.",
        [
            "Application Monthly Income: SYNTHETIC VALUE A",
            "Paystub Monthly Income: SYNTHETIC VALUE B",
        ],
    ),
    (
        "low_confidence_extraction",
        "The employment start date is faint and could not be read reliably.",
        ["Employment Start Date: [FAINT / UNREADABLE]", "Extraction Quality: LOW"],
    ),
    (
        "unsupported_extraction",
        "The uploaded attachment is an unsupported document and needs reviewer inspection.",
        ["Attachment Type: UNKNOWN", "Extraction Status: UNSUPPORTED"],
    ),
    (
        "missing_document",
        "The borrower signature is missing from the application certification.",
        ["Certification Signature: MISSING", "Income Verification: PRESENT"],
    ),
    (
        "conflicting_values",
        "The mailing address on the application conflicts with the address statement.",
        [
            "Application Mailing Address: 20 SYNTHETIC PINE ROAD, SAMPLE CITY",
            "Address Statement: 22 SYNTHETIC PINE ROAD, SAMPLE CITY",
        ],
    ),
    (
        "low_confidence_extraction",
        "The monthly income field is smudged and cannot be verified from this scan.",
        ["Monthly Income: [SMUDGED / UNREADABLE]", "Extraction Quality: LOW"],
    ),
]

SSN_EXCEPTIONS = [
    (
        "missing_document",
        "The required synthetic identity-verification document was not included.",
        ["Verification Document: NOT PROVIDED", "Identifier: NOT-A-REAL-SSN"],
    ),
    (
        "conflicting_values",
        "The name on the synthetic identity statement conflicts with the application.",
        [
            "Application Name: SYNTHETIC PERSON A",
            "Identity Statement Name: SYNTHETIC PERSON B",
        ],
    ),
    (
        "low_confidence_extraction",
        "The masked identifier field is blurred and cannot be read reliably.",
        ["Identifier: [MASKED / UNREADABLE]", "Extraction Quality: LOW"],
    ),
    (
        "unsupported_extraction",
        "The identity attachment format is unsupported and needs reviewer inspection.",
        ["Attachment Type: UNKNOWN", "Identifier: NOT-A-REAL-SSN"],
    ),
    (
        "missing_document",
        "The synthetic verification statement is missing its required signature.",
        ["Verification Statement: PRESENT", "Reviewer Signature: MISSING"],
    ),
    (
        "conflicting_values",
        "The borrower reference on the identity statement does not match the application.",
        ["Application Reference: SYNTHETIC REF A", "Statement Reference: SYNTHETIC REF B"],
    ),
    (
        "low_confidence_extraction",
        "The issue date is faint and extraction confidence is too low to verify it.",
        ["Issue Date: [FAINT / UNREADABLE]", "Extraction Quality: LOW"],
    ),
    (
        "conflicting_values",
        "Two synthetic identity records contain different placeholder identifiers.",
        ["Record One Identifier: PLACEHOLDER ALPHA", "Record Two Identifier: PLACEHOLDER BETA"],
    ),
    (
        "missing_document",
        "The identity proof is listed in the checklist but no document was attached.",
        ["Checklist Identity Proof: REQUIRED", "Attached Identity Proof: MISSING"],
    ),
    (
        "unsupported_extraction",
        "The uploaded identity page contains no readable supported verification fields.",
        ["Document Type: UNCLASSIFIED", "Extraction Status: UNSUPPORTED"],
    ),
]


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


def draw_document(
    output_path: Path,
    title: str,
    case_id: str,
    doc_type: str,
    category: str,
    exception: str,
    detail_lines: List[str],
    font: ImageFont.ImageFont,
    rng: random.Random,
) -> None:
    image = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    y = MARGIN

    lines = [
        "SYNTHETIC TRAINING SAMPLE - NOT A REAL BORROWER DOCUMENT",
        title,
        f"Case ID: {case_id}",
        f"Document Type: {doc_type}",
        f"Applicant Name: {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        f"Loan Reference: DEMO-{rng.randint(100000, 999999)}",
        f"Borrower Reference: SYN-{rng.randint(10000, 99999)}",
        f"Contact Phone: (202) 555-{rng.randint(1000, 1999)} (FICTIONAL)",
        f"Illustrative Monthly Income: ${rng.randint(3500, 12500):,} (FABRICATED)",
        f"SSN: NOT-A-REAL-SSN-{rng.randint(1000, 9999)} (INVALID PLACEHOLDER)",
        *detail_lines,
        f"Exception Category: {category}",
        f"Exception: {exception}",
        "Reviewer Note: Confirm the source document and resolve through approved review.",
    ]

    for index, line in enumerate(lines):
        wrapped_lines = textwrap.wrap(line, width=WRAP_WIDTH) or [""]
        for wrapped_line in wrapped_lines:
            draw.text((MARGIN, y), wrapped_line, fill="black", font=font)
            y += FONT_SIZE + LINE_SPACING
        if index == 0:
            y += 50
        elif index in {2, 4, 7 + len(detail_lines)}:
            y += 24

    image.save(output_path, dpi=(200, 200))


def main() -> None:
    app_dir = OUTPUT_ROOT / "1003_style_applications"
    ssn_dir = OUTPUT_ROOT / "ssn_verification_style"
    app_dir.mkdir(parents=True, exist_ok=True)
    ssn_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()
    rng = random.Random()

    for index, (category, exception, details) in enumerate(APPLICATION_EXCEPTIONS, start=1):
        draw_document(
            app_dir / f"synthetic_application_{index:02d}.png",
            "MORTGAGE APPLICATION - TRAINING MOCKUP",
            f"CASE-SYN-1003-{index:02d}",
            "Synthetic mortgage application summary (not official Form 1003)",
            category,
            exception,
            details,
            font,
            rng,
        )

    for index, (category, exception, details) in enumerate(SSN_EXCEPTIONS, start=1):
        draw_document(
            ssn_dir / f"synthetic_identity_verification_{index:02d}.png",
            "IDENTITY VERIFICATION - TRAINING MOCKUP",
            f"CASE-SYN-ID-{index:02d}",
            "Synthetic identity verification statement",
            category,
            exception,
            details,
            font,
            rng,
        )

    print(f"Generated {len(APPLICATION_EXCEPTIONS)} mortgage application mockups in {app_dir}")
    print(f"Generated {len(SSN_EXCEPTIONS)} identity verification mockups in {ssn_dir}")


if __name__ == "__main__":
    main()
