from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output
from pytesseract.pytesseract import TesseractNotFoundError

from src.exception_workbench import build_resolution_suggestion

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_DOCUMENTS = SUPPORTED_IMAGES | {".pdf"}
_DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "samples"
_FIELD_PATTERNS = {
    "case_id": r"^Case ID:\s*(.+)$",
    "document_type": r"^Document Type:\s*(.+)$",
    "borrower_id": r"^Borrower ID:\s*(.+)$",
    "borrower_reference": r"^Borrower Reference:\s*(.+)$",
    "ssn": r"^SSN:\s*(.+)$",
    "signature_status": r"^Signature Status:\s*(.+)$",
    "income_verification": r"^Income Verification:\s*(.+)$",
    "required_paystub": r"^Required Paystub:\s*(.+)$",
    "application_property_address": r"^Application Property Address:\s*(.+)$",
    "title_record_address": r"^Title Record Address:\s*(.+)$",
    "current_employer": r"^Current Employer:\s*(.+)$",
    "monthly_income": r"^Monthly Income:\s*(.+)$",
    "application_monthly_income": r"^Application Monthly Income:\s*(.+)$",
    "paystub_monthly_income": r"^Paystub Monthly Income:\s*(.+)$",
    "employment_start_date": r"^Employment Start Date:\s*(.+)$",
    "attachment_type": r"^Attachment Type:\s*(.+)$",
    "extraction_quality": r"^Extraction Quality:\s*(.+)$",
    "certification_signature": r"^Certification Signature:\s*(.+)$",
    "application_mailing_address": r"^Application Mailing Address:\s*(.+)$",
    "address_statement": r"^Address Statement:\s*(.+)$",
    "verification_document": r"^Verification Document:\s*(.+)$",
    "identifier": r"^Identifier:\s*(.+)$",
    "application_name": r"^Application Name:\s*(.+)$",
    "identity_statement_name": r"^Identity Statement Name:\s*(.+)$",
    "reviewer_signature": r"^Reviewer Signature:\s*(.+)$",
    "verification_statement": r"^Verification Statement:\s*(.+)$",
    "issue_date": r"^Issue Date:\s*(.+)$",
    "application_reference": r"^Application Reference:\s*(.+)$",
    "statement_reference": r"^Statement Reference:\s*(.+)$",
    "record_one_identifier": r"^Record One Identifier:\s*(.+)$",
    "record_two_identifier": r"^Record Two Identifier:\s*(.+)$",
    "checklist_identity_proof": r"^Checklist Identity Proof:\s*(.+)$",
    "attached_identity_proof": r"^Attached Identity Proof:\s*(.+)$",
    "reviewer_note": r"^Reviewer Note:\s*(.+)$",
    "exception_category": r"^Exception Category:\s*(.+)$",
    "required_document_status": r"^Required Document Status:\s*(.+)$",
    "monthly_gross_income": r"^Monthly Gross Income:\s*(.+)$",
    "appraisal_property_address": r"^Appraisal Property Address:\s*(.+)$",
    "exception": r"^Exception:\s*(.+)$",
}
_WRAPPABLE_FIELDS = {
    "document_type",
    "appraisal_property_address",
    "title_record_address",
    "application_mailing_address",
    "address_statement",
    "application_name",
    "identity_statement_name",
    "exception",
}


def _ocr_image(image: Image.Image) -> Tuple[str, Optional[float]]:
    try:
        result = pytesseract.image_to_data(image, output_type=Output.DICT)
    except TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR is required for scanned documents. Install it with "
            "`brew install tesseract` on macOS or your platform's package manager."
        ) from error

    text_lines: Dict[Tuple[str, str, str, str], list[str]] = {}
    confidences = []
    for index, value in enumerate(result["conf"]):
        word = result["text"][index].strip()
        if word:
            line_key = (
                result["block_num"][index],
                result["par_num"][index],
                result["line_num"][index],
                result["page_num"][index],
            )
            text_lines.setdefault(line_key, []).append(word)
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence / 100.0)

    text = "\n".join(" ".join(words) for words in text_lines.values())
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    return text, mean_confidence


def extract_document_text(path: str | Path) -> Dict[str, Any]:
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENTS:
        raise ValueError(f"Unsupported document format '{suffix}'. Use PDF or a supported image.")

    if suffix == ".pdf":
        page_text = []
        ocr_confidences = []
        methods = set()
        with fitz.open(document_path) as document:
            for page in document:
                embedded_text = page.get_text().strip()
                if embedded_text:
                    page_text.append(embedded_text)
                    methods.add("embedded_text")
                    continue

                pixmap = page.get_pixmap(dpi=240, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                recognized_text, confidence = _ocr_image(image)
                page_text.append(recognized_text)
                methods.add("ocr")
                if confidence is not None:
                    ocr_confidences.append(confidence)
        text = "\n\n".join(page_text).strip()
        method = "mixed" if len(methods) > 1 else next(iter(methods), "embedded_text")
    else:
        with Image.open(document_path) as image:
            text, confidence = _ocr_image(image.convert("RGB"))
        methods = {"ocr"}
        ocr_confidences = [confidence] if confidence is not None else []
        method = "ocr"

    if not text:
        raise ValueError(f"No readable text was found in {document_path}")
    return {
        "document_name": document_path.name,
        "text": text,
        "extraction_method": method,
        "ocr_confidence": (
            round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else None
        ),
    }


def extract_uploaded_document(file_name: str, content: bytes) -> Dict[str, Any]:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENTS:
        raise ValueError(f"Unsupported document format '{suffix}'. Use PDF or a supported image.")
    if not content:
        raise ValueError("The uploaded document is empty.")

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        extraction = extract_document_text(temporary_path)
        extraction["document_name"] = Path(file_name).name
        return extraction
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_document_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    last_field: Optional[str] = None
    for line in text.splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip()
        matched_field = False
        for field_name, pattern in _FIELD_PATTERNS.items():
            match = re.match(pattern, normalized_line, flags=re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip()
                last_field = field_name
                matched_field = True
                break
        if not matched_field and normalized_line and last_field in _WRAPPABLE_FIELDS:
            fields[last_field] = f"{fields[last_field]} {normalized_line}".strip()
    return fields


def describe_exception(fields: Dict[str, str], extracted_text: str) -> str:
    exception = fields.get("exception")
    if exception:
        return exception

    signature_status = fields.get("signature_status", "").lower()
    if signature_status in {"missing", "unsigned", "not provided"}:
        return "The borrower signature is missing from the mortgage application."

    document_status = fields.get("required_document_status", "").lower()
    if document_status in {"missing", "not provided", "not received"}:
        return "The required income verification document is missing from the loan file."

    appraisal_address = fields.get("appraisal_property_address")
    title_address = fields.get("title_record_address")
    if appraisal_address and title_address and appraisal_address.casefold() != title_address.casefold():
        return "The property address on the appraisal conflicts with the title record."

    return extracted_text


def review_document(path: str | Path, case_id: Optional[str] = None) -> Dict[str, Any]:
    extraction = extract_document_text(path)
    fields = extract_document_fields(extraction["text"])
    derived_case_id = case_id or fields.get("case_id") or "CASE-OCR-NEW"
    exception_text = describe_exception(fields, extraction["text"])
    if exception_text == extraction["text"]:
        exception_text = "No supported mortgage exception was detected in the extracted document fields."
        suggestion = {
            "case_id": derived_case_id,
            "exception_type": "unknown",
            "suggested_action": "abstain",
            "reason": "The extracted fields did not identify a supported exception; reviewer assessment is required.",
            "closest_case_id": "",
            "confidence": 0.0,
            "human_review_required": True,
        }
    else:
        suggestion = build_resolution_suggestion(derived_case_id, exception_text)
    return {
        "document_name": extraction["document_name"],
        "extraction_method": extraction["extraction_method"],
        "ocr_confidence": extraction["ocr_confidence"],
        "extracted_text": extraction["text"],
        "extracted_fields": fields,
        "exception_text": exception_text,
        "suggestion": suggestion,
    }


def review_sample_documents() -> list[Dict[str, Any]]:
    results = []
    for document_path in sorted(_DEFAULT_SAMPLES.glob("*.png")):
        results.append(review_document(document_path))
    if not results:
        raise FileNotFoundError(f"No PNG sample documents found in {_DEFAULT_SAMPLES}")
    return results


def main() -> None:
    if len(sys.argv) > 1:
        results = [review_document(path) for path in sys.argv[1:]]
    else:
        results = review_sample_documents()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
