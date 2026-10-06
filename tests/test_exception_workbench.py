import shutil
from pathlib import Path

import fitz
import pandas as pd
import pytest

from src.document_ocr import (
    describe_exception,
    extract_document_fields,
    extract_document_text,
    extract_uploaded_document,
)
from src.exception_workbench import (
    build_resolution_suggestion,
    _index_retrieval_chunks,
    create_exception_case,
    _build_retrieval_chunks,
    delete_exception_case,
    chunk_text,
    find_similar_cases,
    load_exception_queue,
    update_exception_case,
    update_exception_status,
)
from src.rag_generation import generate_grounded_response


def test_signature_case_returns_matching_theme():
    matches = find_similar_cases("The borrower forgot to sign the form.", top_k=3)
    assert matches.iloc[0]["exception_type"] == "signature"
    assert list(matches["case_id"].head(2)) == ["CASE-002", "CASE-001"]
    assert matches.iloc[0]["similarity"] > matches.iloc[2]["similarity"]
    assert matches.iloc[0]["matched_chunk"]


def test_chunk_text_splits_long_text_with_overlap():
    words = [f"word{index}" for index in range(11)]
    chunks = chunk_text(" ".join(words), chunk_size=5, overlap=2)

    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7 word8 word9 word10",
    ]


def test_chunk_text_rejects_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=5, overlap=5)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=0)


def test_vector_database_persists_chunk_embeddings_and_metadata(tmp_path):
    cases = pd.DataFrame(
        [
            {
                "case_id": "CASE-VECTOR-1",
                "exception_text": "Borrower signature is missing.",
                "approved_resolution": "Request the signed form.",
                "supporting_evidence": "Reviewer confirmed the signature field was empty.",
            }
        ]
    )
    chunks = _build_retrieval_chunks(cases)
    collection = _index_retrieval_chunks(chunks, persist_directory=tmp_path / "chroma")
    assert collection.count() == len(chunks)

    reopened_collection = _index_retrieval_chunks(chunks, persist_directory=tmp_path / "chroma")
    result = reopened_collection.get(include=["documents", "metadatas"])
    assert reopened_collection.count() == len(chunks)
    assert set(result["metadatas"][0]) >= {"case_id", "source", "content_hash"}
    assert any("signature" in document.lower() for document in result["documents"])


def test_generated_rag_response_uses_retrieved_evidence(monkeypatch):
    import json

    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"response": "Prior case CASE-001 supports requesting a signature."}).encode()

    def fake_urlopen(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.rag_generation.urlopen", fake_urlopen)
    answer = generate_grounded_response(
        "Borrower signature is missing.",
        [("CASE-001", "Exception: Signature field is blank.", "Request the signed page.")],
        model="test-model",
        ollama_url="http://localhost:11434",
    )

    payload = json.loads(observed["request"].data.decode())
    assert payload["model"] == "test-model"
    assert "Exception: Signature field is blank." in payload["prompt"]
    assert "CASE-001" in payload["prompt"]
    assert answer.startswith("Prior case CASE-001")


def test_resolution_suggestion_has_expected_shape():
    suggestion = build_resolution_suggestion(
        case_id="CASE-101",
        exception_text="The borrower forgot to sign the form.",
        sensitive_case=False,
    )

    assert set(suggestion.keys()) == {
        "case_id",
        "exception_type",
        "suggested_action",
        "reason",
        "closest_case_id",
        "confidence",
        "human_review_required",
    }
    assert suggestion["suggested_action"] in {"request_missing_document", "propose_field_correction", "route_to_specialist", "abstain"}
    assert suggestion["human_review_required"] is True or suggestion["confidence"] < 0.80


def test_unrelated_case_abstains_or_requires_human_review():
    suggestion = build_resolution_suggestion(
        case_id="CASE-999",
        exception_text="The escrow workflow has an external tax exception not represented in these synthetic examples.",
        sensitive_case=False,
    )

    assert suggestion["suggested_action"] in {"abstain", "route_to_specialist"}
    assert suggestion["human_review_required"] is True


def test_extracts_labeled_fields_from_ocr_text():
    text = """MORTGAGE APPLICATION REVIEW
Case ID: CASE-OCR-201
Document Type: Mortgage Application
Borrower ID: SYNTHETIC-BORROWER-001
Signature Status: MISSING
"""
    fields = extract_document_fields(text)

    assert fields["case_id"] == "CASE-OCR-201"
    assert fields["document_type"] == "Mortgage Application"
    assert fields["signature_status"] == "MISSING"
    assert describe_exception(fields, text) == (
        "The borrower signature is missing from the mortgage application."
    )


def test_detects_address_conflict_from_extracted_fields():
    fields = extract_document_fields(
        """Appraisal Property Address: 123 Example Avenue, Sampletown, CA 90001
Title Record Address: 123 Example Avenue, Sampletown, CA 90010"""
    )

    assert describe_exception(fields, "") == (
        "The property address on the appraisal conflicts with the title record."
    )


def test_appends_ocr_wrapped_exception_lines_to_field():
    fields = extract_document_fields(
        """Exception: The recent paystub required for income verification
is missing from the loan file."""
    )

    assert fields["exception"] == (
        "The recent paystub required for income verification is missing from the loan file."
    )


def test_ocr_extracts_text_from_scanned_pdf(tmp_path):
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is required for scanned-PDF OCR")

    sample_image = Path(__file__).resolve().parents[1] / "samples" / "synthetic_mortgage_application.png"
    scanned_pdf = tmp_path / "scanned-application.pdf"
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        page.insert_image(page.rect, filename=str(sample_image))
        document.save(scanned_pdf)

    result = extract_document_text(scanned_pdf)

    assert result["extraction_method"] == "ocr"
    assert "Signature Status: MISSING" in result["text"]


def test_uploaded_sample_document_extracts_fields_and_exception():
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is required for scanned-image OCR")

    sample_image = Path(__file__).resolve().parents[1] / "samples" / "synthetic_mortgage_application.png"
    extraction = extract_uploaded_document(sample_image.name, sample_image.read_bytes())
    fields = extract_document_fields(extraction["text"])

    assert extraction["document_name"] == sample_image.name
    assert fields["case_id"] == "CASE-OCR-201"
    assert describe_exception(fields, extraction["text"]) == (
        "The borrower signature is missing from the mortgage application."
    )


def test_uploaded_document_can_be_added_to_sqlite_queue(tmp_path):
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is required for scanned-image OCR")

    seed_path = tmp_path / "seed.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-SEED,"Seed case",other,abstain,False\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)

    sample_image = Path(__file__).resolve().parents[1] / "samples" / "synthetic_mortgage_application.png"
    extraction = extract_uploaded_document(sample_image.name, sample_image.read_bytes())
    fields = extract_document_fields(extraction["text"])
    exception_text = describe_exception(fields, extraction["text"])
    create_exception_case(
        {
            "case_id": "CASE-UPLOAD-TEST",
            "exception_text": exception_text,
            "expected_type": "signature",
            "expected_action": "request_missing_document",
            "sensitive_case": False,
            "status": "New",
        },
        database_path=database_path,
    )

    queue = load_exception_queue(database_path=database_path, seed_path=seed_path).set_index("case_id")
    assert queue.loc["CASE-UPLOAD-TEST", "exception_text"] == (
        "The borrower signature is missing from the mortgage application."
    )
    assert queue.loc["CASE-UPLOAD-TEST", "status"] == "New"


def test_all_generated_samples_ocr_with_exception_category():
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is required for scanned-image OCR")

    samples_root = Path(__file__).resolve().parents[1] / "samples" / "generated"
    documents = sorted(samples_root.rglob("*.png"))
    assert len(documents) == 20
    categories = set()

    for document in documents:
        extraction = extract_document_text(document)
        fields = extract_document_fields(extraction["text"])
        assert fields.get("exception"), f"Exception not extracted from {document.name}"
        assert fields.get("exception_category"), f"Exception category not extracted from {document.name}"
        categories.add(fields["exception_category"].lower().replace(" ", "_"))

    assert categories == {
        "missing_document",
        "conflicting_values",
        "low_confidence_extraction",
        "unsupported_extraction",
    }


def test_queue_is_seeded_from_csv_and_status_persists_in_sqlite(tmp_path):
    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-301,"Missing borrower signature",signature,request_missing_document,False\n'
    )

    seeded = load_exception_queue(database_path=database_path, seed_path=seed_path)
    assert seeded.iloc[0]["status"] == "New"

    update_exception_status("CASE-301", "In Review", database_path=database_path)
    reloaded = load_exception_queue(database_path=database_path, seed_path=seed_path)
    assert reloaded.iloc[0]["status"] == "In Review"


def test_queue_seeds_only_when_sqlite_table_is_empty(tmp_path):
    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-302,"Missing borrower signature",signature,request_missing_document,False\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)

    update_exception_status("CASE-302", "Resolved", database_path=database_path)
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-303,"New row in changed CSV",income,request_missing_document,True\n'
    )

    reloaded = load_exception_queue(database_path=database_path, seed_path=seed_path)
    assert reloaded["case_id"].tolist() == ["CASE-302"]
    assert reloaded.iloc[0]["status"] == "Resolved"


def test_sqlite_queue_has_only_specified_columns(tmp_path):
    import sqlite3

    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-304,"Missing borrower signature",signature,request_missing_document,True\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)

    connection = sqlite3.connect(database_path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(exception_queue)").fetchall()
    finally:
        connection.close()

    assert tables == [("exception_queue",)]
    assert [(column[1], column[2]) for column in columns] == [
        ("case_id", "TEXT"),
        ("exception_text", "TEXT"),
        ("expected_type", "TEXT"),
        ("expected_action", "TEXT"),
        ("sensitive_case", "INTEGER"),
        ("status", "TEXT"),
    ]


def test_status_update_rejects_invalid_status_and_unknown_case(tmp_path):
    import sqlite3

    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-305,"Missing borrower signature",signature,request_missing_document,True\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)

    with pytest.raises(ValueError, match="Unsupported queue status"):
        update_exception_status("CASE-305", "Waiting", database_path=database_path)
    with pytest.raises(KeyError, match="CASE-999"):
        update_exception_status("CASE-999", "Resolved", database_path=database_path)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT status FROM exception_queue WHERE case_id = 'CASE-305'"
        ).fetchone()[0] == "New"
    finally:
        connection.close()


def test_sqlite_queue_supports_create_update_and_delete(tmp_path):
    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-306,"Missing borrower signature",signature,request_missing_document,False\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)
    create_exception_case(
        {
            "case_id": "CASE-CRUD",
            "exception_text": "The property address conflicts with the title record.",
            "expected_type": "address",
            "expected_action": "propose_field_correction",
            "sensitive_case": True,
            "status": "New",
        },
        database_path=database_path,
    )

    update_exception_case(
        "CASE-CRUD",
        {
            "exception_text": "The appraisal address conflicts with the title record.",
            "status": "In Review",
            "sensitive_case": False,
        },
        database_path=database_path,
    )
    updated_case = load_exception_queue(database_path=database_path, seed_path=seed_path).set_index("case_id").loc["CASE-CRUD"]
    assert updated_case["status"] == "In Review"
    assert not bool(updated_case["sensitive_case"])
    assert updated_case["exception_text"] == "The appraisal address conflicts with the title record."

    delete_exception_case("CASE-CRUD", database_path=database_path)
    remaining_cases = load_exception_queue(database_path=database_path, seed_path=seed_path)
    assert "CASE-CRUD" not in remaining_cases["case_id"].tolist()


def test_delete_unknown_case_reports_error(tmp_path):
    seed_path = tmp_path / "incoming.csv"
    database_path = tmp_path / "workbench.db"
    seed_path.write_text(
        "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        'CASE-307,"Missing borrower signature",signature,request_missing_document,False\n'
    )
    load_exception_queue(database_path=database_path, seed_path=seed_path)

    with pytest.raises(KeyError, match="CASE-999"):
        delete_exception_case("CASE-999", database_path=database_path)
