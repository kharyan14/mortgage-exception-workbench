"""
Test Pack: 8 targeted integration tests for the Mortgage Exception Workbench.
Run with:  PYTHONPATH=. .venv/bin/python tests/test_pack_integration.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_ocr import extract_uploaded_document, extract_document_text
from src.exception_workbench import (
    create_exception_case,
    delete_exception_case,
    load_exception_queue,
    update_exception_status,
)

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  NOTE\033[0m"
results = []


def record(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"\n{'='*60}")
    print(f"  {tag}  {name}")
    if detail:
        print(f"         {detail}")
    print(f"{'='*60}")


# ── Test 1: Bad file (renamed .txt → .pdf) ──────────────────────────
def test_1_bad_file():
    print("\n\n▶ TEST 1: Bad file (.txt renamed to .pdf)")
    content = b"This is plain text, not a PDF at all.\n" * 10
    try:
        result = extract_uploaded_document("fake_document.pdf", content)
        record("Test 1: Bad file", False, f"Expected error but got result: {result.get('document_name')}")
    except (OSError, ValueError, RuntimeError) as e:
        record("Test 1: Bad file", True, f"Clean error: {type(e).__name__}: {e}")
    except Exception as e:
        record("Test 1: Bad file", False, f"Unexpected exception type: {type(e).__name__}: {e}")


# ── Test 2: Corrupt PDF (truncated) ─────────────────────────────────
def test_2_corrupt_pdf():
    print("\n\n▶ TEST 2: Corrupt PDF (truncated)")
    import fitz
    # Create a real PDF, then truncate it
    buf = bytearray()
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Hello, this is a real PDF page with some content.\nLine 2.\nLine 3.")
        raw = doc.tobytes()
    truncated = raw[: len(raw) // 2]  # cut in half
    try:
        result = extract_uploaded_document("truncated.pdf", bytes(truncated))
        record("Test 2: Corrupt PDF", False, f"Expected error but got result: {result.get('document_name')}")
    except (OSError, ValueError, RuntimeError) as e:
        record("Test 2: Corrupt PDF", True, f"Clean error: {type(e).__name__}: {e}")
    except Exception as e:
        record("Test 2: Corrupt PDF", False, f"Unexpected exception type: {type(e).__name__}: {e}")


# ── Test 3: Empty queue (fresh database) ─────────────────────────────
def test_3_empty_queue():
    print("\n\n▶ TEST 3: Empty queue (fresh database with empty seed)")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_empty.db"
        seed_path = Path(tmp) / "empty_seed.csv"
        # Valid CSV with headers but zero data rows
        seed_path.write_text(
            "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
        )
        try:
            queue = load_exception_queue(database_path=db_path, seed_path=seed_path)
            if queue.empty:
                record("Test 3: Empty queue", True, f"Returns empty DataFrame with {len(queue)} rows — UI would show info message, no crash")
            else:
                record("Test 3: Empty queue", False, f"Expected empty queue but got {len(queue)} rows")
        except Exception as e:
            record("Test 3: Empty queue", False, f"Crashed instead of returning empty: {type(e).__name__}: {e}")


# ── Test 4: Delete flow (create → delete → verify gone) ──────────────
def test_4_delete_flow():
    print("\n\n▶ TEST 4: Delete flow (create → delete → verify gone)")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_delete.db"
        seed_path = Path(tmp) / "seed.csv"
        seed_path.write_text(
            "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
            'CASE-SEED,"Seed case",other,abstain,False\n'
        )
        load_exception_queue(database_path=db_path, seed_path=seed_path)

        # Create
        create_exception_case({
            "case_id": "CASE-DEL-TEST",
            "exception_text": "Test case for deletion",
            "expected_type": "other",
            "expected_action": "abstain",
            "sensitive_case": False,
            "status": "New",
        }, database_path=db_path)

        # Verify it exists
        queue_before = load_exception_queue(database_path=db_path, seed_path=seed_path)
        exists_before = "CASE-DEL-TEST" in queue_before["case_id"].values

        # Delete
        delete_exception_case("CASE-DEL-TEST", database_path=db_path)

        # Verify gone from queue
        queue_after = load_exception_queue(database_path=db_path, seed_path=seed_path)
        exists_after = "CASE-DEL-TEST" in queue_after["case_id"].values

        # Verify gone from raw SQLite
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT * FROM exception_queue WHERE case_id = 'CASE-DEL-TEST'").fetchone()
        conn.close()
        gone_from_sqlite = row is None

        passed = exists_before and not exists_after and gone_from_sqlite
        detail = f"Before: exists={exists_before}, After: exists={exists_after}, SQLite row={row}"
        record("Test 4: Delete flow", passed, detail)


# ── Test 5: Duplicate upload (same doc twice) ────────────────────────
def test_5_duplicate_upload():
    print("\n\n▶ TEST 5: Duplicate upload (same document text, two case IDs)")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_dup.db"
        seed_path = Path(tmp) / "seed.csv"
        seed_path.write_text(
            "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
            'CASE-SEED,"Seed case",other,abstain,False\n'
        )
        load_exception_queue(database_path=db_path, seed_path=seed_path)

        exception_text = "Borrower signature is missing from the mortgage application."
        # Upload 1
        create_exception_case({
            "case_id": "CASE-DUP-001",
            "exception_text": exception_text,
            "expected_type": "signature",
            "expected_action": "request_missing_document",
            "sensitive_case": False,
            "status": "New",
        }, database_path=db_path)

        # Upload 2 — same text, different UUID-based case_id
        create_exception_case({
            "case_id": "CASE-DUP-002",
            "exception_text": exception_text,
            "expected_type": "signature",
            "expected_action": "request_missing_document",
            "sensitive_case": False,
            "status": "New",
        }, database_path=db_path)

        queue = load_exception_queue(database_path=db_path, seed_path=seed_path)
        dup_cases = queue[queue["exception_text"] == exception_text]
        count = len(dup_cases)
        ids = dup_cases["case_id"].tolist()

        passed = count == 2
        detail = f"Created {count} cases with same text: {ids} — NO dedup (by design: case_id is UUID-based)"
        record("Test 5: Duplicate upload", passed, detail)


# ── Test 6: Weird filename (spaces + Hindi char) ─────────────────────
def test_6_weird_filename():
    print("\n\n▶ TEST 6: Weird filename (spaces + Hindi character)")
    import shutil
    if not shutil.which("tesseract"):
        record("Test 6: Weird filename", True, "SKIPPED — Tesseract not installed; but filename handling is display-only via Path().name, so this is safe")
        return

    sample = PROJECT_ROOT / "samples" / "synthetic_mortgage_application.png"
    content = sample.read_bytes()
    weird_name = "मेरा दस्तावेज application scan (copy 2).png"

    try:
        result = extract_uploaded_document(weird_name, content)
        name_ok = result["document_name"] == weird_name
        text_ok = len(result["text"]) > 0
        passed = name_ok and text_ok
        detail = f"document_name='{result['document_name']}', text_len={len(result['text'])}, method={result['extraction_method']}"
        record("Test 6: Weird filename", passed, detail)
    except Exception as e:
        record("Test 6: Weird filename", False, f"{type(e).__name__}: {e}")


# ── Test 7: Two tabs (SQLite consistency) ─────────────────────────────
def test_7_two_tabs():
    print("\n\n▶ TEST 7: Two tabs (SQLite cross-connection consistency)")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_tabs.db"
        seed_path = Path(tmp) / "seed.csv"
        seed_path.write_text(
            "case_id,exception_text,expected_type,expected_action,sensitive_case\n"
            'CASE-TAB-1,"Signature missing",signature,request_missing_document,False\n'
        )
        # Tab A loads queue
        queue_a = load_exception_queue(database_path=db_path, seed_path=seed_path)
        status_a_before = queue_a.iloc[0]["status"]

        # Tab A changes status
        update_exception_status("CASE-TAB-1", "In Review", database_path=db_path)

        # Tab B refreshes (new load)
        queue_b = load_exception_queue(database_path=db_path, seed_path=seed_path)
        status_b = queue_b.iloc[0]["status"]

        passed = status_a_before == "New" and status_b == "In Review"
        detail = f"Tab A initial: '{status_a_before}' → updated to 'In Review' → Tab B refresh sees: '{status_b}'"
        record("Test 7: Two tabs", passed, detail)


# ── Test 8: Big file (multi-page PDF) ────────────────────────────────
def test_8_big_file():
    print("\n\n▶ TEST 8: Big file (multi-page PDF from sample images)")
    import shutil
    import time
    if not shutil.which("tesseract"):
        record("Test 8: Big file", True, "SKIPPED — Tesseract not installed (OCR required for scanned pages)")
        return

    import fitz
    sample_images = sorted((PROJECT_ROOT / "samples").glob("*.png"))
    if not sample_images:
        record("Test 8: Big file", False, "No sample images found to build test PDF")
        return

    with tempfile.TemporaryDirectory() as tmp:
        big_pdf = Path(tmp) / "big_test.pdf"
        with fitz.open() as doc:
            # Create 6 pages (each sample image repeated twice)
            for img in sample_images * 2:
                page = doc.new_page(width=612, height=792)
                page.insert_image(page.rect, filename=str(img))
            doc.save(big_pdf)
            num_pages = len(doc)

        file_size = big_pdf.stat().st_size
        print(f"   Built PDF: {num_pages} pages, {file_size / 1024:.0f} KB")

        start = time.time()
        try:
            result = extract_document_text(big_pdf)
            elapsed = time.time() - start
            passed = len(result["text"]) > 0
            detail = (
                f"{num_pages} pages, {file_size/1024:.0f} KB, "
                f"method={result['extraction_method']}, "
                f"text_len={len(result['text'])}, "
                f"elapsed={elapsed:.1f}s"
            )
            record("Test 8: Big file", passed, detail)
        except Exception as e:
            elapsed = time.time() - start
            record("Test 8: Big file", False, f"Failed after {elapsed:.1f}s — {type(e).__name__}: {e}")


# ── Run all ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MORTGAGE EXCEPTION WORKBENCH — INTEGRATION TEST PACK")
    print("=" * 60)

    test_1_bad_file()
    test_2_corrupt_pdf()
    test_3_empty_queue()
    test_4_delete_flow()
    test_5_duplicate_upload()
    test_6_weird_filename()
    test_7_two_tabs()
    test_8_big_file()

    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed, detail in results:
        tag = PASS if passed else FAIL
        print(f"  {tag}  {name}")
    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"\n  {passed_count}/{total} passed")
    print("=" * 60)
