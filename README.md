# Mortgage Exception Workbench

This is a fresh, synthetic mortgage document exception workbench.

## What is included
- Semantic matching using sentence embeddings and cosine similarity
- Historical mortgage exception dataset with approved resolutions
- Resolution suggestion engine with a strict JSON output shape
- OCR/text extraction for scanned images and PDF documents
- Synthetic scanned application, income-document checklist, and property-address comparison samples
- Field extraction and end-to-end document-to-suggestion intake
- Streamlit document upload connected to OCR and case creation
- SQLite-backed create, read, update, and delete operations for queue cases
- Persistent exception queue with New and In Review statuses
- Streamlit UI for queue browsing, case selection, semantic matches, and suggestions
- Simple validation tests for matching and routing behavior

## OCR document intake
Scanned PNG/JPEG/TIFF/BMP/WebP and PDFs are accepted. PDFs with an embedded text layer use that text; scanned PDF pages are rendered locally and passed through Tesseract OCR. OCR returns the recognized text, extraction method, an indicative OCR confidence, and labeled fields when present. The extracted exception is then passed to the same matching and suggestion flow.

Tesseract must be installed separately for image and scanned-PDF OCR:

```bash
brew install tesseract
```

Run the three synthetic scanned mortgage document samples through OCR and matching:

```bash
python -m src.document_ocr
```

Or review specific files:

```bash
python -m src.document_ocr samples/synthetic_mortgage_application.png
```

In the Streamlit workbench, upload a supported PDF or image scan, extract its text, review/edit the detected exception, choose its type and proposed action, then add it to the SQLite queue. Queue rows can also be edited and deleted from the app. Uploaded documents are processed temporarily for OCR; the document file itself is not stored in SQLite.

Regenerate the synthetic samples if needed:

```bash
python scripts/create_sample_documents.py
```

Generate 10 additional mortgage-application-style and 10 identity-verification-style OCR samples, each labeled with one of the prototype exception categories:

```bash
python scripts/generate_synthetic_document_samples.py
```

The generated documents are original training mockups, not reproductions of the official Uniform Residential Loan Application (Form 1003). Applicant names and demo references/amounts are randomly generated; contact numbers use a reserved fictional range. SSN fields contain an explicitly invalid placeholder, never a numeric SSN.

OCR confidence is a text-recognition signal only, not confidence that extracted fields or suggested actions are correct. All examples contain fabricated borrower identifiers and addresses; do not use real borrower documents in this prototype.

## RAG architecture
The project splits historical exceptions, approved resolutions, and supporting reviewer notes into separate overlapping word chunks. SentenceTransformer embeds each chunk and query. ChromaDB stores chunk text, vectors, and case/source metadata persistently under `data/vector_store/`. At lookup, the query vector searches Chroma with cosine distance; the app ranks cases by their best evidence chunk and displays the retrieved text.

The retrieved chunks and approved resolutions are passed to a local Ollama model (`llama3.1:latest`) to generate a grounded reviewer note. This is the generation step in RAG. A separate deterministic Python rule still selects the allowed next action and applies the human-review threshold; the LLM cannot resolve cases or make credit/underwriting decisions.

Install and start Ollama, and pull the model if it is not already installed:

```bash
ollama pull llama3.1
```

Set `WORKBENCH_LLM_MODEL` or `OLLAMA_HOST` to use a different local Ollama model/server. If Ollama is unavailable, the UI reports a generation error instead of pretending generated text exists. The Chroma index is persistent and can be rebuilt from `data/past_exceptions.csv`.

Chunk size and overlap are configured by `DEFAULT_CHUNK_SIZE` and `DEFAULT_CHUNK_OVERLAP` in `src/exception_workbench.py` (80 words, 20-word overlap).

## Safety & Policy Guardrails
The system includes multi-layer AI and operational guardrails (`src/guardrails.py`) to safeguard inputs, RAG generation, and automated resolution suggestion:

1. **Input Sanitization & PII Redaction**: Automatically detects and redacts sensitive PII (SSNs, email addresses, phone numbers) before sending context to LLM prompts.
2. **Prompt Injection & Adversarial Defense**: Inspects OCR extracted text and incoming exception notes for jailbreak or prompt injection attempts (e.g. "ignore previous instructions", "system override", "approve loan immediately") and blocks ungrounded action suggestions.
3. **Retrieval Grounding Verification**: Enforces minimum semantic similarity thresholds (0.55). Low confidence or ungrounded context automatically forces an `abstain` suggested action and mandates human review.
4. **Output Policy & Authority Audit**: Scans generated LLM outputs to ensure the model does not make unauthorized credit approval claims, claim underwriting completion, or leak unredacted PII.
5. **Real-time UI Audit**: The Streamlit interface displays real-time status badges for Input Safety, PII Redaction, Grounding Verification, and Policy Compliance alongside audit log details.

## Milestone 1: Matching brain
Historical examples are chunked, indexed in ChromaDB, and searched semantically. The UI shows retrieved evidence chunks and their source cases.

## Milestone 2: Resolution suggestion
Retrieved chunks ground a local LLM reviewer note. The allowed suggested action remains deterministic and is accompanied by the similarity score and human-review flag.

## Milestone 3: Exception queue and UI
The app stores its queue in `workbench.db`, using one SQLite table named `exception_queue`. On first run, if that table is empty, it is seeded once from `data/exception_queue.csv`; after seeding, SQLite is the source of truth and the CSV is not read again. The app loads queue rows with pandas `read_sql`. Its CRUD flow creates cases from reviewed OCR text, reads and filters queue cases, updates editable fields with SQL `UPDATE`, and deletes a selected case with SQL `DELETE`. Setting a status to Resolved changes the queue only; it does not yet record approval, edits/rejections, or an audit log. Add those sign-off behaviors before treating that status as a human-approved resolution.

Start the UI:

```bash
streamlit run app.py
```

The UI loads the sentence-transformer model on first match; that initial startup may download the model if it is not cached locally.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 -m src.exception_workbench
python3 -m src.document_ocr
streamlit run app.py
```

## Notes
This version is intentionally limited to synthetic mortgage-document data and a human-in-the-loop safety rule.
