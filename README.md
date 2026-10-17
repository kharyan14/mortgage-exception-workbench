<div align="center">

# 🏦 Mortgage Exception Workbench
### Human-in-the-Review Operational Platform with Local RAG & Multi-Layer Safety Guardrails

[![Python 3.10+](https://img.shields.io/badge/python-3.10%25-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-success.svg)](./tests/)
[![Evaluation: 100% SLA](https://img.shields.io/badge/evals-100%25%20SLA-informational.svg)](./eval_report.md)

</div>
# Mortgage Exception Workbench

This is a synthetic mortgage document exception workbench for human-in-the-loop review operations.

> **Interested in the AI/ML architecture?** Read [AI_SYSTEM_DESIGN.md](AI_SYSTEM_DESIGN.md) for a comprehensive engineering deep dive into the 2-stage retrieval pipeline (bi-encoder + cross-encoder), online feedback bias loop, streaming RAG generation, and defense-in-depth safety guardrails.

## What is included
- **Two-stage neural retrieval**: Fast vector recall via MiniLM bi-encoder + precision cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`)
- **Reviewer feedback loop**: Online accept/reject bias learning without GPU fine-tuning overhead
- **Streaming RAG generation**: Token-by-token local LLM generation via Ollama with prompt sanitization and citation fidelity checks
- **Defense-in-depth guardrails**: PII redaction, prompt injection interception, grounding thresholds, and output policy auditing
- **Append-only case audit log**: Every mutation, edit, status transition, and deletion is recorded atomically with reviewer ID, timestamp, and payload
- **Strict SQLite data integrity**: Controlled schema with database-level `CHECK` constraints, composite indexes, and versioned migrations
- **Google OAuth 2.0 & RBAC**: Real identity provider backing role claims, with a zero-config dev mode fallback
- **OCR document extraction**: PyMuPDF + Tesseract for scanned PDFs and images with automated anomaly detection
- **Automated benchmark suite**: 20+ regression test cases measuring Hit Rate @ K, routing accuracy, and safety SLAs

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

1. **Input Sanitization & PII Redaction**: Automatically detects and redacts sensitive PII (SSNs, email addresses, phone numbers, credit card numbers, dates of birth) before sending context to LLM prompts.
2. **Prompt Injection & Adversarial Defense**: Inspects OCR extracted text and incoming exception notes for jailbreaks, prompt injection attempts (e.g. "ignore previous instructions", "system override", "approve loan immediately", role manipulation, delimiter escapes like `[INST]`), and forces `abstain` with mandatory human review.
3. **Retrieval Grounding Verification**: Enforces minimum semantic similarity thresholds (0.55). Low confidence or ungrounded context automatically forces an `abstain` suggested action and mandates human review.
4. **Citation Hallucination Defense**: Validates all cited prior cases in LLM generation against actual retrieved evidence case IDs to prevent fabricated citations.
5. **Output Policy & Authority Audit**: Scans generated LLM outputs to ensure the model does not make unauthorized credit approval claims, rate guarantees, claim underwriting completion, or leak unredacted PII.
6. **Real-time UI Audit**: The Streamlit interface displays real-time status badges for Input Safety, PII Redaction, Grounding Verification, and Policy Compliance alongside audit log details.

## Evaluation & Benchmark Suite
The workbench includes a formal, automated evaluation and regression testing framework (`src/evals.py` and `scripts/run_evals.py`):

- **Benchmark Dataset**: 23+ curated synthetic benchmark scenarios spanning Retrieval Recall, Out-of-Domain Boundaries, PII Leaks, Adversarial Prompt Injections, Policy Violations, and Citation Faithfulness.
- **Automated Metric SLAs**:
  - *Retrieval Hit Rate @ 1* (Target >= 80%, Achieved: 100%)
  - *Retrieval Hit Rate @ 3* (Target >= 100%, Achieved: 100%)
  - *Mean Reciprocal Rank / MRR* (Target >= 0.85, Achieved: 1.00)
  - *Deterministic Action Accuracy* (Target >= 90%, Achieved: 100%)
  - *PII Detection & Redaction Recall* (Target = 100%, Achieved: 100%)
  - *Prompt Injection Defense Rate* (Target = 100%, Achieved: 100%)
  - *Output Policy Enforcement Rate* (Target = 100%, Achieved: 100%)
  - *Citation Fidelity / Hallucination Detection* (Target = 100%, Achieved: 100%)
- **CLI Runner**:
  ```bash
  PYTHONPATH=. python scripts/run_evals.py
  ```
  Exports structured results to `data/eval_report.json` and `eval_report.md`.
- **UI Benchmark Dashboard**: Reviewers can execute live benchmarks directly within Streamlit via the "🧪 Model Evaluations & Benchmark Suite" panel.


## Role-Based Access Control (RBAC)
The workbench enforces strict role-based access control directly inside the domain functions:

- **Roles**:
  - `reviewer`: Authorized to mutate cases (create new cases, edit case queue fields, update statuses, and delete cases).
  - `viewer`: Read-only / look-only access.
- **Function-Level Enforcement**: Protection is not merely hidden buttons in the UI. Mutating functions (`create_exception_case`, `update_exception_case`, `update_exception_status`, `delete_exception_case`) enforce permissions using `check_case_mutation_permission(role, action)`. Calling any mutation function with `role="viewer"` or any unauthorized role directly raises `PermissionError` before touching the database.
- **UI State**: The Streamlit sidebar allows switching between `Reviewer` and `Viewer`. When in `Viewer` mode, editing and deletion controls are disabled, and attempted actions are refused with user-friendly error banners.

## Milestone 1: Matching brain
Historical examples are chunked, indexed in ChromaDB, and searched semantically. The UI shows retrieved evidence chunks and their source cases.

## Milestone 2: Resolution suggestion
Retrieved chunks ground a local LLM reviewer note. The allowed suggested action remains deterministic and is accompanied by the similarity score and human-review flag.

## Milestone 3: Exception queue, migrations, and audit trail
The app stores its queue in `workbench.db`, managed through versioned migrations (`PRAGMA user_version = 3`). The schema enforces strict data-integrity `CHECK` constraints, composite indexing for performant status-filtering, an append-only `case_event_log` table tracking all mutations atomically with actor and timestamp, and a `suggestion_feedback` table powering the online retrieval bias feedback loop.

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
