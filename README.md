
<div align="center">

# 🏦 Mortgage Exception Workbench
### Human-in-the-Review Operational Platform with Local RAG & Multi-Layer Safety Guardrails

[![Python 3.10+](https://img.shields.io/badge/python-3.10%25-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-success.svg)](./tests/)
[![Evaluation: 100% SLA](https://img.shields.io/badge/evals-100%25%20SLA-informational.svg)](./eval_report.md)

</div>

---

## 🧭 Overview

The **Mortgage Exception Workbench** is a production-grade synthetic document review and exception-handling platform built for mortgage operations. It combines advanced **Two-Stage Neural Information Retrieval** with a local Large Language Model (`Llama 3.1` via Ollama) to assist human underwriters in resolving document exceptions quickly and accurately—backed by strict data integrity, audit logs, and multi-layer safety guardrails.

> **Architecture Deep Dive:** For a complete engineering breakdown of the retrieval pipeline, feedback loop, and safety mechanisms, read [AI_SYSTEM_DESIGN.md](AI_SYSTEM_DESIGN.md).

---

## 🚀 Key Technical Features

* **Two-Stage Neural Retrieval:** Fast vector recall via MiniLM bi-encoder + precision cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) backed by ChromaDB.
* **Defense-in-Depth Guardrails (`src/guardrails.py`):** Automated PII redaction, prompt injection interception, confidence grounding thresholds (< 0.55), and citation fidelity auditing.
* **Append-Only Audit Log (`workbench.db`):** Atomic tracking of every mutation, status transition, and deletion with actor metadata and versioned migrations.
* **Strict Role-Based Access Control (RBAC):** Function-level permission enforcement preventing `viewer` roles from mutating state (`PermissionError`).
* **OCR Document Extraction:** PyMuPDF + Tesseract integration for scanned PDFs and images with automated anomaly handling.
* **Automated Benchmark Suite:** 20+ regression test cases validating retrieval metrics and safety SLAs (view results in [eval_report.md](eval_report.md)).

---

## ⚙️ Quick Start

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) with Llama 3.1 pulled:
  ```bash
  ollama pull llama3.1

```

* Tesseract OCR installed (for scanned document processing):
```bash
brew install tesseract

```



### Installation & Execution

```bash
# Clone and install dependencies
git clone [https://github.com/kharyan14/mortgage-exception-workbench.git](https://github.com/kharyan14/mortgage-exception-workbench.git)
cd mortgage-exception-workbench
python3 -m pip install -r requirements.txt

# Run test suite and launch Workbench UI
python3 -m pytest -q
streamlit run app.py

```

---

## 📂 Project Structure

* **`src/`**: Core logic (Retrieval, OCR, Guardrails, RBAC, Database schema).
* **`tests/`**: Automated test suite and regression checks.
* **`scripts/`**: Data generation and benchmark runner scripts.
* **`samples/`**: Synthetic mortgage document mockups.

---

## 🛡️ License

Distributed under the MIT License. See LICENSE for details.

```

```
