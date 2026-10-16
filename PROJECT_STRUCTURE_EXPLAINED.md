# Project Structure & Architecture: Explained Like You're 5
### A Beginner-Friendly Map of the Mortgage Exception Workbench

> **Welcome!** If you feel overwhelmed looking at all the Python files, databases, and AI models in this repository, **take a deep breath**.  
> This guide breaks down the entire system using simple, relatable language and zero jargon. By the end of this guide, you will know **what every single file does**, **where data travels**, and **where to put new code**.

---

## 1. The Big Picture: Think of it Like a High-End Restaurant

To understand how the files fit together, imagine this whole project is a **high-end restaurant with strict safety standards**:

```
                       ┌──────────────────────────────┐
                       │     THE DINING ROOM (UI)     │
                       │           app.py             │
                       └──────────────┬───────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │   THE BOUNCER   │       │   FOOD INTAKE   │       │ THE HEALTH INSP │
   │   src/auth.py   │       │ src/document_ocr│       │src/guardrails.py│
   │  (Checks ID &   │       │(Unpacks boxes & │       │(Checks for dirt │
   │   permissions)  │       │ reads labels)   │       │  & poison/PII)  │
   └─────────────────┘       └─────────────────┘       └─────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   THE HEAD CHEF & REGISTER   │
                       │   src/exception_workbench.py │
                       │ (Coordinates cooking & keeps │
                       │  the permanent paper trail)  │
                       └──────────────┬───────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      ┌──────────────────────┐                  ┌──────────────────────┐
      │   THE FOOD CRITIC    │                  │   THE STORYTELLER    │
      │   src/reranker.py    │                  │  src/rag_generation  │
      │(Tastes & ranks the 3 │                  │ (Writes nice notes   │
      │   best past dishes)  │                  │   for the diner)     │
      └──────────────────────┘                  └──────────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │      THE STORAGE PANTRY      │
                       │  workbench.db & data/ folder │
                       └──────────────────────────────┘
```

1. **`app.py` is the Dining Room & Waiter:** It's the screen you look at. It shows buttons, takes your orders, and displays results.
2. **`src/auth.py` is the Bouncer:** Stands at the front door. Are you a Manager (`reviewer`) who can change things, or just a Guest (`viewer`) who can only look?
3. **`src/document_ocr.py` is the Delivery Intake:** When a delivery truck drops off a paper document or scanned PDF, this component scans it and turns messy pictures into plain text.
4. **`src/guardrails.py` is the Health Inspector & Metal Detector:** Checks incoming text for toxic stuff (like credit card numbers, SSNs, or malicious hacking tricks) and blocks them.
5. **`src/exception_workbench.py` is the Head Chef & Official Ledger:** The brain of the whole operation. It talks to the database, manages the queue of work, and writes in an indelible ink diary (`case_event_log`) every time something changes.
6. **`src/reranker.py` is the Expert Food Critic:** When you ask, *"Have we seen a problem like this before?"*, this AI tastes the top 10 historical cases and ranks the absolute top 3 best matches.
7. **`src/rag_generation.py` is the Storyteller:** Takes those 3 past cases and types out an explanation on the screen token-by-token (streaming) so you don't have to wait.
8. **`workbench.db` & `data/` is the Storage Pantry:** The SQLite database and CSV files where past cases, current queues, and reviewer votes are safely locked up.
9. **`tests/` & `src/evals.py` is the Surprise Inspection Team:** Automated robot inspectors that run 50+ tests to make sure the kitchen never makes a mistake.

---

## 2. The Complete Folder Tree

Here is every file and folder in the project, organized by purpose:

```text
mortgage-exception-workbench/
│
├── app.py                      # 🖥️ FRONTEND: The Streamlit user interface
├── requirements.txt            # 📦 RECIPE: List of all Python packages needed
├── workbench.db                # 🗄️ DATABASE: The SQLite database file
├── .env.example                # 🔑 CONFIG: Example template for secrets & OAuth
│
├── src/                        # 🧠 THE ENGINE (All business logic lives here)
│   ├── auth.py                 # 1. Login, Google OAuth, and permissions
│   ├── document_ocr.py         # 2. Reading PDFs and images (PyMuPDF + Tesseract)
│   ├── guardrails.py           # 3. Security shields (PII redaction & prompt injection)
│   ├── exception_workbench.py  # 4. Core database CRUD, queue, and search orchestration
│   ├── reranker.py             # 5. Smart cross-encoder AI that ranks past cases
│   ├── rag_generation.py       # 6. AI text generator that streams explanations
│   └── evals.py                # 7. Automated exam that grades system accuracy
│
├── data/                       # 📁 DATA PANTRY (Seed data & vector indices)
│   ├── exception_queue.csv     # Starter seed cases loaded on first run
│   ├── past_exceptions.csv     # Library of historical approved cases
│   ├── incoming_exceptions.csv # Sample batch cases for testing
│   └── vector_store/           # ChromaDB folder where AI embeddings live
│
├── samples/                    # 📄 SAMPLE PAPERS (Images & mock mortgage forms)
│   ├── synthetic_income_document.png
│   ├── synthetic_mortgage_application.png
│   ├── synthetic_property_address.png
│   └── generated/              # 20 extra synthetic test forms
│
├── scripts/                    # 🛠️ UTILITY TOOLS (Helpful one-off helper scripts)
│   ├── create_sample_documents.py              # Makes the mock images in samples/
│   ├── generate_synthetic_document_samples.py  # Generates 20 extra fake forms
│   └── run_evals.py                            # Runs the benchmark test exam
│
└── tests/                      # 🧪 QUALITY CHECKS (Automated tests)
    ├── test_pack_integration.py     # 8 end-to-end integration tests
    ├── test_exception_workbench.py  # 26 unit tests for database and logic
    ├── test_guardrails.py           # 9 unit tests for PII and security shields
    └── test_evals.py                # 10 tests verifying the evaluation suite
```

---

## 3. Step-by-Step: What Happens When a User Clicks Something?

Follow this simple timeline to see how code flows across files:

```
[User clicks 'Extract document text' in app.py]
                   │
                   ▼
1. INTAKE: app.py calls src/document_ocr.py
   - Reads the PDF or image.
   - Extracts text like: "Borrower forgot to sign form 1003".
                   │
                   ▼
2. SECURITY SHIELD: app.py calls src/guardrails.py
   - Checks for real SSNs or credit card numbers -> replaces them with [REDACTED].
   - Checks for hacker tricks like "Ignore instructions" -> blocks if found.
                   │
                   ▼
3. SAVE TO QUEUE: app.py calls src/exception_workbench.py
   - Saves the case into SQLite table: `exception_queue`.
   - Records who did it and when in: `case_event_log`.
                   │
                   ▼
4. SEARCH PAST CASES: src/exception_workbench.py queries ChromaDB
   - Stage 1 (Fast Recall): Finds the top 10 most similar historical cases.
                   │
                   ▼
5. PRECISION RANKING: src/exception_workbench.py calls src/reranker.py
   - Stage 2 (Smart Re-rank): Cross-Encoder AI scores all 10 pairs.
   - Stage 3 (Feedback): Adds thumbs-up/down bias.
   - Returns the top 3 best matching cases.
                   │
                   ▼
6. STREAMING EXPLANATION: app.py calls src/rag_generation.py
   - Sends the 3 cases to Ollama (Llama 3.1).
   - Words stream onto the screen one-by-one so you don't have to wait.
                   │
                   ▼
7. REVIEWER FEEDBACK: Reviewer clicks 👍 or 👎
   - Saves vote into SQLite table: `suggestion_feedback`.
   - The AI will remember this vote next time!
```

---

## 4. "Where Do I Put My Code?" (The Golden Rules)

Whenever you want to build or change something, use this cheat sheet to know **which file to touch**:

| If you want to... | Edit this file |
| :--- | :--- |
| **Change how the screen looks, add buttons, or change colors** | [`app.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/app.py) or [`.streamlit/config.toml`](file:///Users/aryankhandelwal/projects/exception%20workbench/.streamlit/config.toml) |
| **Change who is allowed to log in or edit permissions** | [`src/auth.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/auth.py) |
| **Improve how scanned PDFs or photos are read** | [`src/document_ocr.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/document_ocr.py) |
| **Add a new database column, table, or change case statuses** | [`src/exception_workbench.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/exception_workbench.py) |
| **Tweak how past cases are ranked or change the thumbs up/down math** | [`src/reranker.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/reranker.py) |
| **Change the prompt given to the AI or switch LLM models** | [`src/rag_generation.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/rag_generation.py) |
| **Add a new security rule (e.g. block passport numbers or new jailbreaks)**| [`src/guardrails.py`](file:///Users/aryankhandelwal/projects/exception%20workbench/src/guardrails.py) |
| **Add a new automated test to prove your code works** | Add a test in [`tests/`](file:///Users/aryankhandelwal/projects/exception%20workbench/tests/) |

---

## 5. The Three Magic Commands You Need to Know

You only need to remember three terminal commands to run and test everything:

### 1. Run the app:
```bash
.venv/bin/streamlit run app.py
```
*(Open your browser at `http://localhost:8501`)*

### 2. Run all tests to make sure nothing is broken:
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

### 3. Run the automated exam report card:
```bash
python scripts/run_evals.py
```
