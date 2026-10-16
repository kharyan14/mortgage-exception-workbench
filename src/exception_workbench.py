from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PAST_CASES = _PROJECT_ROOT / "data" / "past_exceptions.csv"
_DEFAULT_INCOMING_CASES = _PROJECT_ROOT / "data" / "incoming_exceptions.csv"
_DEFAULT_QUEUE_CSV = _PROJECT_ROOT / "data" / "exception_queue.csv"
_DEFAULT_DATABASE = _PROJECT_ROOT / "workbench.db"
_DEFAULT_VECTOR_STORE = _PROJECT_ROOT / "data" / "vector_store"

QUEUE_STATUSES = ("New", "In Review", "Resolved")

# Controlled vocabulary enforced by both Python and SQLite CHECK constraints.
# expected_type and expected_action are *reviewer labels*, not model outputs.
EXCEPTION_TYPES = (
    "signature",
    "income",
    "address",
    "missing_document",
    "conflicting_values",
    "low_confidence_extraction",
    "other",
)
EXCEPTION_ACTIONS = (
    "request_missing_document",
    "propose_field_correction",
    "route_to_specialist",
    "abstain",
)

QUEUE_COLUMNS = (
    "case_id",
    "exception_text",
    "expected_type",
    "expected_action",
    "sensitive_case",
    "status",
)
ROLE_REVIEWER = "reviewer"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_REVIEWER, ROLE_VIEWER)
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CHUNK_OVERLAP = 20

# ── Schema versioning ────────────────────────────────────────────────────────
# Increment _SCHEMA_VERSION each time the DDL changes.  The migration runner
# (below) applies numbered migration functions in order so existing workbench.db
# files are evolved without manual drops or re-seeds.
_SCHEMA_VERSION = 3

_model: Optional[SentenceTransformer] = None


def check_case_mutation_permission(role: str, action: str = "modify") -> None:
    """Enforces role-based access control. Only 'reviewer' can mutate cases.

    Raises:
        PermissionError: If the role is 'viewer' or any unauthorized role.
    """
    normalized_role = str(role or "").strip().lower()
    if normalized_role != ROLE_REVIEWER:
        raise PermissionError(
            f"Permission denied: role '{role}' is not authorized to {action} cases. "
            f"Only '{ROLE_REVIEWER}' can change cases; '{ROLE_VIEWER}' is read-only."
        )


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_past_cases(path: Optional[str | Path] = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else _DEFAULT_PAST_CASES
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"No past case data found in {data_path}")
    return df


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_lengths = np.linalg.norm(a, axis=1, keepdims=True)
    b_lengths = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.divide(a, a_lengths, out=np.zeros_like(a), where=a_lengths != 0)
    b_norm = np.divide(b, b_lengths, out=np.zeros_like(b), where=b_lengths != 0)
    return np.einsum("ij,kj->ik", a_norm, b_norm, optimize=False)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def _build_retrieval_chunks(cases: pd.DataFrame) -> list[dict[str, str]]:
    chunks = []
    for _, case in cases.iterrows():
        supporting_evidence = str(case.get("supporting_evidence", "") or "")
        sources = (
            ("exception", f"Exception: {case['exception_text']}"),
            ("approved_resolution", f"Approved resolution: {case['approved_resolution']}"),
            ("supporting_evidence", f"Supporting evidence: {supporting_evidence}"),
        )
        for source_name, source_text in sources:
            if not source_text.split(":", 1)[1].strip():
                continue
            for chunk_index, text in enumerate(chunk_text(source_text)):
                chunks.append(
                    {
                        "case_id": str(case["case_id"]),
                        "text": text,
                        "chunk_id": f"{case['case_id']}:{source_name}:{chunk_index + 1}",
                        "source": source_name,
                    }
                )
    return chunks


def _get_vector_collection(
    persist_directory: Optional[str | Path] = None,
) -> chromadb.Collection:
    directory = Path(persist_directory) if persist_directory is not None else _DEFAULT_VECTOR_STORE
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(directory))
    return client.get_or_create_collection(
        name="mortgage_exception_chunks",
        metadata={"hnsw:space": "cosine"},
    )


def _index_retrieval_chunks(
    chunks: list[dict[str, str]],
    persist_directory: Optional[str | Path] = None,
) -> chromadb.Collection:
    collection = _get_vector_collection(persist_directory)
    desired_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing["ids"])
    desired_ids = set(desired_by_id)

    obsolete_ids = existing_ids - desired_ids
    if obsolete_ids:
        collection.delete(ids=sorted(obsolete_ids))

    existing_metadata = {
        chunk_id: metadata or {}
        for chunk_id, metadata in zip(existing["ids"], existing["metadatas"] or [])
    }
    changed_chunks = []
    for chunk_id, chunk in desired_by_id.items():
        content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        if existing_metadata.get(chunk_id, {}).get("content_hash") != content_hash:
            changed_chunks.append((chunk, content_hash))

    if changed_chunks:
        model = get_model()
        embeddings = model.encode(
            [chunk["text"] for chunk, _ in changed_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk, _ in changed_chunks],
            documents=[chunk["text"] for chunk, _ in changed_chunks],
            embeddings=np.asarray(embeddings, dtype=float).tolist(),
            metadatas=[
                {
                    "case_id": chunk["case_id"],
                    "source": chunk["source"],
                    "content_hash": content_hash,
                }
                for chunk, content_hash in changed_chunks
            ],
        )
    return collection


def find_similar_cases(
    new_exception: str,
    top_k: int = 3,
    past_cases_path: Optional[str | Path] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    vector_store_path: Optional[str | Path] = None,
    database_path: Optional[str | Path] = None,
    use_reranker: bool = True,
) -> pd.DataFrame:
    """Retrieve and re-rank similar past cases for a new exception.

    Pipeline:
    1. Bi-encoder (MiniLM) approximate recall via ChromaDB.
    2. Cross-encoder re-ranking of the candidate set (if ``use_reranker``).
    3. Feedback-bias adjustment derived from stored accept/reject votes.

    The returned DataFrame has both ``similarity`` (bi-encoder cosine, for
    display) and ``rerank_score`` (post-reranker + feedback, used for ranking).
    """
    if not new_exception or not new_exception.strip():
        raise ValueError("new_exception must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    df = load_past_cases(past_cases_path)
    retrieval_chunks = _build_retrieval_chunks(df)
    query_chunks = chunk_text(new_exception, chunk_size=chunk_size, overlap=chunk_overlap)
    if not retrieval_chunks or not query_chunks:
        return df.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            matched_chunk=pd.Series(dtype=str),
            rerank_score=pd.Series(dtype=float),
        )

    collection = _index_retrieval_chunks(retrieval_chunks, persist_directory=vector_store_path)
    model = get_model()
    encoded_query_chunks = model.encode(
        query_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result_count = collection.count()
    if result_count == 0:
        return df.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            matched_chunk=pd.Series(dtype=str),
            rerank_score=pd.Series(dtype=float),
        )

    search_results = collection.query(
        query_embeddings=np.asarray(encoded_query_chunks, dtype=float).tolist(),
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    best_chunk_for_case: dict[str, tuple[float, str]] = {}
    for query_index, case_ids in enumerate(search_results["ids"]):
        for chunk_id, document, metadata, distance in zip(
            case_ids,
            search_results["documents"][query_index],
            search_results["metadatas"][query_index],
            search_results["distances"][query_index],
        ):
            case_id = str(metadata["case_id"])
            similarity = 1.0 - float(distance)
            current = best_chunk_for_case.get(case_id)
            if current is None or similarity > current[0]:
                best_chunk_for_case[case_id] = (similarity, document)

    df = df.copy()
    df["similarity"] = df["case_id"].astype(str).map(
        {case_id: match[0] for case_id, match in best_chunk_for_case.items()}
    )
    df["matched_chunk"] = df["case_id"].astype(str).map(
        {case_id: match[1] for case_id, match in best_chunk_for_case.items()}
    )
    # Pre-filter to candidates that actually got a bi-encoder score before reranking.
    candidates = df.dropna(subset=["similarity"]).sort_values("similarity", ascending=False)

    if use_reranker and not candidates.empty:
        from src.reranker import rerank  # local import to avoid circular at module level
        feedback_bias = _load_feedback_bias(database_path=database_path)
        ranked = rerank(
            query=new_exception,
            candidates=candidates,
            feedback_bias=feedback_bias,
            top_k=top_k,
        )
    else:
        candidates["rerank_score"] = candidates["similarity"]
        ranked = candidates.head(top_k)

    return ranked.reset_index(drop=True)



def _normalize_action_resolution(resolution: str) -> str:
    text = (resolution or "").lower()
    if any(keyword in text for keyword in ["missing", "document", "signature", "not included", "not provided"]):
        return "request_missing_document"
    if any(keyword in text for keyword in ["correct", "update", "match", "change", "fix", "discrepancy"]):
        return "propose_field_correction"
    if any(keyword in text for keyword in ["specialist", "expert", "review", "escalate", "underwriting"]):
        return "route_to_specialist"
    return "abstain"


from src.guardrails import evaluate_guardrails


def build_resolution_suggestion(
    case_id: str,
    exception_text: str,
    sensitive_case: bool = False,
    top_k: int = 3,
    matches: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build a deterministic resolution suggestion based on retrieved evidence and guardrails.

    Operational logic:
    1. Retrieves similar precedent cases if not supplied.
    2. Runs multi-stage guardrail evaluation (PII, injection, grounding).
    3. If confidence < 0.55 (safety threshold) or injection detected: forces 'abstain'.
    4. If confidence < 0.80 or sensitive_case=True: flags human_review_required=True.
    5. Normalizes approved historical resolution into a standardized operational action.

    Returns:
        Dict containing case_id, exception_type, suggested_action, reason,
        closest_case_id, confidence, human_review_required, and guardrails audit.
    """
    matches = matches if matches is not None else find_similar_cases(exception_text, top_k=top_k)
    guardrail_eval = evaluate_guardrails(
        exception_text=exception_text,
        matches=matches,
        sensitive_case=sensitive_case,
    )

    if matches.empty:
        return {
            "case_id": case_id,
            "exception_type": "unknown",
            "suggested_action": "abstain",
            "reason": "No approved historical evidence was available; human review is required.",
            "closest_case_id": "",
            "confidence": 0.0,
            "human_review_required": True,
            "guardrails": guardrail_eval,
        }
    best_match = matches.iloc[0]
    confidence = float(best_match["similarity"])
    matched_chunk = str(best_match.get("matched_chunk", best_match["exception_text"]))

    if confidence < 0.55:
        suggested_action = "abstain"
        reason = (
            f"No strong match was found above the project safety threshold. "
            f"Closest prior case {best_match['case_id']} had a similarity of {confidence:.4f}, "
            f"which is below the safe routing threshold. Retrieved evidence: {matched_chunk}"
        )
    else:
        suggested_action = _normalize_action_resolution(best_match["approved_resolution"])
        reason = (
            f"Closest prior case {best_match['case_id']} had a similar issue: "
            f"{best_match['exception_text']} Retrieved evidence: {matched_chunk} "
            f"The approved resolution was '{best_match['approved_resolution']}'."
        )

    # Override suggestion if guardrails forced action (e.g. prompt injection)
    if guardrail_eval["force_action"]:
        suggested_action = guardrail_eval["force_action"]
        reason = f"Guardrail trip ({'; '.join(guardrail_eval['reasons'])}). " + reason

    human_review_required = bool(
        confidence < 0.80 or sensitive_case or suggested_action == "abstain" or guardrail_eval["force_human_review"]
    )

    return {
        "case_id": case_id,
        "exception_type": str(best_match["exception_type"]),
        "suggested_action": suggested_action,
        "reason": reason,
        "closest_case_id": str(best_match["case_id"]),
        "confidence": round(confidence, 4),
        "human_review_required": human_review_required,
        "guardrails": guardrail_eval,
    }


def suggest_resolution_for_case(
    case_row: Dict[str, Any],
    top_k: int = 3,
    matches: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    return build_resolution_suggestion(
        case_id=str(case_row.get("case_id", "CASE-NEW")),
        exception_text=str(case_row.get("exception_text", "")),
        sensitive_case=bool(case_row.get("sensitive_case", False)),
        top_k=top_k,
        matches=matches,
    )


def load_incoming_cases(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _as_sqlite_bool(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def _now_utc() -> str:
    """ISO-8601 UTC timestamp suitable for storing in SQLite TEXT columns."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _append_event(
    connection: sqlite3.Connection,
    case_id: str,
    event_type: str,
    actor: str,
    reason: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert one row into case_event_log inside an already-open transaction.

    Must be called within a ``with connection:`` block so the log write and the
    queue change are committed or rolled back together.
    """
    connection.execute(
        """
        INSERT INTO case_event_log
            (case_id, event_type, actor, reason, payload, occurred_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(case_id),
            str(event_type),
            str(actor),
            str(reason),
            json.dumps(payload) if payload else None,
            _now_utc(),
        ),
    )


# ── Schema migrations ─────────────────────────────────────────────────────────
#
# Each function receives an open sqlite3.Connection and is responsible for
# exactly one schema version step.  They are called inside a transaction.
# Add a new _migrate_vN function and bump _SCHEMA_VERSION to evolve the schema
# without dropping existing workbench.db files.


def _migrate_v1(connection: sqlite3.Connection) -> None:
    """Create exception_queue with data-integrity CHECK constraints.

    expected_type and expected_action are reviewer-assigned labels, not
    model outputs.  The CHECK values mirror EXCEPTION_TYPES / EXCEPTION_ACTIONS
    so the database itself rejects invalid data independently of Python.
    """
    valid_types = ", ".join(f"'{t}'" for t in EXCEPTION_TYPES)
    valid_actions = ", ".join(f"'{a}'" for a in EXCEPTION_ACTIONS)
    valid_statuses = ", ".join(f"'{s}'" for s in QUEUE_STATUSES)

    connection.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS exception_queue (
            case_id        TEXT    NOT NULL PRIMARY KEY
                               CHECK(length(trim(case_id)) > 0),
            exception_text TEXT    NOT NULL
                               CHECK(length(trim(exception_text)) > 0),
            -- Reviewer label: one of the controlled exception categories.
            expected_type  TEXT    NOT NULL DEFAULT 'other'
                               CHECK(expected_type IN ({valid_types})),
            -- Reviewer label: next action recommended for this case.
            expected_action TEXT   NOT NULL DEFAULT 'abstain'
                               CHECK(expected_action IN ({valid_actions})),
            sensitive_case INTEGER NOT NULL DEFAULT 0
                               CHECK(sensitive_case IN (0, 1)),
            status         TEXT    NOT NULL DEFAULT 'New'
                               CHECK(status IN ({valid_statuses})),
            created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )


def _migrate_v2(connection: sqlite3.Connection) -> None:
    """Create the append-only case_event_log and the composite queue index.

    The event log records every create / edit / status-change / approve / delete
    action with actor, ISO-8601 UTC timestamp, human-readable reason, and an
    optional JSON payload of changed fields.  It is written in the same
    transaction as the matching queue change so the two are always consistent.

    The composite index on (status, created_at) is created unconditionally here;
    for a v1 schema it is cheap because the table is small.  At higher volume
    it pays off immediately for the filtered queue queries added to
    load_exception_queue.
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS case_event_log (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id     TEXT    NOT NULL,
            event_type  TEXT    NOT NULL
                            CHECK(event_type IN
                                  ('created', 'edited', 'status_changed',
                                   'approved', 'deleted')),
            actor       TEXT    NOT NULL,
            reason      TEXT    NOT NULL DEFAULT '',
            payload     TEXT,           -- JSON blob of changed fields, nullable
            occurred_at TEXT    NOT NULL -- ISO-8601 UTC
        );

        CREATE INDEX IF NOT EXISTS idx_event_log_case
            ON case_event_log (case_id, occurred_at);

        -- DD4: composite index so filter/order queries avoid a full table scan.
        CREATE INDEX IF NOT EXISTS idx_queue_status_created
            ON exception_queue (status, created_at);
        """
    )



def _migrate_v3(connection: sqlite3.Connection) -> None:
    """Create suggestion_feedback table for the accept/reject feedback loop.

    Each row records a reviewer's verdict on a RAG suggestion for a given
    case.  The feedback loop uses these votes to compute a per-case bias delta
    that is applied on top of the cross-encoder re-rank score at query time —
    no GPU training loop required.
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS suggestion_feedback (
            feedback_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id      TEXT    NOT NULL,
            -- The past case whose suggestion the reviewer voted on.
            matched_case_id TEXT NOT NULL,
            verdict      TEXT    NOT NULL
                             CHECK(verdict IN ('accept', 'reject')),
            actor        TEXT    NOT NULL,
            occurred_at  TEXT    NOT NULL  -- ISO-8601 UTC
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_case
            ON suggestion_feedback (matched_case_id);
        """
    )


_MIGRATIONS: List[Any] = [_migrate_v1, _migrate_v2, _migrate_v3]


def _run_migrations(connection: sqlite3.Connection) -> None:
    """Apply any pending schema migrations using PRAGMA user_version.

    Reads the stored user_version, runs each missing migration in order inside
    its own transaction, then writes the new user_version.  This means:
    - Fresh databases get the full schema in one shot.
    - Existing workbench.db files are evolved without a DROP/re-seed.
    """
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    for version_index in range(current_version, _SCHEMA_VERSION):
        migration_fn = _MIGRATIONS[version_index]
        with connection:
            migration_fn(connection)
            # user_version is not a bind-parameter target; use f-string (safe: integer only)
            connection.execute(f"PRAGMA user_version = {version_index + 1}")


# ── Seed ─────────────────────────────────────────────────────────────────────

def _seed_queue(connection: sqlite3.Connection, seed_path: Path) -> None:
    """Populate exception_queue from the CSV seed file if the table is empty."""
    row_count = connection.execute("SELECT COUNT(*) FROM exception_queue").fetchone()[0]
    if row_count:
        return

    if not seed_path.is_file():
        raise FileNotFoundError(f"Queue seed CSV not found: {seed_path}")
    seed = pd.read_csv(seed_path)
    missing_columns = set(QUEUE_COLUMNS[:-1]).difference(seed.columns)
    if missing_columns:
        raise ValueError(f"Queue seed CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if seed["case_id"].duplicated().any():
        raise ValueError("Queue seed CSV contains duplicate case IDs.")
    if "status" not in seed.columns:
        seed["status"] = "New"
    invalid_statuses = set(seed["status"].dropna().astype(str)).difference(QUEUE_STATUSES)
    if invalid_statuses:
        raise ValueError(f"Queue seed CSV contains unsupported statuses: {', '.join(sorted(invalid_statuses))}")
    invalid_types = set(seed["expected_type"].dropna().astype(str)).difference(EXCEPTION_TYPES)
    if invalid_types:
        raise ValueError(f"Queue seed CSV contains invalid exception types: {', '.join(sorted(invalid_types))}")
    invalid_actions = set(seed["expected_action"].dropna().astype(str)).difference(EXCEPTION_ACTIONS)
    if invalid_actions:
        raise ValueError(f"Queue seed CSV contains invalid exception actions: {', '.join(sorted(invalid_actions))}")

    with connection:
        for _, row in seed.iterrows():
            connection.execute(
                """
                INSERT INTO exception_queue
                    (case_id, exception_text, expected_type, expected_action,
                     sensitive_case, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["case_id"]),
                    str(row["exception_text"]),
                    str(row["expected_type"]),
                    str(row["expected_action"]),
                    _as_sqlite_bool(row["sensitive_case"]),
                    str(row["status"]),
                ),
            )
            _append_event(
                connection,
                case_id=str(row["case_id"]),
                event_type="created",
                actor="seed",
                reason="Initial seed load",
            )


# ── Public queue API ──────────────────────────────────────────────────────────

def load_exception_queue(
    database_path: Optional[str | Path] = None,
    seed_path: Optional[str | Path] = None,
    *,
    status_filter: Optional[str] = None,
    order_by: str = "created_at",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load the exception queue, optionally filtered and ordered in SQL.

    DD4: filter/order/limit are pushed into the SQL query so only the needed
    rows are transferred from SQLite to Python, even when the queue is large.

    Args:
        status_filter: One of QUEUE_STATUSES or None (returns all statuses).
        order_by: Column name to sort by; defaults to ``created_at``.
        limit: Maximum number of rows to return; None means no cap.
    """
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    csv_path = Path(seed_path) if seed_path is not None else _DEFAULT_QUEUE_CSV
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        _run_migrations(connection)
        _seed_queue(connection, csv_path)

        # DD4: build filter SQL — never interpolate user strings, use bind params.
        allowed_order_columns = {
            "case_id", "created_at", "status", "expected_type", "sensitive_case"
        }
        if order_by not in allowed_order_columns:
            order_by = "created_at"

        conditions: list[str] = []
        params: list[Any] = []
        if status_filter and status_filter in QUEUE_STATUSES:
            conditions.append("status = ?")
            params.append(status_filter)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        queue = pd.read_sql(
            f"""
            SELECT case_id, exception_text, expected_type, expected_action,
                   sensitive_case, status, created_at
            FROM exception_queue
            {where_clause}
            ORDER BY {order_by}
            {limit_clause}
            """,
            connection,
            params=params,
        )
    finally:
        connection.close()

    missing_columns = set(QUEUE_COLUMNS).difference(queue.columns)
    if missing_columns:
        raise ValueError(f"SQLite exception queue is missing columns: {', '.join(sorted(missing_columns))}")
    invalid_statuses = set(queue["status"].dropna().astype(str)).difference(QUEUE_STATUSES)
    if invalid_statuses:
        raise ValueError(f"SQLite exception queue contains unsupported statuses: {', '.join(sorted(invalid_statuses))}")
    queue["sensitive_case"] = queue["sensitive_case"].astype(bool)
    return queue


def update_exception_status(
    case_id: str,
    status: str,
    database_path: Optional[str | Path] = None,
    role: str = ROLE_REVIEWER,
    actor: str = "reviewer",
    reason: str = "",
) -> None:
    check_case_mutation_permission(role, action="update status for")
    if status not in QUEUE_STATUSES:
        raise ValueError(f"Unsupported queue status: {status}")

    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                "UPDATE exception_queue SET status = ? WHERE case_id = ?",
                (status, str(case_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Case ID not found in exception queue: {case_id}")
            _append_event(
                connection,
                case_id=case_id,
                event_type="status_changed",
                actor=actor,
                reason=reason or f"Status changed to {status}",
                payload={"status": status},
            )
    finally:
        connection.close()


def create_exception_case(
    case: Dict[str, Any],
    database_path: Optional[str | Path] = None,
    role: str = ROLE_REVIEWER,
    actor: str = "reviewer",
    reason: str = "",
) -> None:
    check_case_mutation_permission(role, action="create")
    case_id = str(case.get("case_id", "")).strip()
    exception_text = str(case.get("exception_text", "")).strip()
    # expected_type and expected_action are reviewer labels, not model outputs.
    expected_type = str(case.get("expected_type", "other")).strip()
    expected_action = str(case.get("expected_action", "abstain")).strip()
    status = str(case.get("status", "New"))

    if not case_id or not exception_text:
        raise ValueError("A case ID and exception text are required.")
    if status not in QUEUE_STATUSES:
        raise ValueError(f"Unsupported queue status: {status}")
    if expected_type not in EXCEPTION_TYPES:
        raise ValueError(f"Invalid reviewer label for expected_type: '{expected_type}'")
    if expected_action not in EXCEPTION_ACTIONS:
        raise ValueError(f"Invalid reviewer label for expected_action: '{expected_action}'")

    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO exception_queue
                    (case_id, exception_text, expected_type, expected_action,
                     sensitive_case, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    exception_text,
                    expected_type,
                    expected_action,
                    _as_sqlite_bool(case.get("sensitive_case", False)),
                    status,
                ),
            )
            _append_event(
                connection,
                case_id=case_id,
                event_type="created",
                actor=actor,
                reason=reason or "Case created",
                payload={
                    "expected_type": expected_type,
                    "expected_action": expected_action,
                    "status": status,
                    "sensitive_case": bool(case.get("sensitive_case", False)),
                },
            )
    finally:
        connection.close()


def update_exception_case(
    case_id: str,
    updates: Dict[str, Any],
    database_path: Optional[str | Path] = None,
    role: str = ROLE_REVIEWER,
    actor: str = "reviewer",
    reason: str = "",
) -> None:
    check_case_mutation_permission(role, action="update")
    allowed_columns = {"exception_text", "expected_type", "expected_action", "sensitive_case", "status"}
    if not updates:
        return
    invalid_columns = set(updates).difference(allowed_columns)
    if invalid_columns:
        raise ValueError(f"Unsupported exception fields: {', '.join(sorted(invalid_columns))}")
    if "status" in updates and updates["status"] not in QUEUE_STATUSES:
        raise ValueError(f"Unsupported queue status: {updates['status']}")
    # Validate reviewer labels before hitting the DB constraint.
    if "expected_type" in updates and updates["expected_type"] not in EXCEPTION_TYPES:
        raise ValueError(f"Invalid reviewer label for expected_type: '{updates['expected_type']}'")
    if "expected_action" in updates and updates["expected_action"] not in EXCEPTION_ACTIONS:
        raise ValueError(f"Invalid reviewer label for expected_action: '{updates['expected_action']}'")

    values = dict(updates)
    if "sensitive_case" in values:
        values["sensitive_case"] = _as_sqlite_bool(values["sensitive_case"])
    set_clause = ", ".join(f"{column} = ?" for column in values)

    # Determine event_type: status-only change vs general edit vs approval.
    if set(updates.keys()) == {"status"} and updates.get("status") == "Resolved":
        event_type = "approved"
    elif "status" in updates and set(updates.keys()) == {"status"}:
        event_type = "status_changed"
    else:
        event_type = "edited"

    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                f"UPDATE exception_queue SET {set_clause} WHERE case_id = ?",
                (*values.values(), str(case_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Case ID not found in exception queue: {case_id}")
            _append_event(
                connection,
                case_id=case_id,
                event_type=event_type,
                actor=actor,
                reason=reason or f"Case {event_type}",
                payload={k: (bool(v) if k == "sensitive_case" else v) for k, v in updates.items()},
            )
    finally:
        connection.close()


def delete_exception_case(
    case_id: str,
    database_path: Optional[str | Path] = None,
    role: str = ROLE_REVIEWER,
    actor: str = "reviewer",
    reason: str = "",
) -> None:
    check_case_mutation_permission(role, action="delete")
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            # Log before delete so the case_id FK reference is still valid.
            _append_event(
                connection,
                case_id=case_id,
                event_type="deleted",
                actor=actor,
                reason=reason or "Case deleted",
            )
            cursor = connection.execute(
                "DELETE FROM exception_queue WHERE case_id = ?",
                (str(case_id),),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Case ID not found in exception queue: {case_id}")
    finally:
        connection.close()


def record_suggestion_feedback(
    case_id: str,
    matched_case_id: str,
    verdict: str,
    actor: str = "reviewer",
    database_path: Optional[str | Path] = None,
) -> None:
    """Record a reviewer's accept or reject verdict on a retrieval suggestion.

    Args:
        case_id: The active queue case the reviewer was looking at.
        matched_case_id: The past case ID whose suggestion was rated.
        verdict: ``'accept'`` or ``'reject'``.
        actor: Reviewer identifier (email when OAuth is active).
    """
    if verdict not in ("accept", "reject"):
        raise ValueError(f"verdict must be 'accept' or 'reject', got: {verdict!r}")
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO suggestion_feedback
                    (case_id, matched_case_id, verdict, actor, occurred_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(case_id), str(matched_case_id), verdict, str(actor), _now_utc()),
            )
    finally:
        connection.close()


def _load_feedback_bias(
    database_path: Optional[str | Path] = None,
    accept_weight: float = 0.05,
    reject_weight: float = 0.08,
    cap: float = 0.25,
) -> dict[str, float]:
    """Compute a per-case score delta from stored accept/reject votes.

    Algorithm:
        bias = min(accepts * accept_weight - rejects * reject_weight, cap)
              clipped to [-cap, +cap]

    This is applied additively to the cross-encoder sigmoid score before the
    final ranking, so consistently accepted matches rise and consistently
    rejected ones fall.
    """
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    if not db_path.exists():
        return {}
    try:
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                """
                SELECT matched_case_id,
                       SUM(CASE WHEN verdict = 'accept' THEN 1 ELSE 0 END) AS accepts,
                       SUM(CASE WHEN verdict = 'reject' THEN 1 ELSE 0 END) AS rejects
                FROM suggestion_feedback
                GROUP BY matched_case_id
                """
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.OperationalError:
        # Table may not exist yet on older databases still being migrated.
        return {}

    bias: dict[str, float] = {}
    for matched_case_id, accepts, rejects in rows:
        delta = float(accepts) * accept_weight - float(rejects) * reject_weight
        bias[str(matched_case_id)] = max(-cap, min(cap, delta))
    return bias



def load_case_event_log(
    case_id: Optional[str] = None,
    database_path: Optional[str | Path] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Return the event log for one case or the whole workbench (newest first).

    Args:
        case_id: Filter to a single case; None returns all cases.
        limit: Maximum rows to return.
    """
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        params: list[Any] = []
        where = ""
        if case_id is not None:
            where = "WHERE case_id = ?"
            params.append(str(case_id))
        log = pd.read_sql(
            f"""
            SELECT event_id, case_id, event_type, actor, reason, payload, occurred_at
            FROM case_event_log
            {where}
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT {int(limit)}
            """,
            connection,
            params=params,
        )
    finally:
        connection.close()
    return log


def run_case_batch(path: str | Path, top_k: int = 3) -> list[dict]:
    df = load_incoming_cases(path)
    return [suggest_resolution_for_case(row.to_dict(), top_k=top_k) for _, row in df.iterrows()]


def main() -> None:
    demo_cases = run_case_batch(Path(__file__).resolve().parents[1] / "data" / "incoming_exceptions.csv")
    print(json.dumps(demo_cases, indent=2))


if __name__ == "__main__":
    main()
