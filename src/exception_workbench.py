from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

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
QUEUE_COLUMNS = (
    "case_id",
    "exception_text",
    "expected_type",
    "expected_action",
    "sensitive_case",
    "status",
)
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CHUNK_OVERLAP = 20

_model: Optional[SentenceTransformer] = None


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
) -> pd.DataFrame:
    if not new_exception or not new_exception.strip():
        raise ValueError("new_exception must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    df = load_past_cases(past_cases_path)
    retrieval_chunks = _build_retrieval_chunks(df)
    query_chunks = chunk_text(new_exception, chunk_size=chunk_size, overlap=chunk_overlap)
    if not retrieval_chunks or not query_chunks:
        return df.iloc[0:0].assign(similarity=pd.Series(dtype=float), matched_chunk=pd.Series(dtype=str))

    collection = _index_retrieval_chunks(retrieval_chunks, persist_directory=vector_store_path)
    model = get_model()
    encoded_query_chunks = model.encode(
        query_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result_count = collection.count()
    if result_count == 0:
        return df.iloc[0:0].assign(similarity=pd.Series(dtype=float), matched_chunk=pd.Series(dtype=str))

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
    ranked = df.sort_values("similarity", ascending=False).head(top_k).copy()
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


def _as_sqlite_bool(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def _initialize_exception_queue(
    connection: sqlite3.Connection,
    seed_path: Path,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS exception_queue (
            case_id TEXT PRIMARY KEY,
            exception_text TEXT,
            expected_type TEXT,
            expected_action TEXT,
            sensitive_case INTEGER,
            status TEXT
        )
        """
    )
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

    rows = []
    for _, row in seed.iterrows():
        rows.append(
            (
                str(row["case_id"]),
                str(row["exception_text"]),
                str(row["expected_type"]),
                str(row["expected_action"]),
                _as_sqlite_bool(row["sensitive_case"]),
                str(row["status"]),
            )
        )
    connection.executemany(
        """
        INSERT INTO exception_queue
            (case_id, exception_text, expected_type, expected_action, sensitive_case, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def load_exception_queue(
    database_path: Optional[str | Path] = None,
    seed_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    csv_path = Path(seed_path) if seed_path is not None else _DEFAULT_QUEUE_CSV
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            _initialize_exception_queue(connection, csv_path)
        queue = pd.read_sql(
            "SELECT case_id, exception_text, expected_type, expected_action, sensitive_case, status "
            "FROM exception_queue ORDER BY case_id",
            connection,
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
) -> None:
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
    finally:
        connection.close()


def create_exception_case(
    case: Dict[str, Any],
    database_path: Optional[str | Path] = None,
) -> None:
    case_id = str(case.get("case_id", "")).strip()
    exception_text = str(case.get("exception_text", "")).strip()
    expected_type = str(case.get("expected_type", "")).strip()
    expected_action = str(case.get("expected_action", "")).strip()
    status = str(case.get("status", "New"))
    if not case_id or not exception_text:
        raise ValueError("A case ID and exception text are required.")
    if status not in QUEUE_STATUSES:
        raise ValueError(f"Unsupported queue status: {status}")

    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO exception_queue
                    (case_id, exception_text, expected_type, expected_action, sensitive_case, status)
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
    finally:
        connection.close()


def update_exception_case(
    case_id: str,
    updates: Dict[str, Any],
    database_path: Optional[str | Path] = None,
) -> None:
    allowed_columns = {"exception_text", "expected_type", "expected_action", "sensitive_case", "status"}
    if not updates:
        return
    invalid_columns = set(updates).difference(allowed_columns)
    if invalid_columns:
        raise ValueError(f"Unsupported exception fields: {', '.join(sorted(invalid_columns))}")
    if "status" in updates and updates["status"] not in QUEUE_STATUSES:
        raise ValueError(f"Unsupported queue status: {updates['status']}")

    values = dict(updates)
    if "sensitive_case" in values:
        values["sensitive_case"] = _as_sqlite_bool(values["sensitive_case"])
    set_clause = ", ".join(f"{column} = ?" for column in values)
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
    finally:
        connection.close()


def delete_exception_case(
    case_id: str,
    database_path: Optional[str | Path] = None,
) -> None:
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DATABASE
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                "DELETE FROM exception_queue WHERE case_id = ?",
                (str(case_id),),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Case ID not found in exception queue: {case_id}")
    finally:
        connection.close()


def run_case_batch(path: str | Path, top_k: int = 3) -> list[dict]:
    df = load_incoming_cases(path)
    return [suggest_resolution_for_case(row.to_dict(), top_k=top_k) for _, row in df.iterrows()]


def main() -> None:
    demo_cases = run_case_batch(Path(__file__).resolve().parents[1] / "data" / "incoming_exceptions.csv")
    print(json.dumps(demo_cases, indent=2))


if __name__ == "__main__":
    main()
