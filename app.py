from __future__ import annotations

import hashlib
import sqlite3
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.document_ocr import describe_exception, extract_document_fields, extract_uploaded_document
from src.exception_workbench import (
    EXCEPTION_ACTIONS,
    EXCEPTION_TYPES,
    QUEUE_STATUSES,
    ROLE_REVIEWER,
    ROLE_VIEWER,
    create_exception_case,
    delete_exception_case,
    find_similar_cases,
    load_case_event_log,
    load_exception_queue,
    record_suggestion_feedback,
    suggest_resolution_for_case,
    update_exception_case,
)
from src.rag_generation import generate_grounded_response_stream
from src.auth import require_auth, logout
from src.evals import run_evaluation_suite

# EXCEPTION_TYPES and SUGGESTED_ACTIONS are imported from exception_workbench
# so the UI vocabulary stays in sync with the DB CHECK constraints.
SUGGESTED_ACTIONS = EXCEPTION_ACTIONS
DEFAULT_ACTION_BY_TYPE = {
    "signature": "request_missing_document",
    "income": "request_missing_document",
    "address": "propose_field_correction",
    "missing_document": "request_missing_document",
    "conflicting_values": "propose_field_correction",
    "low_confidence_extraction": "route_to_specialist",
    "other": "abstain",
}


st.set_page_config(page_title="Mortgage Exception Workbench", page_icon="📄", layout="wide")

# ── Authentication ────────────────────────────────────────────────────────────
# require_auth() returns the Identity dict and shows the OAuth login button
# (or the legacy role selector in dev mode).  Everything below this line is
# only reached once the user is authenticated.
identity = require_auth()
current_role = identity["role"]
actor_id = identity["email"]   # real email when OAuth is active; role string in dev mode

# Sidebar: identity card + logout
with st.sidebar:
    st.header("Access & Permissions")
    if identity["authenticated"]:
        st.success(f"✅ Signed in as **{identity['name']}**  \n`{actor_id}`")
        if st.button("Sign out", key="logout_btn"):
            logout()
            st.rerun()
    if current_role == ROLE_VIEWER:
        st.info("👁️ **Viewer Mode**: Look only. Deletions, edits, and creation are refused.")
    else:
        st.success("✏️ **Reviewer Mode**: Authorized to change, update, and delete cases.")

st.title("Mortgage Exception Workbench")
st.caption(
    f"Synthetic mortgage-document cases · Signed in as **{identity['name']}** "
    f"· Role: **{current_role.capitalize()}** "
    f"({'can change cases' if current_role == ROLE_REVIEWER else 'look only'})"
)

st.subheader("Add a document")
st.warning("Use synthetic documents only. Do not upload real borrower or lender documents to this prototype.")
uploaded_file = st.file_uploader(
    "Upload a PDF or scanned image",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
)
upload_draft = None
if uploaded_file is not None:
    uploaded_content = uploaded_file.getvalue()
    upload_token = hashlib.sha256(uploaded_content).hexdigest()
    if st.session_state.get("upload_token") != upload_token:
        st.session_state["upload_token"] = upload_token
        st.session_state.pop("upload_draft", None)

    if st.button("Extract document text", type="primary"):
        try:
            extraction = extract_uploaded_document(uploaded_file.name, uploaded_content)
            fields = extract_document_fields(extraction["text"])
            exception_text = describe_exception(fields, extraction["text"])
            suggested_type = "other"
            if fields.get("signature_status", "").lower() in {"missing", "unsigned", "not provided"}:
                suggested_type = "signature"
            elif fields.get("required_document_status", "").lower() in {"missing", "not provided", "not received"}:
                suggested_type = "income"
            elif fields.get("appraisal_property_address") and fields.get("title_record_address"):
                suggested_type = "address"
            elif exception_text != extraction["text"]:
                lowered_exception = exception_text.lower()
                if any(word in lowered_exception for word in ("signature", "signed", "sign")):
                    suggested_type = "signature"
                elif any(word in lowered_exception for word in ("paystub", "income", "employment")):
                    suggested_type = "income"
                elif any(word in lowered_exception for word in ("address", "property", "title")):
                    suggested_type = "address"

            if exception_text == extraction["text"]:
                exception_text = ""
            st.session_state["upload_draft"] = {
                "document_name": extraction["document_name"],
                "extraction_method": extraction["extraction_method"],
                "ocr_confidence": extraction["ocr_confidence"],
                "extracted_text": extraction["text"],
                "exception_text": exception_text,
                "expected_type": suggested_type,
            }
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not extract this document: {error}")

    upload_draft = st.session_state.get("upload_draft")
    if upload_draft is not None:
        st.write(
            f"**Extraction:** {upload_draft['extraction_method']} · "
            f"OCR confidence: {upload_draft['ocr_confidence'] if upload_draft['ocr_confidence'] is not None else 'N/A'}"
        )
        with st.expander("View extracted document text"):
            st.text(upload_draft["extracted_text"])
        with st.form("create_exception_from_upload"):
            exception_text = st.text_area(
                "Exception description",
                value=upload_draft["exception_text"],
                help="Review and correct the extracted exception before adding it to the queue.",
            )
            expected_type = st.selectbox(
                "Exception type",
                EXCEPTION_TYPES,
                index=EXCEPTION_TYPES.index(upload_draft["expected_type"]),
            )
            expected_action = st.selectbox(
                "Expected next action",
                SUGGESTED_ACTIONS,
                index=SUGGESTED_ACTIONS.index(DEFAULT_ACTION_BY_TYPE[upload_draft["expected_type"]]),
            )
            sensitive_case = st.checkbox("Mark this case as sensitive")
            create_submitted = st.form_submit_button(
                "Add case to queue",
                disabled=(current_role == ROLE_VIEWER),
            )
        if current_role == ROLE_VIEWER:
            st.caption("🔒 Look only mode: Viewers cannot add new cases.")
        if create_submitted:
            if not exception_text.strip():
                st.error("Enter or correct the exception description before adding this case.")
            else:
                new_case_id = f"CASE-UPLOAD-{uuid4().hex[:8].upper()}"
                try:
                    create_exception_case(
                        {
                            "case_id": new_case_id,
                            "exception_text": exception_text,
                            "expected_type": expected_type,
                            "expected_action": expected_action,
                            "sensitive_case": sensitive_case,
                            "status": "New",
                        },
                        role=current_role,
                        actor=actor_id,
                        reason="Created from uploaded document",
                    )
                except (OSError, sqlite3.Error, ValueError, PermissionError) as error:
                    st.error(f"Could not add the extracted case: {error}")
                else:
                    st.session_state.pop("upload_draft", None)
                    st.session_state.pop("upload_token", None)
                    st.success(f"Added {new_case_id} to the exception queue.")
                    st.rerun()

st.subheader("Exception queue")
status_filter = st.selectbox("Filter queue", ("All", "New", "In Review", "Resolved"))

# DD4: filtering is pushed into SQL so pandas never loads unwanted rows.
try:
    queue = load_exception_queue(
        status_filter=None if status_filter == "All" else status_filter,
        order_by="created_at",
    )
except (OSError, ValueError) as error:
    st.error(f"Could not load the exception queue: {error}")
    st.stop()

visible_queue = queue  # already filtered by SQL

is_viewer = current_role == ROLE_VIEWER
if is_viewer:
    st.info("👁️ **Look only mode**: Queue editing is disabled for viewers. Unauthorized changes will be refused.")

display_columns = [
    column
    for column in ("case_id", "expected_type", "exception_text", "expected_action", "status", "sensitive_case")
    if column in visible_queue.columns
]
edited_visible_queue = st.data_editor(
    visible_queue[display_columns],
    key=f"exception_queue_editor_{status_filter}_{current_role}",
    column_config={
        "status": st.column_config.SelectboxColumn(
            "Status",
            options=list(QUEUE_STATUSES),
            required=True,
            disabled=is_viewer,
        ),
        "expected_type": st.column_config.SelectboxColumn(
            "Case type",
            options=list(EXCEPTION_TYPES),
            required=True,
            disabled=is_viewer,
        ),
        "exception_text": st.column_config.TextColumn("Exception", disabled=is_viewer),
        "expected_action": st.column_config.SelectboxColumn(
            "Expected action",
            options=list(SUGGESTED_ACTIONS),
            required=True,
            disabled=is_viewer,
        ),
        "sensitive_case": st.column_config.CheckboxColumn("Sensitive", disabled=is_viewer),
    },
    disabled=True if is_viewer else ["case_id"],
    num_rows="fixed",
    hide_index=True,
    use_container_width=True,
)

original_rows = queue.set_index("case_id")
edited_rows = edited_visible_queue.set_index("case_id")
editable_columns = [column for column in display_columns if column != "case_id"]
changed_cases = []
for case_id in edited_rows.index:
    changes = {
        column: edited_rows.at[case_id, column]
        for column in editable_columns
        if edited_rows.at[case_id, column] != original_rows.at[case_id, column]
    }
    if changes:
        changed_cases.append((str(case_id), changes))
if changed_cases:
    try:
        for changed_case_id, changes in changed_cases:
            update_exception_case(
                changed_case_id,
                changes,
                role=current_role,
                actor=actor_id,
                reason="Inline queue edit",
            )
    except (KeyError, OSError, sqlite3.Error, ValueError, PermissionError) as error:
        st.error(f"Could not save case changes: {error}")
        st.stop()
    st.rerun()

total_cases = len(visible_queue)
# Counts always reflect the whole queue, not just the filtered view.
try:
    full_queue_for_counts = load_exception_queue()
    status_counts = full_queue_for_counts["status"].value_counts()
except (OSError, ValueError):
    status_counts = queue["status"].value_counts()

metric_columns = st.columns(4)
metric_columns[0].metric("Cases shown", total_cases)
metric_columns[1].metric("New", int(status_counts.get("New", 0)))
metric_columns[2].metric("In review", int(status_counts.get("In Review", 0)))
metric_columns[3].metric("Resolved", int(status_counts.get("Resolved", 0)))

if visible_queue.empty:
    st.info("No cases match this status filter.")
    st.stop()

case_ids = edited_visible_queue["case_id"].astype(str).tolist()
selected_case_id = st.selectbox("Select a case to inspect", case_ids)
selected_row = queue.loc[queue["case_id"].astype(str) == selected_case_id].iloc[0]
case = selected_row.to_dict()
case["sensitive_case"] = str(case.get("sensitive_case", "")).strip().lower() in {"true", "1", "yes"}

st.divider()
st.subheader(f"Case {selected_case_id}")
case_columns = st.columns(3)
case_columns[0].write(f"**Type:** {case.get('expected_type', 'Unclassified')}")
case_columns[1].write(f"**Status:** {case['status']}")
case_columns[2].write(f"**Sensitive case:** {'Yes' if case['sensitive_case'] else 'No'}")
st.write(f"**Exception:** {case['exception_text']}")

st.subheader("Delete case")
if is_viewer:
    st.info("🔒 Look only mode: Viewers cannot delete cases.")
else:
    st.warning("Deleting removes this case from the workbench database.")
confirm_delete = st.checkbox(
    f"Confirm deletion of {selected_case_id}",
    disabled=is_viewer,
)
delete_reason = st.text_input(
    "Reason for deletion (recorded in audit log)",
    placeholder="e.g. Duplicate entry, entered in error",
    disabled=is_viewer or not confirm_delete,
    key="delete_reason_input",
)
if st.button("Delete selected case", disabled=not confirm_delete or is_viewer):
    try:
        delete_exception_case(
            selected_case_id,
            role=current_role,
            actor=actor_id,
            reason=delete_reason.strip() or "No reason provided",
        )
    except (KeyError, OSError, sqlite3.Error, PermissionError) as error:
        st.error(f"Could not delete case: {error}")
    else:
        st.success(f"Deleted {selected_case_id}.")
        st.rerun()

st.subheader("📋 Audit trail")
try:
    event_log = load_case_event_log(case_id=selected_case_id)
except (OSError, sqlite3.Error) as _log_err:
    event_log = None
with st.expander(f"Event log for {selected_case_id}", expanded=False):
    if event_log is None or event_log.empty:
        st.info("No events recorded yet for this case.")
    else:
        st.dataframe(
            event_log[["occurred_at", "event_type", "actor", "reason", "payload"]].rename(
                columns={
                    "occurred_at": "When (UTC)",
                    "event_type": "Event",
                    "actor": "Actor",
                    "reason": "Reason",
                    "payload": "Changed fields",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.subheader("Similar approved cases")
try:
    matches = find_similar_cases(case["exception_text"], top_k=3)
    suggestion = suggest_resolution_for_case(case, top_k=3, matches=matches)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"Could not generate matching results: {error}")
    st.stop()

# Show bi-encoder similarity AND cross-encoder rerank_score when available.
_display_cols = ["case_id", "exception_type", "exception_text", "approved_resolution", "matched_chunk"]
_rename = {
    "case_id": "Past case",
    "exception_type": "Type",
    "exception_text": "Past exception",
    "approved_resolution": "Approved resolution",
    "matched_chunk": "Retrieved evidence chunk",
    "similarity": "Bi-encoder sim.",
    "rerank_score": "Re-rank score",
}
for _col in ("similarity", "rerank_score"):
    if _col in matches.columns:
        _display_cols.append(_col)

st.dataframe(
    matches[[c for c in _display_cols if c in matches.columns]].rename(columns=_rename),
    hide_index=True,
    use_container_width=True,
)

# ── Feedback thumbs per matched case ─────────────────────────────────────────
if not matches.empty and not is_viewer:
    st.caption("Rate each suggestion to improve future re-ranking:")
    for _i, _mrow in matches.iterrows():
        _mid = str(_mrow["case_id"])
        _fb_cols = st.columns([6, 1, 1])
        _fb_cols[0].write(f"**{_mid}** — {str(_mrow.get('exception_text', ''))[:80]}…")
        if _fb_cols[1].button("👍", key=f"accept_{selected_case_id}_{_mid}_{_i}",
                              help="This suggestion was helpful"):
            try:
                record_suggestion_feedback(
                    case_id=selected_case_id,
                    matched_case_id=_mid,
                    verdict="accept",
                    actor=actor_id,
                )
                st.toast(f"Accepted {_mid} — re-rank will improve over time.")
            except (OSError, sqlite3.Error, ValueError) as _err:
                st.error(f"Could not save feedback: {_err}")
        if _fb_cols[2].button("👎", key=f"reject_{selected_case_id}_{_mid}_{_i}",
                              help="This suggestion was not helpful"):
            try:
                record_suggestion_feedback(
                    case_id=selected_case_id,
                    matched_case_id=_mid,
                    verdict="reject",
                    actor=actor_id,
                )
                st.toast(f"Rejected {_mid} — re-rank will downweight this match.")
            except (OSError, sqlite3.Error, ValueError) as _err:
                st.error(f"Could not save feedback: {_err}")


st.subheader("RAG-generated evidence summary")
if matches.empty:
    st.info("No historical evidence was retrieved, so generation is skipped and the system abstains.")
else:
    evidence = tuple(
        (
            str(row["case_id"]),
            str(row["matched_chunk"]),
            str(row["approved_resolution"]),
        )
        for _, row in matches.iterrows()
    )

    try:
        # DD: stream tokens one-by-one for responsiveness.
        with st.spinner("Connecting to Ollama…"):
            # Materialise generator so we can pass it to write_stream.
            _stream_gen = generate_grounded_response_stream(case["exception_text"], evidence)
        st.write_stream(_stream_gen)
        st.caption(
            "Generated locally with Ollama from the retrieved chunks above; "
            "verify evidence before acting."
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"RAG generation failed: {error}")


st.subheader("Suggested next step")
suggestion_columns = st.columns(3)
suggestion_columns[0].metric("Suggested action", suggestion["suggested_action"].replace("_", " ").title())
suggestion_columns[1].metric("Similarity confidence", f"{suggestion['confidence']:.1%}")
suggestion_columns[2].metric(
    "Human review",
    "Required" if suggestion["human_review_required"] or case["sensitive_case"] else "Recommended",
)
st.write(suggestion["reason"])
if case["sensitive_case"]:
    st.warning("Sensitive cases always require human review.")
elif suggestion["human_review_required"]:
    st.warning("This suggestion is below the 0.80 project threshold and requires human review.")

st.subheader("🛡️ Safety & Policy Guardrails")
guardrails = suggestion.get("guardrails", {})
g_cols = st.columns(4)

if guardrails.get("prompt_injection_detected"):
    g_cols[0].error("🚨 Injection Flagged")
else:
    g_cols[0].success("🛡️ Input Safe")

if guardrails.get("pii_detected"):
    g_cols[1].warning(f"🔒 PII Redacted: {', '.join(guardrails.get('pii_types', []))}")
else:
    g_cols[1].success("🔒 No PII Found")

if guardrails.get("grounding_passed"):
    g_cols[2].success("🎯 Grounding Verified")
else:
    g_cols[2].error("⚠️ Low Grounding")

if guardrails.get("output_safe", True):
    g_cols[3].success("📋 Policy Compliant")
else:
    g_cols[3].error("🚨 Policy Violation")

with st.expander("View Guardrail Audit Logs"):
    st.json(guardrails)

st.divider()
st.subheader("🧪 Model Evaluations & Benchmark Suite")
st.caption("Automated regression testing across retrieval hit rate, deterministic action routing accuracy, and guardrail defenses.")

if st.button("Run Benchmark Evaluation Suite", key="run_evals_btn"):
    with st.spinner("Running 20+ benchmark test cases across retrieval, guardrails, and policy enforcement..."):
        st.session_state["eval_results"] = run_evaluation_suite()

eval_results = st.session_state.get("eval_results")
if eval_results:
    m = eval_results["metrics"]
    e_cols = st.columns(4)
    e_cols[0].metric("Retrieval Recall @ 3", f"{m['retrieval_hit_rate_at_3']:.0%}")
    e_cols[1].metric("Action Accuracy", f"{m['action_routing_accuracy']:.0%}")
    e_cols[2].metric("Injection Defense", f"{m['injection_defense_rate']:.0%}")
    e_cols[3].metric("PII Protection", f"{m['pii_detection_recall']:.0%}")

    pass_status = "✅ All SLA Targets Met" if eval_results["all_slas_met"] else "⚠️ Some Targets Failed"
    st.info(f"**Benchmark Summary:** {eval_results['total_passed']}/{eval_results['total_test_cases']} tests passed ({m['overall_pass_rate']:.1%}) · {pass_status}")

    with st.expander("View Full Benchmark Test Cases & Details", expanded=False):
        eval_df = pd.DataFrame([
            {
                "Test ID": r["id"],
                "Category": r["category"],
                "Query / Input": (r.get("query") or r.get("text") or "")[:60] + "...",
                "Status": "✅ PASS" if r["passed"] else "❌ FAIL",
            }
            for r in eval_results["case_results"]
        ])
        st.dataframe(eval_df, hide_index=True, use_container_width=True)

st.caption(
    "All status changes, edits, and deletions are recorded in the case_event_log table with actor and timestamp. "
    "expected_type and expected_action are reviewer labels — not model outputs."
)
