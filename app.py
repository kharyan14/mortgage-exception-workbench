from __future__ import annotations

import hashlib
import sqlite3
from uuid import uuid4
from typing import Tuple

import streamlit as st

from src.document_ocr import describe_exception, extract_document_fields, extract_uploaded_document
from src.exception_workbench import (
    QUEUE_STATUSES,
    create_exception_case,
    delete_exception_case,
    find_similar_cases,
    load_exception_queue,
    suggest_resolution_for_case,
    update_exception_case,
)
from src.rag_generation import generate_grounded_response

EXCEPTION_TYPES = (
    "signature",
    "income",
    "address",
    "missing_document",
    "conflicting_values",
    "low_confidence_extraction",
    "other",
)
SUGGESTED_ACTIONS = (
    "request_missing_document",
    "propose_field_correction",
    "route_to_specialist",
    "abstain",
)
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
st.title("Mortgage Exception Workbench")
st.caption("Synthetic mortgage-document cases · Reviewer remains in control")

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
            create_submitted = st.form_submit_button("Add case to queue")
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
                        }
                    )
                except (OSError, sqlite3.Error, ValueError) as error:
                    st.error(f"Could not add the extracted case: {error}")
                else:
                    st.session_state.pop("upload_draft", None)
                    st.session_state.pop("upload_token", None)
                    st.success(f"Added {new_case_id} to the exception queue.")
                    st.rerun()

try:
    queue = load_exception_queue()
except (OSError, ValueError) as error:
    st.error(f"Could not load the exception queue: {error}")
    st.stop()

st.subheader("Exception queue")
status_filter = st.selectbox("Filter queue", ("All", "New", "In Review", "Resolved"))
visible_queue = queue if status_filter == "All" else queue[queue["status"] == status_filter]

display_columns = [
    column
    for column in ("case_id", "expected_type", "exception_text", "expected_action", "status", "sensitive_case")
    if column in visible_queue.columns
]
edited_visible_queue = st.data_editor(
    visible_queue[display_columns],
    key=f"exception_queue_editor_{status_filter}",
    column_config={
        "status": st.column_config.SelectboxColumn(
            "Status",
            options=list(QUEUE_STATUSES),
            required=True,
        ),
        "expected_type": st.column_config.TextColumn("Case type"),
        "exception_text": st.column_config.TextColumn("Exception"),
        "expected_action": st.column_config.SelectboxColumn(
            "Expected action",
            options=list(SUGGESTED_ACTIONS),
            required=True,
        ),
        "sensitive_case": st.column_config.CheckboxColumn("Sensitive"),
    },
    disabled=["case_id"],
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
            update_exception_case(changed_case_id, changes)
    except (KeyError, OSError, sqlite3.Error, ValueError) as error:
        st.error(f"Could not save case changes: {error}")
        st.stop()
    st.rerun()

total_cases = len(visible_queue)
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
st.warning("Deleting removes this case from the workbench database.")
confirm_delete = st.checkbox(f"Confirm deletion of {selected_case_id}")
if st.button("Delete selected case", disabled=not confirm_delete):
    try:
        delete_exception_case(selected_case_id)
    except (KeyError, OSError, sqlite3.Error) as error:
        st.error(f"Could not delete case: {error}")
    else:
        st.success(f"Deleted {selected_case_id}.")
        st.rerun()

st.subheader("Similar approved cases")
try:
    matches = find_similar_cases(case["exception_text"], top_k=3)
    suggestion = suggest_resolution_for_case(case, top_k=3, matches=matches)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"Could not generate matching results: {error}")
    st.stop()

st.dataframe(
    matches[
        ["case_id", "exception_type", "exception_text", "approved_resolution", "matched_chunk", "similarity"]
    ].rename(
        columns={
            "case_id": "Past case",
            "exception_type": "Type",
            "exception_text": "Past exception",
            "approved_resolution": "Approved resolution",
            "matched_chunk": "Retrieved evidence chunk",
            "similarity": "Similarity",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

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

    @st.cache_data(show_spinner=False)
    def cached_grounded_response(query: str, retrieved_evidence: Tuple[Tuple[str, str, str], ...]) -> str:
        return generate_grounded_response(query, retrieved_evidence)

    try:
        with st.spinner("Generating an evidence-grounded note with the local Ollama model..."):
            grounded_response = cached_grounded_response(case["exception_text"], evidence)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"RAG generation failed: {error}")
    else:
        st.write(grounded_response)
        st.caption("Generated locally with Ollama from the retrieved chunks shown above; verify evidence before acting.")

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

st.caption(
    "Changing a case to Resolved updates its queue status only; reviewer approval and an audit record are not "
    "captured yet. Add those controls before treating a resolved status as a signed-off decision."
)
