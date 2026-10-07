from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Regular expression patterns for PII detection
SSN_PATTERN = re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Prompt injection patterns commonly attempted in OCR or user text inputs
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"approve\s+(this\s+)?(loan|application|credit)\s+immediately", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(security|review|underwriting|checks)", re.IGNORECASE),
    re.compile(r"forget\s+(your\s+)?(rules|role|guidelines)", re.IGNORECASE),
    re.compile(r"do\s+not\s+require\s+human\s+review", re.IGNORECASE),
]

# Policy rules for LLM generation (forbidden credit / authority claims)
FORBIDDEN_LLM_CLAIMS = [
    re.compile(r"\b(i\s+have\s+approved|loan\s+is\s+approved|credit\s+approved)\b", re.IGNORECASE),
    re.compile(r"\b(underwriting\s+complete|final\s+decision\s+made|auto-approved)\b", re.IGNORECASE),
    re.compile(r"\b(override\s+policy|bypassing\s+review)\b", re.IGNORECASE),
]


def detect_pii(text: str) -> Dict[str, Any]:
    """Detect presence of sensitive PII (SSN, Email, Phone) in text."""
    if not text:
        return {"pii_detected": False, "pii_types": [], "matches_count": 0}

    detected_types: List[str] = []
    matches_count = 0

    if SSN_PATTERN.search(text):
        detected_types.append("SSN")
        matches_count += len(SSN_PATTERN.findall(text))
    if EMAIL_PATTERN.search(text):
        detected_types.append("EMAIL")
        matches_count += len(EMAIL_PATTERN.findall(text))
    if PHONE_PATTERN.search(text):
        detected_types.append("PHONE")
        matches_count += len(PHONE_PATTERN.findall(text))

    return {
        "pii_detected": len(detected_types) > 0,
        "pii_types": detected_types,
        "matches_count": matches_count,
    }


def sanitize_pii(text: str) -> str:
    """Sanitize and redact sensitive PII patterns from text before passing to LLM."""
    if not text:
        return text
    sanitized = SSN_PATTERN.sub("[REDACTED_SSN]", text)
    sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
    sanitized = PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)
    return sanitized


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """Check text for potential adversarial prompt injections or jailbreaks."""
    if not text:
        return False, None
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Detected prompt injection pattern: '{match.group(0)}'"
    return False, None


def validate_llm_output(
    generated_text: str,
    evidence_cases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate LLM generated response against output policy guardrails."""
    if not generated_text:
        return {
            "is_safe": False,
            "violations": ["Empty LLM output"],
            "sanitized_text": "",
        }

    violations: List[str] = []

    # Check for forbidden claims (LLM attempting to auto-approve or claim credit authority)
    for pattern in FORBIDDEN_LLM_CLAIMS:
        match = pattern.search(generated_text)
        if match:
            violations.append(f"LLM attempted unauthorized authority claim: '{match.group(0)}'")

    # Check for PII leakage in generated text
    pii_check = detect_pii(generated_text)
    if pii_check["pii_detected"]:
        violations.append(f"LLM output contains unredacted PII: {', '.join(pii_check['pii_types'])}")

    sanitized_text = sanitize_pii(generated_text)

    return {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "sanitized_text": sanitized_text,
    }


def evaluate_guardrails(
    exception_text: str,
    matches: Optional[pd.DataFrame] = None,
    generated_response: Optional[str] = None,
    sensitive_case: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates input, retrieval grounding, and output guardrails.
    Returns comprehensive guardrail analysis and enforcement directives.
    """
    # 1. Input Guardrails
    pii_info = detect_pii(exception_text)
    is_injection, injection_reason = detect_prompt_injection(exception_text)
    sanitized_input = sanitize_pii(exception_text)

    input_passed = not is_injection

    # 2. Retrieval Grounding Guardrail
    grounding_passed = True
    grounding_reason = "Retrieval evidence meets safety threshold."
    if matches is not None and not matches.empty:
        top_similarity = float(matches.iloc[0]["similarity"])
        if top_similarity < 0.55:
            grounding_passed = False
            grounding_reason = f"Top similarity ({top_similarity:.4f}) is below safe threshold 0.55."
    elif matches is not None and matches.empty:
        grounding_passed = False
        grounding_reason = "No retrieval evidence matched."

    # 3. Output Policy Guardrail
    output_info: Dict[str, Any] = {"is_safe": True, "violations": []}
    if generated_response:
        output_info = validate_llm_output(generated_response)

    # Combined Guardrail Status
    all_passed = input_passed and grounding_passed and output_info["is_safe"]

    reasons: List[str] = []
    if is_injection:
        reasons.append(injection_reason or "Prompt injection detected")
    if not grounding_passed:
        reasons.append(grounding_reason)
    if not output_info["is_safe"]:
        reasons.extend(output_info["violations"])
    if pii_info["pii_detected"]:
        reasons.append(f"PII detected and redacted: {', '.join(pii_info['pii_types'])}")
    if sensitive_case:
        reasons.append("Case is explicitly marked sensitive.")

    force_action = None
    if is_injection:
        force_action = "abstain"
    elif not grounding_passed:
        force_action = "abstain"

    return {
        "passed": all_passed,
        "pii_detected": pii_info["pii_detected"],
        "pii_types": pii_info["pii_types"],
        "sanitized_input": sanitized_input,
        "prompt_injection_detected": is_injection,
        "injection_reason": injection_reason,
        "grounding_passed": grounding_passed,
        "grounding_reason": grounding_reason,
        "output_safe": output_info["is_safe"],
        "output_violations": output_info["violations"],
        "sanitized_output": output_info.get("sanitized_text", generated_response or ""),
        "reasons": reasons,
        "force_human_review": not all_passed or sensitive_case,
        "force_action": force_action,
    }
