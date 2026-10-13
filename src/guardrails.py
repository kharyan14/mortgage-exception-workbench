from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Regular expression patterns for PII detection
SSN_PATTERN = re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
DOB_PATTERN = re.compile(r"\b(?:DOB|Date of Birth|Birth Date)[:\s]+(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", re.IGNORECASE)

# Prompt injection patterns commonly attempted in OCR or user text inputs
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"approve\s+(this\s+)?(loan|application|credit)\s+immediately", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(security|review|underwriting|checks)", re.IGNORECASE),
    re.compile(r"forget\s+(your\s+)?(rules|role|guidelines)", re.IGNORECASE),
    re.compile(r"do\s+not\s+require\s+human\s+review", re.IGNORECASE),
    re.compile(r"(reveal|print|show|repeat)\s+(the\s+)?(system\s+prompt|initial\s+prompt|instructions)", re.IGNORECASE),
    re.compile(r"\b(act\s+as\s+(an\s+)?unrestricted|dan\s+mode|developer\s+mode)\b", re.IGNORECASE),
    re.compile(r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<system>|SYSTEM:)", re.IGNORECASE),
    re.compile(r"(disregard|drop)\s+(all\s+)?guidelines", re.IGNORECASE),
]

# Policy rules for LLM generation (forbidden credit / authority claims)
FORBIDDEN_LLM_CLAIMS = [
    re.compile(r"\b(i\s+have\s+approved|loan\s+is\s+approved|credit\s+approved)\b", re.IGNORECASE),
    re.compile(r"\b(underwriting\s+complete|final\s+decision\s+made|auto-approved)\b", re.IGNORECASE),
    re.compile(r"\b(override\s+policy|bypassing\s+review)\b", re.IGNORECASE),
    re.compile(r"\b(guarantee\s+(approval|closing|rate)|rate\s+is\s+locked\s+guaranteed)\b", re.IGNORECASE),
    re.compile(r"\b(no\s+human\s+review\s+needed|skip\s+human\s+review)\b", re.IGNORECASE),
]

# Pattern to extract case citations (e.g., CASE-001, CASE-102)
CASE_CITATION_PATTERN = re.compile(r"\bCASE-[A-Za-z0-9_-]+\b")


def detect_pii(text: str) -> Dict[str, Any]:
    """Detect presence of sensitive PII (SSN, Email, Phone, Credit Card, DOB) in text."""
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
    if CREDIT_CARD_PATTERN.search(text):
        detected_types.append("CREDIT_CARD")
        matches_count += len(CREDIT_CARD_PATTERN.findall(text))
    if DOB_PATTERN.search(text):
        detected_types.append("DOB")
        matches_count += len(DOB_PATTERN.findall(text))

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
    sanitized = CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", sanitized)
    sanitized = DOB_PATTERN.sub("[REDACTED_DOB]", sanitized)
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


def validate_citations(
    generated_text: str,
    evidence_cases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate that any case ID cited in the generated text actually exists in the retrieved evidence.
    Detects citation hallucinations (e.g. LLM citing non-existent CASE-999).
    """
    if not generated_text or evidence_cases is None:
        return {"citations_valid": True, "cited_cases": [], "hallucinated_cases": []}

    cited_cases = list(set(CASE_CITATION_PATTERN.findall(generated_text)))
    valid_evidence_set = set(str(c).strip() for c in evidence_cases)

    hallucinated_cases = [c for c in cited_cases if c not in valid_evidence_set]

    return {
        "citations_valid": len(hallucinated_cases) == 0,
        "cited_cases": cited_cases,
        "hallucinated_cases": hallucinated_cases,
    }


def compute_groundedness_score(
    generated_text: str,
    evidence_chunks: List[str],
) -> Dict[str, Any]:
    """
    Evaluate the groundedness of a generated reviewer note against retrieved evidence chunks.
    Calculates sentence-level grounding based on keyword and semantic entity presence.
    """
    if not generated_text or not evidence_chunks:
        return {
            "groundedness_score": 0.0,
            "grounded_sentences_count": 0,
            "total_sentences_count": 0,
            "is_grounded": False,
        }

    combined_evidence = " ".join(evidence_chunks).lower()
    combined_evidence_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", combined_evidence))
    # Basic stopwords to filter out common noise
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
        "on", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "from", "up",
        "down", "in", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "don", "should", "now", "it", "this", "that", "these",
        "those", "i", "you", "he", "she", "we", "they", "reviewer", "note"
    }

    sentences = [s.strip() for s in re.split(r"[.!?]+", generated_text) if len(s.strip()) > 10]
    if not sentences:
        return {
            "groundedness_score": 1.0,
            "grounded_sentences_count": 0,
            "total_sentences_count": 0,
            "is_grounded": True,
        }

    grounded_count = 0
    for sentence in sentences:
        words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", sentence)]
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        if not content_words:
            grounded_count += 1
            continue
        # Check fraction of content words supported in evidence (exact or stem prefix)
        supported_words = sum(
            1 for w in content_words
            if w in combined_evidence_words
            or any(
                (w.startswith(ew) or ew.startswith(w)) and len(ew) >= 4 and len(w) >= 4
                for ew in combined_evidence_words
            )
        )
        support_ratio = supported_words / len(content_words)
        if support_ratio >= 0.35:
            grounded_count += 1

    score = round(grounded_count / len(sentences), 4)
    return {
        "groundedness_score": score,
        "grounded_sentences_count": grounded_count,
        "total_sentences_count": len(sentences),
        "is_grounded": score >= 0.70,
    }


def validate_llm_output(
    generated_text: str,
    evidence_cases: Optional[List[str]] = None,
    evidence_chunks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate LLM generated response against output policy guardrails and citation fidelity."""
    if not generated_text:
        return {
            "is_safe": False,
            "violations": ["Empty LLM output"],
            "warnings": [],
            "sanitized_text": "",
            "citation_info": {"citations_valid": True, "cited_cases": [], "hallucinated_cases": []},
            "groundedness": {"groundedness_score": 0.0, "is_grounded": False},
        }

    violations: List[str] = []
    warnings: List[str] = []

    # Check for forbidden claims (LLM attempting to auto-approve or claim credit authority)
    for pattern in FORBIDDEN_LLM_CLAIMS:
        match = pattern.search(generated_text)
        if match:
            violations.append(f"LLM attempted unauthorized authority claim: '{match.group(0)}'")

    # Check for PII leakage in generated text
    pii_check = detect_pii(generated_text)
    if pii_check["pii_detected"]:
        violations.append(f"LLM output contains unredacted PII: {', '.join(pii_check['pii_types'])}")

    # Check for hallucinated case citations
    citation_info = validate_citations(generated_text, evidence_cases)
    if not citation_info["citations_valid"]:
        hallucinations = ", ".join(citation_info["hallucinated_cases"])
        violations.append(f"LLM hallucinated non-retrieved case citations: {hallucinations}")

    # Check groundedness
    groundedness = compute_groundedness_score(generated_text, evidence_chunks or [])
    if evidence_chunks and not groundedness["is_grounded"]:
        warnings.append(f"Low groundedness score ({groundedness['groundedness_score']:.2f} < 0.70); output may contain unsupported claims")

    sanitized_text = sanitize_pii(generated_text)

    return {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "sanitized_text": sanitized_text,
        "citation_info": citation_info,
        "groundedness": groundedness,
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
    evidence_case_ids: List[str] = []
    evidence_chunks: List[str] = []

    if matches is not None and not matches.empty:
        evidence_case_ids = [str(cid) for cid in matches["case_id"].tolist()]
        evidence_chunks = [str(c) for c in matches.get("matched_chunk", matches["exception_text"]).tolist()]
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
        output_info = validate_llm_output(
            generated_response,
            evidence_cases=evidence_case_ids if evidence_case_ids else None,
            evidence_chunks=evidence_chunks if evidence_chunks else None,
        )

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
