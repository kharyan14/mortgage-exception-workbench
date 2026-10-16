"""RAG generation module — Ollama-backed, streaming-capable.

Public API
----------
generate_grounded_response_stream(query, evidence, ...)
    Generator: yields text tokens one at a time so the caller can stream
    them to the UI with ``st.write_stream``.

generate_grounded_response(query, evidence, ...)
    Blocking convenience wrapper that joins the stream into a single string.
    Retained for callers (evals, batch scripts) that do not need streaming.
"""
from __future__ import annotations

import json
import os
from typing import Generator, Iterable, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_GENERATION_MODEL = "llama3.1:latest"


from src.guardrails import detect_prompt_injection, sanitize_pii, validate_llm_output


def _build_prompt(query: str, evidence: list[Tuple[str, str, str]]) -> str:
    """Assemble the grounded reviewer prompt from PII-sanitised inputs."""
    sanitized_query = sanitize_pii(query)
    evidence_context = "\n\n".join(
        f"[{index}] Prior case {case_id}\n"
        f"Retrieved chunk: {sanitize_pii(chunk)}\n"
        f"Approved prior resolution: {sanitize_pii(approved_resolution)}"
        for index, (case_id, chunk, approved_resolution) in enumerate(evidence, start=1)
    )
    return (
        "You are assisting a mortgage-document operations reviewer. Use only the retrieved "
        "synthetic case evidence below. Do not infer borrower facts, make credit, legal, or "
        "underwriting decisions, or claim an action has been taken. If the evidence is not "
        "enough to support a next step, say that the reviewer should investigate. Give a brief "
        "evidence-grounded explanation and a cautious suggested next step. Cite supporting "
        "prior cases using their case IDs.\n\n"
        f"Incoming exception:\n{sanitized_query}\n\n"
        f"Retrieved evidence:\n{evidence_context}\n\n"
        "Evidence-grounded reviewer note:"
    )


def _build_request(prompt: str, model: str, base_url: str, stream: bool) -> Request:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.1, "num_predict": 180},
        }
    ).encode("utf-8")
    return Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def generate_grounded_response_stream(
    query: str,
    evidence: Iterable[Tuple[str, str, str]],
    model: str | None = None,
    ollama_url: str | None = None,
) -> Generator[str, None, None]:
    """Stream the LLM response token-by-token.

    Yields text fragments as they arrive from Ollama's NDJSON stream.
    Each yielded string is a partial token (usually a few characters).

    Usage with Streamlit::

        st.write_stream(generate_grounded_response_stream(query, evidence))

    Guardrails are checked on the *full* output after streaming completes;
    if a violation is found the last yielded chunk is a guardrail notice.
    The caller should treat everything yielded so far as provisional until
    the generator is exhausted.
    """
    evidence_items = list(evidence)
    if not query.strip():
        raise ValueError("query must not be empty")
    if not evidence_items:
        raise ValueError("At least one retrieved evidence item is required.")

    is_injection, injection_reason = detect_prompt_injection(query)
    if is_injection:
        raise ValueError(f"Guardrail safety block: {injection_reason}")

    prompt = _build_prompt(query, evidence_items)
    chosen_model = model or os.environ.get("WORKBENCH_LLM_MODEL", DEFAULT_GENERATION_MODEL)
    base_url = (ollama_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
    request = _build_request(prompt, chosen_model, base_url, stream=True)

    full_response_parts: list[str] = []
    try:
        with urlopen(request, timeout=120) as response:
            try:
                line_iter = iter(response)
            except TypeError:
                line_iter = iter(response.read().splitlines(keepends=True))
            for raw_line in line_iter:
                line = (raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")).strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    full_response_parts.append(token)
                    yield token
                if chunk.get("done"):
                    break
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama generation failed with HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama and ensure the model is installed."
        ) from error
    except TimeoutError as error:
        raise RuntimeError("Ollama did not generate a response within 120 seconds.") from error

    full_response = "".join(full_response_parts).strip()
    if not full_response:
        raise RuntimeError("Ollama returned an empty generated response.")

    # Post-stream output guardrail check.
    output_guardrail = validate_llm_output(
        full_response,
        evidence_cases=[case_id for case_id, _, _ in evidence_items],
        evidence_chunks=[chunk for _, chunk, _ in evidence_items],
    )
    if not output_guardrail["is_safe"]:
        violations_str = "; ".join(output_guardrail["violations"])
        # Emit a suffix notice so the reviewer sees the guardrail flag inline.
        yield f"\n\n⚠️ [Guardrail Notice: {violations_str}]"


def generate_grounded_response(
    query: str,
    evidence: Iterable[Tuple[str, str, str]],
    model: str | None = None,
    ollama_url: str | None = None,
) -> str:
    """Blocking wrapper — collects the full streamed response into a string.

    Retained for evals, batch scripts, and any caller that does not need
    token-by-token streaming.
    """
    return "".join(
        generate_grounded_response_stream(query, evidence, model=model, ollama_url=ollama_url)
    )
