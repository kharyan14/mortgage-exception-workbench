from __future__ import annotations

import json
import os
from typing import Iterable, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_GENERATION_MODEL = "llama3.1:latest"


def generate_grounded_response(
    query: str,
    evidence: Iterable[Tuple[str, str, str]],
    model: str | None = None,
    ollama_url: str | None = None,
) -> str:
    evidence_items = list(evidence)
    if not query.strip():
        raise ValueError("query must not be empty")
    if not evidence_items:
        raise ValueError("At least one retrieved evidence item is required.")

    evidence_context = "\n\n".join(
        f"[{index}] Prior case {case_id}\n"
        f"Retrieved chunk: {chunk}\n"
        f"Approved prior resolution: {approved_resolution}"
        for index, (case_id, chunk, approved_resolution) in enumerate(evidence_items, start=1)
    )
    prompt = (
        "You are assisting a mortgage-document operations reviewer. Use only the retrieved "
        "synthetic case evidence below. Do not infer borrower facts, make credit, legal, or "
        "underwriting decisions, or claim an action has been taken. If the evidence is not "
        "enough to support a next step, say that the reviewer should investigate. Give a brief "
        "evidence-grounded explanation and a cautious suggested next step. Cite supporting "
        "prior cases using their case IDs.\n\n"
        f"Incoming exception:\n{query}\n\n"
        f"Retrieved evidence:\n{evidence_context}\n\n"
        "Evidence-grounded reviewer note:"
    )
    request_body = json.dumps(
        {
            "model": model or os.environ.get("WORKBENCH_LLM_MODEL", DEFAULT_GENERATION_MODEL),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 180},
        }
    ).encode("utf-8")
    base_url = (ollama_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
    request = Request(
        f"{base_url}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama generation failed with HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama and ensure the model is installed."
        ) from error
    except TimeoutError as error:
        raise RuntimeError("Ollama did not generate a response within 120 seconds.") from error

    answer = str(result.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty generated response.")
    return answer
