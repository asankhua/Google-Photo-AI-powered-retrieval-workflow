"""
Groq-powered classifier.

Takes one piece of raw user feedback and returns structured JSON tagging:
  - is_retrieval_related  (filters out noise)
  - failure_mode          (the funnel stage that broke)
  - photo_type
  - remembered_cues[]     (what the user DID remember)
  - forgotten_cues[]      (what they had forgotten / lacked)
  - severity              (1-5)
  - evidence_quote        (verbatim snippet grounding the classification)

Goes BEYOND sentiment: it locates *where in the retrieval journey* each user broke
down, which lets us compare and rank opportunity areas.

Uses Groq (OpenAI-compatible SDK). Set GROQ_API_KEY in the environment.
"""
import json
import os
import time
from typing import Dict, Any

from groq import Groq, RateLimitError

from .taxonomy import (FAILURE_MODES, PHOTO_TYPES, CUE_TYPES, SEVERITY_SCALE,
                       QUERY_STRATEGIES)

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from env
    return _client


def _build_prompt(text: str) -> str:
    modes = "\n".join(f"  - {k}: {v}" for k, v in FAILURE_MODES.items())
    sev = "\n".join(f"  {k} = {v}" for k, v in SEVERITY_SCALE.items())
    return f"""You are an analyst studying why people fail to retrieve photos they \
remember but cannot precisely describe, in Google Photos.

Classify the user feedback below. Retrieval succeeds only if the user passes all \
four funnel stages in order: EXPRESSION -> UNDERSTANDING -> EVALUATION -> REFINEMENT. \
Tag the FIRST stage that breaks.

FAILURE MODES:
{modes}

PHOTO TYPES (pick one): {", ".join(PHOTO_TYPES)}

CUE TYPES (choose any that apply): {", ".join(CUE_TYPES)}

QUERY STRATEGIES (how they searched; pick any that apply): {", ".join(QUERY_STRATEGIES)}

SEVERITY (1-5):
{sev}

Return ONLY valid JSON with exactly these keys:
{{
  "is_retrieval_related": true/false,
  "failure_mode": one of {list(FAILURE_MODES.keys())},
  "photo_type": one of the PHOTO TYPES,
  "remembered_cues": [subset of CUE TYPES the user clearly remembered],
  "forgotten_cues": [subset of CUE TYPES the user lacked or forgot],
  "query_strategies": [subset of QUERY STRATEGIES the user used],
  "query_text": "the actual words they searched, if stated, else null",
  "severity": integer 1-5,
  "evidence_quote": "short verbatim snippet from the feedback",
  "reasoning": "one sentence"
}}

If the feedback is not about retrieving a remembered photo, set \
is_retrieval_related=false and failure_mode="NOT_RETRIEVAL".

USER FEEDBACK:
\"\"\"{text}\"\"\"
"""


def _safe_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


def classify_one(text: str) -> Dict[str, Any]:
    """Classify a single feedback string. Returns the parsed dict (+ raw text)."""
    for attempt in range(6):  # free tier: 8k tokens a minute, so wait and retry
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                max_tokens=2000,          # gpt-oss reasons before emitting JSON
                temperature=0,
                reasoning_effort="low",   # keep reasoning short so JSON completes
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": _build_prompt(text)}],
            )
            break
        except RateLimitError as e:
            if "tokens per day" in str(e) or attempt == 5:
                raise  # daily cap: waiting won't help
            time.sleep(20 * (attempt + 1))
    parsed = _safe_json(resp.choices[0].message.content)
    parsed["source_text"] = text
    return parsed


def classify_batch(texts, on_progress=None):
    """Classify a list of feedback strings.

    Returns exactly one result per input (1:1 alignment), so callers can safely
    zip the results back against the raw records to attach source/url metadata.
    Empty items and API errors get a NOT_RETRIEVAL placeholder rather than being
    dropped, which would silently misalign that metadata.
    """
    results = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            results.append({"source_text": t or "", "skipped": True,
                            "is_retrieval_related": False,
                            "failure_mode": "NOT_RETRIEVAL"})
        else:
            try:
                results.append(classify_one(t))
            except Exception as e:  # keep the pipeline resilient
                results.append({"error": str(e), "source_text": t,
                                "is_retrieval_related": False,
                                "failure_mode": "NOT_RETRIEVAL"})
        if on_progress:
            on_progress(i + 1, len(texts))
    return results
