from __future__ import annotations

import re
from typing import Any

PLAN_VECTOR_ONLY = "VECTOR_ONLY"
PLAN_KEYWORD_ONLY = "KEYWORD_ONLY"
PLAN_GRAPH_ONLY = "GRAPH_ONLY"
PLAN_KEYWORD_PLUS_VECTOR = "KEYWORD_PLUS_VECTOR"
PLAN_GRAPH_PLUS_VECTOR = "GRAPH_PLUS_VECTOR"
PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR = "GRAPH_PLUS_KEYWORD_PLUS_VECTOR"

ROUTER_PINECONE_TOP_K = 20
ROUTER_TIMEOUT_MS = 2500
LOW_CONFIDENCE_TOP_SCORE_THRESHOLD = 0.45
LOW_CONFIDENCE_MIN_UNIQUE_RESULTS = 3

CALL_PATTERN = re.compile(r"\bcall\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
IDENTIFIER_ALLCAPS_PATTERN = re.compile(r"\b[A-Z0-9_]{3,}\b")
IDENTIFIER_SNAKE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
IDENTIFIER_DIGIT_PATTERN = re.compile(r"\b[a-zA-Z_]*\d+[a-zA-Z0-9_]*\b")
EXTENSION_PATTERN = re.compile(r"\.(?:f|f90|inc|for)\b", re.IGNORECASE)
NON_IDENTIFIER_CALL_TERMS = {"chain"}

STRUCTURE_HINTS = (
    "entry point",
    "dependencies",
    "what calls",
    "call chain",
    "data flow",
    "impact",
    "blast radius",
)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(normalized)
    return output


def detect_route_signals(question: str) -> dict[str, Any]:
    text = str(question or "")
    lowered = text.lower()

    call_terms: list[str] = []
    for match in CALL_PATTERN.findall(text):
        name = str(match or "").strip()
        if not name:
            continue
        # "call chain" is a structure intent phrase, not an identifier lookup.
        if name.lower() in NON_IDENTIFIER_CALL_TERMS:
            continue
        call_terms.append(f"CALL {name}")
    allcaps_terms = IDENTIFIER_ALLCAPS_PATTERN.findall(text)
    snake_terms = IDENTIFIER_SNAKE_PATTERN.findall(text)
    digit_terms = IDENTIFIER_DIGIT_PATTERN.findall(text)
    has_extensions = bool(EXTENSION_PATTERN.search(text))

    identifier_terms = _dedupe_keep_order(
        [
            *call_terms,
            *allcaps_terms,
            *snake_terms,
            *digit_terms,
        ]
    )[:12]

    structure_intent = any(phrase in lowered for phrase in STRUCTURE_HINTS)
    identifier_heavy = bool(call_terms or allcaps_terms or snake_terms or digit_terms or has_extensions)

    return {
        "identifier_terms": identifier_terms,
        "structure_intent": structure_intent,
        "has_extensions": has_extensions,
        "identifier_heavy": identifier_heavy,
    }


def select_retrieval_plan(question: str, mode: str | None = None) -> str:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "graph":
        return PLAN_GRAPH_ONLY

    signals = detect_route_signals(question)

    # Rule A has precedence over B/C.
    if signals["identifier_heavy"]:
        return PLAN_KEYWORD_PLUS_VECTOR

    # Rule B.
    if signals["structure_intent"]:
        return PLAN_GRAPH_PLUS_VECTOR

    # Rule C.
    return PLAN_VECTOR_ONLY


def default_route_budgets() -> dict[str, int]:
    return {
        "pinecone_top_k": ROUTER_PINECONE_TOP_K,
        "timeout_ms": ROUTER_TIMEOUT_MS,
    }


def _citation_parts(citation: Any) -> tuple[str, int, int, float]:
    if isinstance(citation, dict):
        file_path = str(citation.get("file_path", ""))
        line_start = int(citation.get("line_start") or 0)
        line_end = int(citation.get("line_end") or line_start)
        score = float(citation.get("score") or 0.0)
        return file_path, line_start, line_end, score

    file_path = str(getattr(citation, "file_path", ""))
    line_start = int(getattr(citation, "line_start", 0) or 0)
    line_end = int(getattr(citation, "line_end", line_start) or line_start)
    score = float(getattr(citation, "score", 0.0) or 0.0)
    return file_path, line_start, line_end, score


def low_confidence_reason(
    citations: list[Any],
    top_score_threshold: float = LOW_CONFIDENCE_TOP_SCORE_THRESHOLD,
    min_unique_results: int = LOW_CONFIDENCE_MIN_UNIQUE_RESULTS,
) -> str | None:
    if not citations:
        return "no_matches"

    ranges: list[tuple[str, int, int]] = []
    scores: list[float] = []
    for citation in citations:
        file_path, line_start, line_end, score = _citation_parts(citation)
        ranges.append((file_path, line_start, line_end))
        scores.append(score)

    unique_count = len(set(ranges))
    if unique_count < len(ranges):
        return "duplicate_ranges"

    if unique_count < max(int(min_unique_results), 1):
        return "insufficient_unique_results"

    top_score = max(scores) if scores else 0.0
    if top_score < float(top_score_threshold):
        return "top_score_below_threshold"

    return None


def escalated_plan(current_plan: str, did_escalate: bool) -> str | None:
    if did_escalate:
        return None
    if current_plan == PLAN_VECTOR_ONLY:
        return PLAN_KEYWORD_PLUS_VECTOR
    if current_plan == PLAN_GRAPH_PLUS_VECTOR:
        return PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR
    return None


def route_debug_template(route: str, signals: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "route": route,
        "signals": signals or {
            "identifier_terms": [],
            "structure_intent": False,
            "has_extensions": False,
        },
        "budgets": default_route_budgets(),
        "steps": [],
        "escalation": {
            "did_escalate": False,
            "reason": None,
        },
    }
