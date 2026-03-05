from __future__ import annotations

from collections import defaultdict
from typing import Any


def normalize_file_path(path: str | None) -> str | None:
    value = str(path or "").replace("\\", "/").strip().lstrip("/")
    if not value:
        return None
    if value.startswith("../"):
        return None
    return value


def _add_path_score(
    score_map: dict[str, float],
    reasons: dict[str, list[str]],
    path: str | None,
    weight: float,
    reason: str,
) -> None:
    normalized = normalize_file_path(path)
    if not normalized:
        return
    score_map[normalized] = score_map.get(normalized, 0.0) + max(weight, 0.0)
    reasons[normalized].append(reason)


def _iter_context_refs(payload: dict[str, Any], section: str) -> list[dict[str, Any]]:
    raw = payload.get(section, {})
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, Any]] = []
    for values in raw.values():
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _iter_impact_rows(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    by_depth = payload.get("byDepth", {})
    if not isinstance(by_depth, dict):
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for depth_key, values in by_depth.items():
        try:
            depth = int(depth_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, dict):
                rows.append((depth, row))
    return rows


def extract_ranked_candidate_files(
    query_result: dict[str, Any] | None,
    context_result: dict[str, Any] | None = None,
    impact_result: dict[str, Any] | None = None,
    max_candidate_files: int = 50,
) -> tuple[list[str], list[dict[str, Any]]]:
    query_payload = query_result if isinstance(query_result, dict) else {}
    context_payload = context_result if isinstance(context_result, dict) else {}
    impact_payload = impact_result if isinstance(impact_result, dict) else {}

    score_map: dict[str, float] = {}
    reasons: dict[str, list[str]] = defaultdict(list)

    process_priority: dict[str, float] = {}
    processes = query_payload.get("processes", [])
    if isinstance(processes, list):
        for process in processes:
            if not isinstance(process, dict):
                continue
            pid = str(process.get("id", "")).strip()
            if not pid:
                continue
            try:
                process_priority[pid] = float(process.get("priority", 0.0) or 0.0)
            except (TypeError, ValueError):
                process_priority[pid] = 0.0

    process_symbols = query_payload.get("process_symbols", [])
    if isinstance(process_symbols, list):
        for row in process_symbols:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("process_id", "")).strip()
            priority_boost = min(max(process_priority.get(pid, 0.0), 0.0), 10.0) * 0.06
            _add_path_score(
                score_map,
                reasons,
                row.get("filePath"),
                2.0 + priority_boost,
                "query.process_symbols",
            )

    definitions = query_payload.get("definitions", [])
    if isinstance(definitions, list):
        for row in definitions:
            if not isinstance(row, dict):
                continue
            _add_path_score(score_map, reasons, row.get("filePath"), 1.1, "query.definitions")

    symbol = context_payload.get("symbol")
    if isinstance(symbol, dict):
        _add_path_score(score_map, reasons, symbol.get("filePath"), 4.0, "context.symbol")

    for ref in _iter_context_refs(context_payload, "incoming"):
        _add_path_score(score_map, reasons, ref.get("filePath"), 2.3, "context.incoming")
    for ref in _iter_context_refs(context_payload, "outgoing"):
        _add_path_score(score_map, reasons, ref.get("filePath"), 2.0, "context.outgoing")

    impact_target = impact_payload.get("target")
    if isinstance(impact_target, dict):
        _add_path_score(score_map, reasons, impact_target.get("filePath"), 3.5, "impact.target")

    depth_weights = {1: 3.0, 2: 2.0, 3: 1.2}
    for depth, row in _iter_impact_rows(impact_payload):
        confidence = row.get("confidence", 1.0)
        try:
            confidence_value = min(max(float(confidence), 0.1), 1.0)
        except (TypeError, ValueError):
            confidence_value = 1.0
        weight = depth_weights.get(depth, 0.8) * confidence_value
        _add_path_score(score_map, reasons, row.get("filePath"), weight, f"impact.depth_{depth}")

    ranked = sorted(score_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    limit = max(int(max_candidate_files), 1)
    ranked = ranked[:limit]

    files = [path for path, _ in ranked]
    ranking_debug = [
        {
            "file_path": path,
            "score": round(score, 4),
            "reasons": reasons.get(path, []),
        }
        for path, score in ranked
    ]
    return files, ranking_debug


def should_run_impact(question: str) -> bool:
    lowered = question.lower()
    triggers = (
        "impact",
        "break",
        "change",
        "changing",
        "modify",
        "refactor",
        "rename",
        "dependency",
        "dependencies",
        "blast radius",
    )
    return any(term in lowered for term in triggers)
