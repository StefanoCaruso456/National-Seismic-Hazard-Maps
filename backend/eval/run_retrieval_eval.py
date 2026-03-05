from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ingest import FORTRAN_EXTENSIONS, chunk_fortran_file, discover_fortran_files

IDENTIFIER_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-/]{3,}")


@dataclass
class EvalQuery:
    query_id: str
    category: str
    question: str
    expected_paths: list[str]
    expected_terms: list[str]
    negative: bool


@dataclass
class CandidateRow:
    file_path: str
    line_start: int
    line_end: int
    snippet: str
    semantic_rank: int
    rerank_rank: int
    semantic_score: float
    hybrid_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LegacyLens retrieval evaluation against API endpoint")
    parser.add_argument("--queries", default="eval/queries.json", help="Path to eval query JSON")
    parser.add_argument(
        "--endpoint",
        default="https://www.nationalseismichazardmaps.com/api/search",
        help="Search endpoint URL",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Requested top_k for retrieval")
    parser.add_argument("--timeout-seconds", type=float, default=25.0, help="HTTP timeout")
    parser.add_argument("--out", default="eval/results.json", help="JSON output path")
    parser.add_argument("--report", default="eval/REPORT.md", help="Markdown summary output path")
    return parser.parse_args()


def load_queries(path: Path) -> list[EvalQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[EvalQuery] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip() or f"Q{len(rows)+1:03d}"
        category = str(item.get("category", "general")).strip() or "general"
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        expected_paths = [str(x).strip() for x in item.get("expected_paths", []) if str(x).strip()]
        expected_terms = [str(x).strip().lower() for x in item.get("expected_terms", []) if str(x).strip()]
        negative = bool(item.get("negative", False))
        rows.append(
            EvalQuery(
                query_id=qid,
                category=category,
                question=question,
                expected_paths=expected_paths,
                expected_terms=expected_terms,
                negative=negative,
            )
        )
    return rows


def post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> tuple[dict[str, Any], int]:
    cmd = [
        "curl",
        "-sS",
        "-m",
        f"{max(timeout_seconds, 3.0):.0f}",
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: application/json",
        "-d",
        json.dumps(payload),
        "-w",
        "\n%{http_code}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = proc.stdout or ""
    if not output and proc.stderr:
        return {"detail": proc.stderr.strip() or "curl_failed"}, 599

    if "\n" in output:
        body, status_line = output.rsplit("\n", 1)
    else:
        body, status_line = output, "0"
    try:
        status = int(status_line.strip())
    except ValueError:
        status = 0
    if status <= 0:
        status = 599

    try:
        parsed = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        parsed = {"detail": body.strip() or "invalid_json_response"}
    return parsed, status


def normalize_candidate_row(raw: dict[str, Any], fallback_rank: int) -> CandidateRow:
    return CandidateRow(
        file_path=str(raw.get("file_path", "unknown")),
        line_start=int(raw.get("line_start") or 1),
        line_end=int(raw.get("line_end") or int(raw.get("line_start") or 1)),
        snippet=str(raw.get("snippet") or ""),
        semantic_rank=int(raw.get("semantic_rank") or fallback_rank),
        rerank_rank=int(raw.get("rerank_rank") or fallback_rank),
        semantic_score=float(raw.get("semantic_score") or raw.get("score") or 0.0),
        hybrid_score=float(raw.get("hybrid_score") or raw.get("score") or 0.0),
    )


def dedupe_rows(rows: list[CandidateRow]) -> list[CandidateRow]:
    out: list[CandidateRow] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (row.file_path, row.line_start, row.line_end)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_candidate_views(response_payload: dict[str, Any]) -> tuple[list[CandidateRow], list[CandidateRow]]:
    debug = response_payload.get("debug", {}) if isinstance(response_payload, dict) else {}
    retrieval = debug.get("retrieval", {}) if isinstance(debug, dict) else {}
    index_block = retrieval.get("index", {}) if isinstance(retrieval, dict) else {}
    lexical_block = retrieval.get("index_lexical", {}) if isinstance(retrieval, dict) else {}
    candidates_raw = []
    if isinstance(index_block, dict):
        candidates_raw.extend(index_block.get("candidates", []) or [])
    if isinstance(lexical_block, dict):
        candidates_raw.extend(lexical_block.get("candidates", []) or [])

    candidates: list[CandidateRow] = []
    for i, row in enumerate(candidates_raw, start=1):
        if not isinstance(row, dict):
            continue
        candidates.append(normalize_candidate_row(row, fallback_rank=i))

    matches = response_payload.get("matches", []) if isinstance(response_payload, dict) else []
    improved_from_matches: list[CandidateRow] = []
    for i, match in enumerate(matches, start=1):
        if not isinstance(match, dict):
            continue
        improved_from_matches.append(
            CandidateRow(
                file_path=str(match.get("file_path", "unknown")),
                line_start=int(match.get("line_start") or 1),
                line_end=int(match.get("line_end") or int(match.get("line_start") or 1)),
                snippet=str(match.get("snippet") or ""),
                semantic_rank=i,
                rerank_rank=i,
                semantic_score=float(match.get("score") or 0.0),
                hybrid_score=float(match.get("score") or 0.0),
            )
        )

    improved = dedupe_rows(improved_from_matches)
    if not improved:
        improved = dedupe_rows(sorted(candidates, key=lambda row: (row.rerank_rank, -row.hybrid_score)))

    baseline_proxy = dedupe_rows(sorted(candidates, key=lambda row: (row.semantic_rank, -row.semantic_score)))
    if not baseline_proxy:
        baseline_proxy = dedupe_rows(improved_from_matches)
    return baseline_proxy, improved


def query_identifiers(question: str) -> list[str]:
    lowered = question.lower()
    tokens = [token.lower() for token in IDENTIFIER_TOKEN_PATTERN.findall(lowered)]
    tokens = [token for token in tokens if any(ch.isdigit() for ch in token) or "_" in token or "-" in token or token.isupper()]
    if not tokens:
        tokens = [token.lower() for token in IDENTIFIER_TOKEN_PATTERN.findall(lowered) if len(token) >= 4]
    return tokens[:8]


def path_hit(file_path: str, expected_paths: list[str]) -> bool:
    probe = file_path.lower()
    for expected in expected_paths:
        if expected.lower() in probe:
            return True
    return False


def judge_relevance(query: EvalQuery, row: CandidateRow) -> int:
    if query.negative:
        return 0

    source = f"{row.file_path}\n{row.snippet}".lower()
    path_match = path_hit(row.file_path, query.expected_paths)
    term_hits = sum(1 for term in query.expected_terms if term in source)
    ident_hits = sum(1 for token in query_identifiers(query.question) if token in source)

    if path_match and term_hits > 0:
        return 3
    if path_match:
        return 2
    if term_hits > 0 and ident_hits > 0:
        return 2
    if term_hits > 0 or ident_hits > 0:
        return 1
    return 0


def precision_at_k(scores: list[int], k: int) -> float:
    window = scores[:k]
    if len(window) < k:
        window = window + [0] * (k - len(window))
    relevant = sum(1 for score in window if score >= 2)
    return relevant / max(k, 1)


def mrr_at_k(scores: list[int], k: int) -> float:
    for idx, score in enumerate(scores[:k], start=1):
        if score >= 2:
            return 1.0 / idx
    return 0.0


def recall_at_k(scores: list[int], k: int) -> float:
    return 1.0 if any(score >= 2 for score in scores[:k]) else 0.0


def safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def line_bounds_valid(repo_root: Path, file_path: str, line_start: int, line_end: int) -> tuple[bool, str]:
    rel = file_path.replace("\\", "/").lstrip("/")
    if not rel:
        return False, ""
    target = (repo_root / rel).resolve()
    if not target.exists() or not target.is_file():
        return False, ""
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = target.read_text(encoding="latin-1", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return False, ""
    start = max(1, int(line_start))
    end = max(start, int(line_end))
    if start > len(lines):
        return False, ""
    end = min(end, len(lines))
    segment = "\n".join(lines[start - 1 : end]).lower()
    return True, segment


def citation_accuracy(
    repo_root: Path,
    sampled_queries: list[EvalQuery],
    per_query_ranked: dict[str, list[CandidateRow]],
) -> float:
    checks = 0
    passes = 0
    for query in sampled_queries:
        rows = per_query_ranked.get(query.query_id, [])[:3]
        for row in rows:
            if query.negative:
                continue
            valid, segment = line_bounds_valid(repo_root, row.file_path, row.line_start, row.line_end)
            checks += 1
            if not valid:
                continue
            if query.expected_terms:
                if any(term in segment for term in query.expected_terms):
                    passes += 1
            elif query.expected_paths:
                if path_hit(row.file_path, query.expected_paths):
                    passes += 1
            else:
                passes += 1
    if checks == 0:
        return 0.0
    return passes / checks


def compute_coverage(repo_root: Path) -> dict[str, Any]:
    files = discover_fortran_files(repo_root=repo_root, extensions=set(FORTRAN_EXTENSIONS))
    if not files:
        return {
            "total_files": 0,
            "files_with_chunks": 0,
            "file_coverage": 0.0,
            "total_chunks": 0,
            "chunks_with_required_metadata": 0,
            "chunk_metadata_coverage": 0.0,
        }

    files_with_chunks = 0
    total_chunks = 0
    valid_chunks = 0
    for path in files:
        chunks = chunk_fortran_file(path=path, repo_root=repo_root, min_tokens=200, max_tokens=500)
        if chunks:
            files_with_chunks += 1
        for chunk in chunks:
            total_chunks += 1
            if (
                chunk.file_path
                and int(chunk.line_start) > 0
                and int(chunk.line_end) >= int(chunk.line_start)
                and chunk.symbol_type
            ):
                valid_chunks += 1

    return {
        "total_files": len(files),
        "files_with_chunks": files_with_chunks,
        "file_coverage": files_with_chunks / max(len(files), 1),
        "total_chunks": total_chunks,
        "chunks_with_required_metadata": valid_chunks,
        "chunk_metadata_coverage": valid_chunks / max(total_chunks, 1),
    }


def summarize_profile(
    queries: list[EvalQuery],
    ranked_rows_by_query: dict[str, list[CandidateRow]],
    latencies_ms: list[float],
) -> dict[str, Any]:
    p5_values: list[float] = []
    mrr5_values: list[float] = []
    r50_values: list[float] = []
    details: list[dict[str, Any]] = []

    category_metrics: dict[str, dict[str, Any]] = {}

    for query in queries:
        rows = ranked_rows_by_query.get(query.query_id, [])
        scores = [judge_relevance(query, row) for row in rows]
        p5 = precision_at_k(scores, 5)
        mrr5 = mrr_at_k(scores, 5)
        r50 = recall_at_k(scores, 50)

        p5_values.append(p5)
        mrr5_values.append(mrr5)
        r50_values.append(r50)

        bucket = category_metrics.setdefault(
            query.category,
            {"count": 0, "p5": [], "mrr5": [], "r50": []},
        )
        bucket["count"] += 1
        bucket["p5"].append(p5)
        bucket["mrr5"].append(mrr5)
        bucket["r50"].append(r50)

        details.append(
            {
                "id": query.query_id,
                "category": query.category,
                "question": query.question,
                "negative": query.negative,
                "precision_at_5": round(p5, 4),
                "mrr_at_5": round(mrr5, 4),
                "recall_at_50": round(r50, 4),
                "top_paths": [row.file_path for row in rows[:10]],
                "top_scores": scores[:10],
            }
        )

    for bucket in category_metrics.values():
        bucket["precision_at_5"] = round(statistics.fmean(bucket["p5"]) if bucket["p5"] else 0.0, 4)
        bucket["mrr_at_5"] = round(statistics.fmean(bucket["mrr5"]) if bucket["mrr5"] else 0.0, 4)
        bucket["recall_at_50"] = round(statistics.fmean(bucket["r50"]) if bucket["r50"] else 0.0, 4)
        del bucket["p5"]
        del bucket["mrr5"]
        del bucket["r50"]

    return {
        "precision_at_5": round(statistics.fmean(p5_values) if p5_values else 0.0, 4),
        "mrr_at_5": round(statistics.fmean(mrr5_values) if mrr5_values else 0.0, 4),
        "recall_at_50": round(statistics.fmean(r50_values) if r50_values else 0.0, 4),
        "latency_ms": {
            "p50": round(safe_quantile(latencies_ms, 0.5), 2),
            "p95": round(safe_quantile(latencies_ms, 0.95), 2),
            "mean": round(statistics.fmean(latencies_ms) if latencies_ms else 0.0, 2),
        },
        "category_metrics": category_metrics,
        "details": details,
    }


def write_report(path: Path, results: dict[str, Any]) -> None:
    baseline = results.get("profiles", {}).get("baseline_proxy", {})
    improved = results.get("profiles", {}).get("improved", {})
    delta = results.get("delta", {})
    coverage = results.get("coverage", {})

    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Generated: {results.get('generated_at')}",
        f"Endpoint: {results.get('endpoint')}",
        f"Queries: {results.get('query_count')}",
        "",
        "## Metrics (Baseline Proxy vs Improved)",
        "",
        "| Metric | Baseline Proxy | Improved | Delta |",
        "|---|---:|---:|---:|",
        f"| Precision@5 | {baseline.get('precision_at_5', 0):.4f} | {improved.get('precision_at_5', 0):.4f} | {delta.get('precision_at_5', 0):+.4f} |",
        f"| MRR@5 | {baseline.get('mrr_at_5', 0):.4f} | {improved.get('mrr_at_5', 0):.4f} | {delta.get('mrr_at_5', 0):+.4f} |",
        f"| Recall@50 | {baseline.get('recall_at_50', 0):.4f} | {improved.get('recall_at_50', 0):.4f} | {delta.get('recall_at_50', 0):+.4f} |",
        "",
        "## Latency",
        "",
        "| Profile | p50 (ms) | p95 (ms) | mean (ms) |",
        "|---|---:|---:|---:|",
        f"| Baseline Proxy | {baseline.get('latency_ms', {}).get('p50', 0):.2f} | {baseline.get('latency_ms', {}).get('p95', 0):.2f} | {baseline.get('latency_ms', {}).get('mean', 0):.2f} |",
        f"| Improved | {improved.get('latency_ms', {}).get('p50', 0):.2f} | {improved.get('latency_ms', {}).get('p95', 0):.2f} | {improved.get('latency_ms', {}).get('mean', 0):.2f} |",
        "",
        "## Citation Accuracy",
        "",
        f"- Baseline Proxy: {results.get('citation_accuracy', {}).get('baseline_proxy', 0):.4f}",
        f"- Improved: {results.get('citation_accuracy', {}).get('improved', 0):.4f}",
        "",
        "## Coverage",
        "",
        f"- Fortran files discovered: {coverage.get('total_files', 0)}",
        f"- Files with chunks: {coverage.get('files_with_chunks', 0)} ({coverage.get('file_coverage', 0):.2%})",
        f"- Chunks with required metadata: {coverage.get('chunks_with_required_metadata', 0)}/{coverage.get('total_chunks', 0)} ({coverage.get('chunk_metadata_coverage', 0):.2%})",
        "",
        "## Targets",
        "",
        f"- Precision@5 > 0.70: {'PASS' if improved.get('precision_at_5', 0) > 0.70 else 'FAIL'}",
        f"- Latency p95 < 3000ms: {'PASS' if improved.get('latency_ms', {}).get('p95', 0) < 3000 else 'FAIL'}",
        f"- File coverage = 100%: {'PASS' if coverage.get('file_coverage', 0) >= 1.0 else 'FAIL'}",
        f"- Chunk metadata coverage = 100%: {'PASS' if coverage.get('chunk_metadata_coverage', 0) >= 1.0 else 'FAIL'}",
        "",
        "## Notes",
        "",
        "- Baseline Proxy is computed from semantic rank ordering of the same retrieved candidate set (before deterministic rerank fusion).",
        "- Improved uses final hybrid rerank ordering.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    query_path = Path(args.queries).resolve()
    queries = load_queries(query_path)
    if not queries:
        raise SystemExit(f"No queries loaded from {query_path}")

    endpoint = str(args.endpoint).strip()
    requested_top_k = max(int(args.top_k), 5)

    baseline_rows: dict[str, list[CandidateRow]] = {}
    improved_rows: dict[str, list[CandidateRow]] = {}
    improved_latencies: list[float] = []
    baseline_latencies: list[float] = []
    errors: list[dict[str, Any]] = []

    for idx, query in enumerate(queries, start=1):
        payload = {
            "question": query.question,
            "top_k": requested_top_k,
            "mode": "search",
            "scope": "repo",
            "debug": True,
        }
        started = time.perf_counter()
        response_payload, status = post_json(endpoint, payload, timeout_seconds=max(args.timeout_seconds, 3.0))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if status == 422 and requested_top_k > 20:
            payload["top_k"] = 20
            started = time.perf_counter()
            response_payload, status = post_json(endpoint, payload, timeout_seconds=max(args.timeout_seconds, 3.0))
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        if status >= 400:
            errors.append(
                {
                    "id": query.query_id,
                    "question": query.question,
                    "status": status,
                    "detail": response_payload.get("detail", "request_failed") if isinstance(response_payload, dict) else "request_failed",
                }
            )
            baseline_rows[query.query_id] = []
            improved_rows[query.query_id] = []
            improved_latencies.append(elapsed_ms)
            baseline_latencies.append(elapsed_ms)
            continue

        baseline_proxy, improved = build_candidate_views(response_payload)
        baseline_rows[query.query_id] = baseline_proxy
        improved_rows[query.query_id] = improved

        improved_latencies.append(elapsed_ms)

        debug = response_payload.get("debug", {}) if isinstance(response_payload, dict) else {}
        retrieval_timings = (debug.get("retrieval", {}) or {}).get("index", {}).get("timings_ms", {})
        top_timings = debug.get("timings_ms", {}) if isinstance(debug, dict) else {}
        rerank_ms = float((retrieval_timings or {}).get("rerank", 0.0) or 0.0)
        lexical_ms = float((top_timings or {}).get("lexical", 0.0) or 0.0)
        context_ms = float((top_timings or {}).get("context_assembly", 0.0) or 0.0)
        baseline_estimate = max(0.0, elapsed_ms - rerank_ms - lexical_ms - context_ms)
        baseline_latencies.append(baseline_estimate)

        if idx % 5 == 0:
            print(f"Evaluated {idx}/{len(queries)} queries...")

    baseline_profile = summarize_profile(queries, baseline_rows, baseline_latencies)
    improved_profile = summarize_profile(queries, improved_rows, improved_latencies)

    repo_root = Path(__file__).resolve().parents[2]
    sampled_positive = [query for query in queries if not query.negative][:10]
    citation_scores = {
        "baseline_proxy": round(citation_accuracy(repo_root, sampled_positive, baseline_rows), 4),
        "improved": round(citation_accuracy(repo_root, sampled_positive, improved_rows), 4),
    }
    coverage = compute_coverage(repo_root=repo_root)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "query_count": len(queries),
        "errors": errors,
        "profiles": {
            "baseline_proxy": baseline_profile,
            "improved": improved_profile,
        },
        "delta": {
            "precision_at_5": round(improved_profile["precision_at_5"] - baseline_profile["precision_at_5"], 4),
            "mrr_at_5": round(improved_profile["mrr_at_5"] - baseline_profile["mrr_at_5"], 4),
            "recall_at_50": round(improved_profile["recall_at_50"] - baseline_profile["recall_at_50"], 4),
        },
        "citation_accuracy": citation_scores,
        "coverage": {
            **coverage,
            "file_coverage": round(float(coverage.get("file_coverage", 0.0)), 4),
            "chunk_metadata_coverage": round(float(coverage.get("chunk_metadata_coverage", 0.0)), 4),
        },
        "targets": {
            "precision_at_5_gt_0_70": bool(improved_profile["precision_at_5"] > 0.70),
            "latency_p95_lt_3000ms": bool(improved_profile["latency_ms"]["p95"] < 3000),
            "file_coverage_100pct": bool(coverage.get("file_coverage", 0.0) >= 1.0),
            "chunk_metadata_coverage_100pct": bool(coverage.get("chunk_metadata_coverage", 0.0) >= 1.0),
        },
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, result)

    print(json.dumps({
        "precision_at_5": result["profiles"]["improved"]["precision_at_5"],
        "mrr_at_5": result["profiles"]["improved"]["mrr_at_5"],
        "recall_at_50": result["profiles"]["improved"]["recall_at_50"],
        "latency_p95_ms": result["profiles"]["improved"]["latency_ms"]["p95"],
        "coverage": result["coverage"],
        "errors": len(errors),
        "out": str(out_path),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
