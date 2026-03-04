from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.main import retrieve_with_optional_uploads


@dataclass
class EvalItem:
    question: str
    expected_paths: list[str]
    category: str


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        payload = json.loads(row)
        question = str(payload.get("question", "")).strip()
        expected_paths = [str(p).strip() for p in payload.get("expected_paths", []) if str(p).strip()]
        category = str(payload.get("category", "general")).strip() or "general"
        if question and expected_paths:
            items.append(EvalItem(question=question, expected_paths=expected_paths, category=category))
    return items


def expected_hit(path: str, expected_paths: Iterable[str]) -> bool:
    normalized = path.lower()
    for expected in expected_paths:
        probe = expected.lower()
        if probe in normalized:
            return True
    return False


def dcg(binary_relevance: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel <= 0:
            continue
        score += rel / math.log2(i + 1.0)
    return score


def ndcg_at_k(paths: list[str], expected_paths: list[str], k: int) -> float:
    rel = [1 if expected_hit(path, expected_paths) else 0 for path in paths[:k]]
    ideal = sorted(rel, reverse=True)
    denom = dcg(ideal, k)
    if denom <= 0:
        return 0.0
    return dcg(rel, k) / denom


def recall_at_k(paths: list[str], expected_paths: list[str], k: int) -> float:
    for path in paths[:k]:
        if expected_hit(path, expected_paths):
            return 1.0
    return 0.0


def evaluate(items: list[EvalItem], top_k: int) -> dict:
    if not items:
        return {
            "questions": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "details": [],
        }

    details: list[dict] = []
    recall5_total = 0.0
    recall10_total = 0.0
    ndcg10_total = 0.0

    for item in items:
        citations, _, _ = retrieve_with_optional_uploads(
            question=item.question,
            top_k=max(top_k, 10),
            uploaded_files=[],
            scope="repo",
            project_id="nshmp-main",
        )
        paths = [citation.file_path for citation in citations]
        r5 = recall_at_k(paths, item.expected_paths, 5)
        r10 = recall_at_k(paths, item.expected_paths, 10)
        ndcg10 = ndcg_at_k(paths, item.expected_paths, 10)

        recall5_total += r5
        recall10_total += r10
        ndcg10_total += ndcg10

        details.append(
            {
                "question": item.question,
                "category": item.category,
                "expected_paths": item.expected_paths,
                "retrieved_paths": paths[:10],
                "recall_at_5": round(r5, 3),
                "recall_at_10": round(r10, 3),
                "ndcg_at_10": round(ndcg10, 3),
            }
        )

    n = len(items)
    return {
        "questions": n,
        "recall_at_5": round(recall5_total / n, 3),
        "recall_at_10": round(recall10_total / n, 3),
        "ndcg_at_10": round(ndcg10_total / n, 3),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LegacyLens retrieval evaluation")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "dataset.jsonl"),
        help="Path to JSONL dataset with question + expected_paths",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K to retrieve per question")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    parser.add_argument("--out", default="", help="Optional output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    items = load_dataset(dataset_path)

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    report = evaluate(items, top_k=max(args.top_k, 1))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote detailed report: {out_path}")


if __name__ == "__main__":
    main()
