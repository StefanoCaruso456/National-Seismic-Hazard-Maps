from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingest import FORTRAN_EXTENSIONS, chunk_fortran_file, discover_fortran_files, token_count


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit indexing prerequisites for LegacyLens")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to repository root (default: project root)",
    )
    parser.add_argument(
        "--namespace",
        default=settings.pinecone_namespace,
        help="Pinecone namespace to check",
    )
    parser.add_argument(
        "--smoke-query",
        default="",
        help="Optional query string to run an end-to-end retrieval probe against Pinecone",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval smoke query")
    parser.add_argument("--json", action="store_true", help="Output JSON (machine-readable)")
    return parser.parse_args()


def pinecone_stats(index_name: str, namespace: str) -> dict[str, Any]:
    if not settings.pinecone_api_key:
        return {"configured": False, "error": "PINECONE_API_KEY is not configured"}
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        if isinstance(stats, dict):
            raw = stats
        else:
            raw = getattr(stats, "to_dict", lambda: {"raw": str(stats)})()

        ns_map = raw.get("namespaces") if isinstance(raw, dict) else None
        ns_count = None
        if isinstance(ns_map, dict):
            ns_entry = ns_map.get(namespace)
            if isinstance(ns_entry, dict):
                ns_count = safe_int(ns_entry.get("vector_count"))

        return {
            "configured": True,
            "index": index_name,
            "namespace": namespace,
            "namespace_vector_count": ns_count,
            "raw": raw,
        }
    except Exception as exc:
        return {"configured": True, "error": str(exc)}


def normalize_matches(query_response: object) -> list:
    if hasattr(query_response, "matches"):
        return query_response.matches or []
    if isinstance(query_response, dict):
        return query_response.get("matches", [])
    return []


def normalize_metadata(match: object) -> dict:
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def match_score(match: object) -> float:
    value = getattr(match, "score", None)
    if value is None and isinstance(match, dict):
        value = match.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def retrieval_smoke(index_name: str, namespace: str, question: str, top_k: int) -> dict[str, Any]:
    clean_question = question.strip()
    if not clean_question:
        return {"enabled": False}
    if not settings.openai_api_key:
        return {"enabled": True, "question": clean_question, "error": "OPENAI_API_KEY is not configured"}
    if not settings.pinecone_api_key:
        return {"enabled": True, "question": clean_question, "error": "PINECONE_API_KEY is not configured"}

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        openai_client = OpenAI(api_key=settings.openai_api_key)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(index_name)

        vector = openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=clean_question,
        ).data[0].embedding
        results = index.query(
            vector=vector,
            top_k=max(top_k, 1),
            include_metadata=True,
            namespace=namespace,
        )
        matches = normalize_matches(results)
        preview = []
        for match in matches[:5]:
            metadata = normalize_metadata(match)
            chunk_text = (
                metadata.get("chunk_text")
                or metadata.get("text")
                or metadata.get("content")
                or metadata.get("code")
                or ""
            )
            preview.append(
                {
                    "file_path": metadata.get("file_path"),
                    "line_start": metadata.get("line_start"),
                    "line_end": metadata.get("line_end"),
                    "score": round(match_score(match), 4),
                    "snippet_preview": chunk_text[:180],
                }
            )

        return {
            "enabled": True,
            "question": clean_question,
            "top_k": max(top_k, 1),
            "match_count": len(matches),
            "preview": preview,
        }
    except Exception as exc:
        return {"enabled": True, "question": clean_question, "error": str(exc)}


def gate_summary(report: dict[str, Any]) -> dict[str, str]:
    file_count = int(report.get("file_count", 0))
    chunk_count = int(report.get("chunk_count", 0))
    chunk_stats = report.get("chunk_tokens", {})
    pinecone = report.get("pinecone", {})
    smoke = report.get("retrieval_smoke", {"enabled": False})

    avg_tokens = int(chunk_stats.get("avg", 0))
    pct_within = float(chunk_stats.get("pct_within_200_500", 0))
    namespace_count = safe_int(pinecone.get("namespace_vector_count"))

    gates = {
        "fortran_files_discovered": "PASS" if file_count > 0 else "FAIL",
        "syntax_aware_chunks_generated": "PASS" if chunk_count > 0 else "FAIL",
        "chunking_quality_200_500_window": "PASS" if 200 <= avg_tokens <= 500 and pct_within >= 80 else "FAIL",
        "pinecone_namespace_populated": "PASS" if namespace_count is not None and namespace_count > 0 else "FAIL",
    }

    if smoke.get("enabled"):
        if "error" in smoke:
            gates["semantic_retrieval_smoke_test"] = "FAIL"
        else:
            gates["semantic_retrieval_smoke_test"] = "PASS" if smoke.get("match_count", 0) > 0 else "FAIL"
    else:
        gates["semantic_retrieval_smoke_test"] = "SKIP"

    return gates


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    files = discover_fortran_files(repo_root=repo_root, extensions=set(FORTRAN_EXTENSIONS))
    chunks = []
    for path in files:
        chunks.extend(chunk_fortran_file(path=path, repo_root=repo_root, min_tokens=200, max_tokens=500))

    token_counts = [token_count(c.chunk_text) for c in chunks]
    within = [t for t in token_counts if 200 <= t <= 500]

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "fortran_extensions": sorted(FORTRAN_EXTENSIONS),
        "file_count": len(files),
        "chunk_count": len(chunks),
        "chunk_tokens": {
            "min": min(token_counts) if token_counts else 0,
            "avg": int(sum(token_counts) / len(token_counts)) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "pct_within_200_500": round((len(within) / len(token_counts)) * 100, 2) if token_counts else 0,
        },
        "pinecone": pinecone_stats(settings.pinecone_index_name, args.namespace),
    }
    report["retrieval_smoke"] = retrieval_smoke(
        index_name=settings.pinecone_index_name,
        namespace=args.namespace,
        question=args.smoke_query,
        top_k=args.top_k,
    )
    report["gates"] = gate_summary(report)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("LegacyLens Audit")
    print(f"- Repo root: {report['repo_root']}")
    print(f"- Fortran files: {report['file_count']}")
    print(f"- Chunks: {report['chunk_count']}")
    print(
        "- Chunk tokens: "
        f"min={report['chunk_tokens']['min']} "
        f"avg={report['chunk_tokens']['avg']} "
        f"max={report['chunk_tokens']['max']} "
        f"within_200_500={report['chunk_tokens']['pct_within_200_500']}%"
    )
    pinecone_section = report["pinecone"]
    if pinecone_section.get("configured"):
        if "error" in pinecone_section:
            print(f"- Pinecone: error={pinecone_section['error']}")
        else:
            print(
                "- Pinecone: "
                f"index={pinecone_section['index']} "
                f"namespace={pinecone_section['namespace']} "
                f"namespace_vector_count={pinecone_section.get('namespace_vector_count')}"
            )
    else:
        print(f"- Pinecone: not configured ({pinecone_section.get('error')})")
    smoke = report["retrieval_smoke"]
    if smoke.get("enabled"):
        if "error" in smoke:
            print(f"- Retrieval smoke: error={smoke['error']}")
        else:
            print(
                "- Retrieval smoke: "
                f"question={smoke['question']!r} "
                f"top_k={smoke['top_k']} "
                f"match_count={smoke['match_count']}"
            )
    else:
        print("- Retrieval smoke: skipped (use --smoke-query '...')")

    print("- Gate summary:")
    for gate_name, status in report["gates"].items():
        print(f"  - {gate_name}={status}")


if __name__ == "__main__":
    main()
