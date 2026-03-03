from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingest import FORTRAN_EXTENSIONS, chunk_fortran_file, discover_fortran_files, token_count


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
                ns_count = ns_entry.get("vector_count")

        return {
            "configured": True,
            "index": index_name,
            "namespace": namespace,
            "namespace_vector_count": ns_count,
            "raw": raw,
        }
    except Exception as exc:
        return {"configured": True, "error": str(exc)}


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


if __name__ == "__main__":
    main()

