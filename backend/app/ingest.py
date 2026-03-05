from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from app.config import settings

if TYPE_CHECKING:
    from openai import OpenAI
    from pinecone import Pinecone

FORTRAN_EXTENSIONS = {".f", ".for", ".f90", ".f95", ".f03", ".f08", ".inc"}
EXCLUDED_DISCOVERY_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "vendor",
    "node_modules",
}
START_PATTERN = re.compile(
    r"^\s*(program|subroutine|function|module(?!\s+procedure)|block\s+data|interface)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
EMBEDDING_METADATA_SCHEMA_VERSION = "v1"


@dataclass
class FileChunk:
    file_path: str
    line_start: int
    line_end: int
    section_name: str
    chunk_text: str


@dataclass
class Section:
    start_line: int
    end_line: int
    name: str
    lines: list[str]


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Fortran repository into Pinecone")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to repository root (default: project root)",
    )
    parser.add_argument(
        "--namespace",
        default=settings.pinecone_namespace,
        help="Pinecone namespace",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(FORTRAN_EXTENSIONS)),
        help="Comma-separated list of file extensions to include",
    )
    parser.add_argument("--target-min-tokens", type=int, default=200)
    parser.add_argument("--target-max-tokens", type=int, default=500)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete all vectors in the target namespace before upserting",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build chunks without embedding/upsert")
    return parser.parse_args()


def discover_fortran_files(repo_root: Path, extensions: set[str]) -> list[Path]:
    root = repo_root.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
        except ValueError:
            continue
        if any(part in EXCLUDED_DISCOVERY_DIRS for part in rel_parts):
            continue
        if path.suffix.lower() in extensions:
            files.append(path)
    return sorted(files)


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    if not lines:
        return []

    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        if START_PATTERN.match(line):
            name = line.strip()[:120]
            starts.append((i, name))

    if not starts:
        return [Section(start_line=1, end_line=len(lines), name="file", lines=lines)]

    sections: list[Section] = []
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
        sections.append(Section(start_line=start, end_line=end, name=name, lines=lines[start - 1 : end]))
    return sections


def split_large_text(
    lines: list[str],
    start_line: int,
    section_name: str,
    min_tokens: int,
    max_tokens: int,
) -> list[FileChunk]:
    chunks: list[FileChunk] = []
    current: list[str] = []
    current_start = start_line

    def flush(current_end: int) -> None:
        nonlocal current, current_start
        text = "\n".join(current).strip()
        if not text:
            current = []
            current_start = current_end + 1
            return
        chunks.append(
            FileChunk(
                file_path="",
                line_start=current_start,
                line_end=current_end,
                section_name=section_name,
                chunk_text=text,
            )
        )
        current = []
        current_start = current_end + 1

    for offset, line in enumerate(lines):
        abs_line = start_line + offset
        candidate = current + [line]
        candidate_tokens = token_count("\n".join(candidate))

        if candidate_tokens > max_tokens and current:
            flush(abs_line - 1)
            current = [line]
            current_start = abs_line
            continue

        current.append(line)

        if not line.strip() and token_count("\n".join(current)) >= min_tokens:
            flush(abs_line)

    if current:
        flush(start_line + len(lines) - 1)

    if len(chunks) <= 1:
        return chunks

    # Merge very small chunks into neighboring chunks when it keeps chunk size bounded.
    merged: list[FileChunk] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        chunk_tokens = token_count(chunk.chunk_text)
        if chunk_tokens < min_tokens:
            if merged:
                prev = merged[-1]
                combined_prev = f"{prev.chunk_text}\n{chunk.chunk_text}".strip()
                if token_count(combined_prev) <= max_tokens:
                    merged[-1] = FileChunk(
                        file_path=prev.file_path,
                        line_start=prev.line_start,
                        line_end=chunk.line_end,
                        section_name=prev.section_name,
                        chunk_text=combined_prev,
                    )
                    i += 1
                    continue
            if i + 1 < len(chunks):
                nxt = chunks[i + 1]
                combined_next = f"{chunk.chunk_text}\n{nxt.chunk_text}".strip()
                if token_count(combined_next) <= max_tokens:
                    merged.append(
                        FileChunk(
                            file_path=nxt.file_path,
                            line_start=chunk.line_start,
                            line_end=nxt.line_end,
                            section_name=chunk.section_name,
                            chunk_text=combined_next,
                        )
                    )
                    i += 2
                    continue
        merged.append(chunk)
        i += 1

    return merged


def chunk_fortran_file(path: Path, repo_root: Path, min_tokens: int, max_tokens: int) -> list[FileChunk]:
    text = read_text_with_fallback(path)
    sections = split_into_sections(text)
    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")

    all_chunks: list[FileChunk] = []
    for section in sections:
        section_chunks = split_large_text(
            lines=section.lines,
            start_line=section.start_line,
            section_name=section.name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        for chunk in section_chunks:
            chunk.file_path = rel_path
        all_chunks.extend(section_chunks)

    return all_chunks


def chunk_id(namespace: str, chunk: FileChunk) -> str:
    raw = f"{namespace}|{chunk.file_path}|{chunk.line_start}|{chunk.line_end}|{chunk.chunk_text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalize_index_description(description: object) -> dict[str, Any]:
    if isinstance(description, dict):
        return description
    if hasattr(description, "to_dict"):
        try:
            payload = description.to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
    return {}


def pinecone_index_dimension(pinecone_client: Any) -> int | None:
    try:
        if not hasattr(pinecone_client, "describe_index"):
            return None
        description = call_with_retries(
            "pinecone describe index",
            lambda: pinecone_client.describe_index(settings.pinecone_index_name),
        )
    except Exception:
        return None

    if hasattr(description, "dimension"):
        try:
            value = int(getattr(description, "dimension"))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    payload = normalize_index_description(description)
    for key in ("dimension", "vector_dimension"):
        try:
            value = int(payload.get(key)) if payload.get(key) is not None else None
        except (TypeError, ValueError):
            value = None
        if value and value > 0:
            return value
    return None


def ensure_index_dimension_match(index_dim: int | None, embedding_dim: int) -> None:
    if not settings.enforce_embedding_dimension:
        return
    if index_dim is None:
        raise RuntimeError(
            "Could not determine Pinecone index dimension; set ENFORCE_EMBEDDING_DIMENSION=false to bypass."
        )
    if embedding_dim != index_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: embedding={embedding_dim}, index={index_dim}. "
            "Recreate index or change embedding model."
        )


def call_with_retries(action: str, fn):
    attempts = max(settings.external_call_retries, 1)
    base_backoff = max(settings.external_call_backoff_seconds, 0.0)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_for = base_backoff * (2 ** (attempt - 1))
            print(
                f"[retry {attempt}/{attempts}] {action} failed: {exc}. "
                f"retrying in {sleep_for:.2f}s"
            )
            if sleep_for > 0:
                time.sleep(sleep_for)


def ensure_clients() -> tuple[Any, Any]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")
    from openai import OpenAI
    from pinecone import Pinecone

    return OpenAI(api_key=settings.openai_api_key), Pinecone(api_key=settings.pinecone_api_key)


def ingest(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}

    files = discover_fortran_files(repo_root=repo_root, extensions=extensions)
    if not files:
        raise RuntimeError(f"No files found under {repo_root} for extensions: {sorted(extensions)}")

    chunks: list[FileChunk] = []
    for path in files:
        chunks.extend(
            chunk_fortran_file(
                path=path,
                repo_root=repo_root,
                min_tokens=args.target_min_tokens,
                max_tokens=args.target_max_tokens,
            )
        )

    print(f"Discovered files: {len(files)}")
    print(f"Generated chunks: {len(chunks)}")

    if args.dry_run:
        avg_tokens = int(sum(token_count(chunk.chunk_text) for chunk in chunks) / max(len(chunks), 1))
        print(f"Dry run complete. Avg chunk tokens: {avg_tokens}")
        return

    openai_client, pinecone_client = ensure_clients()
    index = pinecone_client.Index(settings.pinecone_index_name)
    index_dim = pinecone_index_dimension(pinecone_client)
    if args.delete_existing:
        call_with_retries(
            "pinecone namespace delete_all",
            lambda: index.delete(delete_all=True, namespace=args.namespace),
        )
        print(f"Deleted existing vectors in namespace: {args.namespace}")

    pending_vectors: list[tuple[str, list[float], dict]] = []
    total_upserted = 0
    checked_index_dimension = False
    for chunk_batch in batched(chunks, args.embed_batch_size):
        texts = [chunk.chunk_text for chunk in chunk_batch]
        embeddings = call_with_retries(
            "openai embeddings",
            lambda: openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            ).data,
        )

        for chunk, embedded in zip(chunk_batch, embeddings, strict=True):
            embedding_dim = len(embedded.embedding)
            if not checked_index_dimension:
                ensure_index_dimension_match(index_dim=index_dim, embedding_dim=embedding_dim)
                checked_index_dimension = True
            repo_name = repo_root.name
            metadata = {
                "file_path": chunk.file_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                # Alias fields used by some hybrid retrieval stacks.
                "start_line": chunk.line_start,
                "end_line": chunk.line_end,
                "repo": repo_name,
                "section_name": chunk.section_name,
                "language": "fortran",
                "source_type": "repo",
                "chunk_text": chunk.chunk_text,
                "embedding_model": settings.openai_embedding_model,
                "embedding_provider": "openai",
                "embedding_dimension": embedding_dim,
                "embedding_schema_version": EMBEDDING_METADATA_SCHEMA_VERSION,
            }
            pending_vectors.append((chunk_id(args.namespace, chunk), embedded.embedding, metadata))

            if len(pending_vectors) >= args.upsert_batch_size:
                vector_batch = pending_vectors[: args.upsert_batch_size]
                pending_vectors = pending_vectors[args.upsert_batch_size:]
                call_with_retries(
                    "pinecone upsert",
                    lambda: index.upsert(vectors=vector_batch, namespace=args.namespace),
                )
                total_upserted += len(vector_batch)

    if pending_vectors:
        call_with_retries(
            "pinecone upsert",
            lambda: index.upsert(vectors=pending_vectors, namespace=args.namespace),
        )
        total_upserted += len(pending_vectors)

    print(f"Upserted vectors: {total_upserted}")
    print(f"Namespace: {args.namespace}")


def main() -> None:
    args = parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
