from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from app.config import settings

if TYPE_CHECKING:
    from openai import OpenAI
    from pinecone import Pinecone

FORTRAN_EXTENSIONS = {".f", ".for", ".f90", ".f95", ".f03", ".f08", ".inc"}
START_PATTERN = re.compile(
    r"^\s*(program|subroutine|function|module(?!\s+procedure)|block\s+data|interface)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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
    parser.add_argument("--dry-run", action="store_true", help="Build chunks without embedding/upsert")
    return parser.parse_args()


def discover_fortran_files(repo_root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if "/.git/" in str(path).replace("\\", "/"):
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

    return chunks


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

    vectors_payload: list[tuple[str, list[float], dict]] = []

    for chunk_batch in batched(chunks, args.embed_batch_size):
        texts = [chunk.chunk_text for chunk in chunk_batch]
        embeddings = openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        ).data

        for chunk, embedded in zip(chunk_batch, embeddings, strict=True):
            metadata = {
                "file_path": chunk.file_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "section_name": chunk.section_name,
                "language": "fortran",
                "chunk_text": chunk.chunk_text,
            }
            vectors_payload.append((chunk_id(args.namespace, chunk), embedded.embedding, metadata))

    total_upserted = 0
    for vector_batch in batched(vectors_payload, args.upsert_batch_size):
        index.upsert(vectors=vector_batch, namespace=args.namespace)
        total_upserted += len(vector_batch)

    print(f"Upserted vectors: {total_upserted}")
    print(f"Namespace: {args.namespace}")


def main() -> None:
    args = parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
