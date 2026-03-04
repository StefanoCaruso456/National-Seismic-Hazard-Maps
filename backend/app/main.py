import logging
import math
import re
import time
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.ingest import (
    FORTRAN_EXTENSIONS,
    chunk_fortran_file,
    discover_fortran_files,
    token_count,
)

logger = logging.getLogger("legacylens.api")
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
SAFE_UPLOAD_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
SAFE_PROJECT_ID_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

UPLOAD_MAX_FILES = 8
UPLOAD_MAX_FILE_BYTES = 1_500_000
UPLOAD_MAX_TOTAL_BYTES = 6_000_000
UPLOAD_MIN_TOKENS = 300
UPLOAD_MAX_TOKENS = 500
UPLOAD_OVERLAP_TOKENS = 50
UPLOAD_EMBED_BATCH_SIZE = 64
UPLOAD_PRIORITY_BONUS = 0.04
UPLOAD_SOURCE_BONUS = 0.02
DEFAULT_REPO_URL = "https://github.com/StefanoCaruso456/National-Seismic-Hazard-Maps"
ATTACHMENT_NAMESPACE_PREFIX = "attachments"
ATTACHMENT_NAMESPACE_VERSION = "v1"
ATTACHMENT_VECTOR_PREFIX = "attchunk"
ATTACHMENT_MARKER_PREFIX = "attfile"
UPLOAD_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "attachments_manifest.json"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    scope: str = Field(default="both")
    project_id: str = Field(default="nshmp-main", min_length=1, max_length=80)
    debug: bool = False


class Citation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    score: float
    source_type: str = "repo"
    file_sha: str | None = None
    snippet: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_strength: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    matches: list[Citation] = Field(default_factory=list)
    evidence_strength: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class PineconeDebugResponse(BaseModel):
    configured_index: str
    configured_namespace: str
    available_indexes: list[str] = Field(default_factory=list)
    index_description: dict[str, Any] = Field(default_factory=dict)
    index_stats: dict[str, Any] = Field(default_factory=dict)
    namespace_vector_count: int | None = None
    total_vector_count: int | None = None


class RepoScanResponse(BaseModel):
    repo_root: str
    fortran_extensions: list[str]
    file_count: int
    sample_files: list[str] = Field(default_factory=list)


class RetrievalInfoResponse(BaseModel):
    lexical_weight: float
    min_hybrid_score: float
    candidate_multiplier: int
    max_candidates: int
    rag_max_context_chunks: int
    upload_max_files: int
    upload_max_file_bytes: int
    upload_max_total_bytes: int
    upload_chunk_min_tokens: int
    upload_chunk_max_tokens: int
    upload_chunk_overlap_tokens: int
    repo_url: str
    default_project_id: str
    query_top_k_default: int = 5
    query_top_k_min: int = 1
    query_top_k_max: int = 20


class UploadFileRecord(BaseModel):
    project_id: str
    namespace: str
    file_sha: str
    file_name: str
    file_size: int
    chunk_count: int
    uploaded_at: str
    pinned: bool = False
    persisted: bool = True


class UploadIngestStatus(BaseModel):
    file_sha: str
    file_name: str
    file_size: int
    chunk_count: int
    status: str
    namespace: str
    reason: str | None = None


class UploadIngestResponse(BaseModel):
    project_id: str
    namespace: str
    persisted: bool
    files: list[UploadIngestStatus] = Field(default_factory=list)


class UploadListResponse(BaseModel):
    project_id: str
    namespace: str
    files: list[UploadFileRecord] = Field(default_factory=list)


class UploadDeleteResponse(BaseModel):
    project_id: str
    namespace: str
    file_sha: str
    deleted: bool


class UploadPinRequest(BaseModel):
    project_id: str | None = None
    pinned: bool = True


openai_client: OpenAI | None = None
pinecone_client: Pinecone | None = None


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/retrieval-info", response_model=RetrievalInfoResponse)
def retrieval_info() -> RetrievalInfoResponse:
    default_project = default_project_id()
    return RetrievalInfoResponse(
        lexical_weight=min(max(settings.retrieval_lexical_weight, 0.0), 1.0),
        min_hybrid_score=min(max(settings.retrieval_min_hybrid_score, 0.0), 1.0),
        candidate_multiplier=max(settings.retrieval_candidate_multiplier, 1),
        max_candidates=max(settings.retrieval_max_candidates, 1),
        rag_max_context_chunks=max(settings.rag_max_context_chunks, 1),
        upload_max_files=UPLOAD_MAX_FILES,
        upload_max_file_bytes=UPLOAD_MAX_FILE_BYTES,
        upload_max_total_bytes=UPLOAD_MAX_TOTAL_BYTES,
        upload_chunk_min_tokens=UPLOAD_MIN_TOKENS,
        upload_chunk_max_tokens=UPLOAD_MAX_TOKENS,
        upload_chunk_overlap_tokens=UPLOAD_OVERLAP_TOKENS,
        repo_url=DEFAULT_REPO_URL,
        default_project_id=default_project,
    )


def get_openai_client() -> OpenAI:
    global openai_client
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    if openai_client is None:
        openai_client = OpenAI(api_key=settings.openai_api_key)
    return openai_client


def get_pinecone_index():
    global pinecone_client
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not configured")
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return pinecone_client.Index(settings.pinecone_index_name)


def require_debug_mode() -> None:
    if not settings.app_debug:
        # Avoid exposing internal diagnostics on a public deployment by default.
        raise HTTPException(status_code=404, detail="Not found")


def safe_list_indexes(pc: Pinecone) -> list[str]:
    try:
        indexes = pc.list_indexes()
    except Exception:
        return []

    # Pinecone clients have returned different shapes across versions: handle common cases.
    if isinstance(indexes, list):
        return [getattr(i, "name", str(i)) for i in indexes]
    if hasattr(indexes, "names"):
        try:
            return list(indexes.names())
        except Exception:
            return []
    if hasattr(indexes, "indexes"):
        try:
            return [i.get("name") for i in indexes.indexes if isinstance(i, dict) and i.get("name")]
        except Exception:
            return []
    return []


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def default_project_id() -> str:
    raw = (settings.pinecone_namespace or "nshmp-main").split(":")[0]
    cleaned = SAFE_PROJECT_ID_PATTERN.sub("-", raw).strip("._-")
    return cleaned or "nshmp-main"


def normalize_project_id(value: str | None) -> str:
    candidate = (value or default_project_id()).strip()
    cleaned = SAFE_PROJECT_ID_PATTERN.sub("-", candidate).strip("._-")
    if not cleaned:
        return default_project_id()
    return cleaned[:80]


def normalize_scope(value: str | None) -> str:
    scope = (value or "both").strip().lower()
    if scope in {"repo", "uploads", "both"}:
        return scope
    raise HTTPException(status_code=422, detail="scope must be one of: repo, uploads, both")


def attachments_namespace(project_id: str) -> str:
    return f"{ATTACHMENT_NAMESPACE_PREFIX}:{normalize_project_id(project_id)}:{ATTACHMENT_NAMESPACE_VERSION}"


def attachment_marker_id(file_sha: str) -> str:
    return f"{ATTACHMENT_MARKER_PREFIX}:{file_sha}"


def attachment_chunk_id(file_sha: str, index: int) -> str:
    return f"{ATTACHMENT_VECTOR_PREFIX}:{file_sha}:{index:05d}"


def load_upload_manifest() -> dict[str, dict[str, dict[str, Any]]]:
    if not UPLOAD_MANIFEST_PATH.exists():
        return {}
    try:
        payload = json.loads(UPLOAD_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for project, files in payload.items():
        if not isinstance(files, dict):
            continue
        normalized[project] = {}
        for file_sha, record in files.items():
            if isinstance(record, dict):
                normalized[project][file_sha] = record
    return normalized


def save_upload_manifest(manifest: dict[str, dict[str, dict[str, Any]]]) -> None:
    UPLOAD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def get_upload_record(project_id: str, file_sha: str) -> dict[str, Any] | None:
    manifest = load_upload_manifest()
    return manifest.get(project_id, {}).get(file_sha)


def upsert_upload_record(project_id: str, file_sha: str, record: dict[str, Any]) -> None:
    manifest = load_upload_manifest()
    bucket = manifest.setdefault(project_id, {})
    bucket[file_sha] = record
    save_upload_manifest(manifest)


def remove_upload_record(project_id: str, file_sha: str) -> None:
    manifest = load_upload_manifest()
    project_files = manifest.get(project_id, {})
    if file_sha in project_files:
        del project_files[file_sha]
        if not project_files:
            manifest.pop(project_id, None)
        save_upload_manifest(manifest)


def list_upload_records(project_id: str) -> list[UploadFileRecord]:
    manifest = load_upload_manifest()
    records = manifest.get(project_id, {})
    items: list[UploadFileRecord] = []
    namespace = attachments_namespace(project_id)
    for file_sha, payload in records.items():
        items.append(
            UploadFileRecord(
                project_id=project_id,
                namespace=namespace,
                file_sha=file_sha,
                file_name=str(payload.get("file_name", "unknown")),
                file_size=safe_int(payload.get("file_size")) or 0,
                chunk_count=safe_int(payload.get("chunk_count")) or 0,
                uploaded_at=str(payload.get("uploaded_at", "")) or "",
                pinned=bool(payload.get("pinned", False)),
                persisted=bool(payload.get("persisted", True)),
            )
        )
    items.sort(key=lambda item: item.uploaded_at, reverse=True)
    return items


def delete_attachment_vectors(project_id: str, file_sha: str, chunk_count: int) -> None:
    namespace = attachments_namespace(project_id)
    index = get_pinecone_index()
    # Primary delete path by metadata filter for privacy-safe hard delete by file_sha.
    try:
        call_with_retries(
            "pinecone attachment delete by filter",
            lambda: index.delete(filter={"file_sha": {"$eq": file_sha}}, namespace=namespace),
        )
        return
    except Exception as exc:
        logger.warning("Delete by filter failed for file_sha=%s: %s; falling back to id deletion", file_sha, exc)

    ids = [attachment_marker_id(file_sha)] + [attachment_chunk_id(file_sha, idx) for idx in range(max(chunk_count, 0))]
    for batch in batched(ids, 100):
        call_with_retries(
            "pinecone attachment delete",
            lambda: index.delete(ids=batch, namespace=namespace),
        )

def extract_vector_counts(stats: dict[str, Any], namespace: str) -> tuple[int | None, int | None]:
    if not isinstance(stats, dict):
        return None, None

    namespace_count: int | None = None
    namespaces = stats.get("namespaces")
    if isinstance(namespaces, dict):
        ns_entry = namespaces.get(namespace)
        if isinstance(ns_entry, dict):
            namespace_count = safe_int(ns_entry.get("vector_count"))

    total_count = safe_int(stats.get("total_vector_count"))
    return namespace_count, total_count


@app.get("/api/debug/pinecone", response_model=PineconeDebugResponse)
def debug_pinecone() -> PineconeDebugResponse:
    require_debug_mode()
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not configured")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    available = safe_list_indexes(pc)

    description: dict[str, Any] = {}
    try:
        if hasattr(pc, "describe_index"):
            desc_obj = pc.describe_index(settings.pinecone_index_name)
            # Pydantic can serialize dicts; fallback to string for non-serializable objects.
            description = desc_obj if isinstance(desc_obj, dict) else getattr(desc_obj, "to_dict", lambda: {})()
            if not description:
                description = {"raw": str(desc_obj)}
    except Exception as exc:
        description = {"error": str(exc)}

    stats: dict[str, Any] = {}
    try:
        index = pc.Index(settings.pinecone_index_name)
        if hasattr(index, "describe_index_stats"):
            stats_obj = index.describe_index_stats()
            stats = stats_obj if isinstance(stats_obj, dict) else getattr(stats_obj, "to_dict", lambda: {})()
            if not stats:
                stats = {"raw": str(stats_obj)}
    except Exception as exc:
        stats = {"error": str(exc)}

    namespace_vector_count, total_vector_count = extract_vector_counts(stats, settings.pinecone_namespace)

    return PineconeDebugResponse(
        configured_index=settings.pinecone_index_name,
        configured_namespace=settings.pinecone_namespace,
        available_indexes=available,
        index_description=description,
        index_stats=stats,
        namespace_vector_count=namespace_vector_count,
        total_vector_count=total_vector_count,
    )


@app.get("/api/debug/repo-scan", response_model=RepoScanResponse)
def debug_repo_scan() -> RepoScanResponse:
    require_debug_mode()
    repo_root = Path(__file__).resolve().parents[2]
    files = discover_fortran_files(repo_root=repo_root, extensions=set(FORTRAN_EXTENSIONS))
    sample = [str(p.relative_to(repo_root)).replace("\\", "/") for p in files[:12]]
    return RepoScanResponse(
        repo_root=str(repo_root),
        fortran_extensions=sorted(FORTRAN_EXTENSIONS),
        file_count=len(files),
        sample_files=sample,
    )


@app.get("/api/debug/sample-chunks")
def debug_sample_chunks(file_path: str) -> dict:
    require_debug_mode()
    repo_root = Path(__file__).resolve().parents[2]
    target = (repo_root / file_path).resolve()
    if not target.is_file() or repo_root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid file_path")

    chunks = chunk_fortran_file(
        path=target,
        repo_root=repo_root,
        min_tokens=200,
        max_tokens=500,
    )
    preview = [
        {
            "file_path": c.file_path,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "section_name": c.section_name,
            "token_count": token_count(c.chunk_text),
            "text_preview": c.chunk_text[:260],
        }
        for c in chunks[:8]
    ]
    return {"chunk_count": len(chunks), "preview": preview}


def call_with_retries(action: str, fn):
    attempts = max(settings.external_call_retries, 1)
    base_backoff = max(settings.external_call_backoff_seconds, 0.0)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except HTTPException:
            raise
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_for = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %s/%s): %s; retrying in %.2fs",
                action,
                attempt,
                attempts,
                exc,
                sleep_for,
            )
            if sleep_for > 0:
                time.sleep(sleep_for)


def validate_query_request(
    question: str,
    top_k: int,
    debug: bool = False,
    scope: str = "both",
    project_id: str | None = None,
) -> QueryRequest:
    try:
        return QueryRequest(
            question=question,
            top_k=top_k,
            debug=debug,
            scope=normalize_scope(scope),
            project_id=normalize_project_id(project_id),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def batched(items: list[Any], size: int):
    if size <= 0:
        size = 1
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_openai_client()
    response = call_with_retries(
        "openai embeddings",
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        ),
    )
    return [list(item.embedding) for item in response.data]


def safe_upload_name(filename: str, index: int) -> str:
    base_name = Path(filename or f"upload_{index}.txt").name
    cleaned = SAFE_UPLOAD_NAME_PATTERN.sub("_", base_name).strip("._")
    if not cleaned:
        cleaned = f"upload_{index}.txt"
    return f"{index:02d}_{cleaned[:120]}"


def decode_upload_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def upload_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalize_fetch_vectors(fetch_response: object) -> dict[str, Any]:
    if isinstance(fetch_response, dict):
        vectors = fetch_response.get("vectors")
        return vectors if isinstance(vectors, dict) else {}
    if hasattr(fetch_response, "vectors"):
        vectors = getattr(fetch_response, "vectors")
        return vectors if isinstance(vectors, dict) else {}
    return {}


def attachment_marker_exists(index, namespace: str, file_sha: str) -> bool:
    marker_id = attachment_marker_id(file_sha)
    fetch_response = call_with_retries(
        "pinecone fetch marker",
        lambda: index.fetch(ids=[marker_id], namespace=namespace),
    )
    vectors = normalize_fetch_vectors(fetch_response)
    return marker_id in vectors


def split_lines_with_token_overlap(
    lines: list[str],
    start_line: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int, str]]:
    if not lines:
        return []

    chunks: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        token_total = 0
        end_index = index
        while end_index < len(lines):
            line_tokens = token_count(lines[end_index])
            token_total += line_tokens
            end_index += 1
            if token_total >= max_tokens:
                break

        while end_index < len(lines) and token_total < min_tokens:
            token_total += token_count(lines[end_index])
            end_index += 1

        if end_index <= index:
            end_index = index + 1

        chunk_text = "\n".join(lines[index:end_index]).strip()
        if chunk_text:
            chunks.append((start_line + index, start_line + end_index - 1, chunk_text))

        if end_index >= len(lines):
            break

        overlap_total = 0
        next_index = end_index
        while next_index > index:
            candidate = token_count(lines[next_index - 1])
            if overlap_total + candidate > overlap_tokens and overlap_total >= overlap_tokens:
                break
            overlap_total += candidate
            next_index -= 1
            if overlap_total >= overlap_tokens:
                break

        if next_index <= index:
            next_index = min(index + 1, len(lines) - 1)
        index = next_index

    return chunks


def extract_pdf_pages(raw: bytes) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - handled at runtime if dependency missing
        raise HTTPException(status_code=500, detail=f"PDF parsing unavailable: {exc}") from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((page_index, text))
    return pages


def build_attachment_chunks(
    uploaded_files: list[dict[str, Any]],
    source_type: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata_chunks: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}

    for upload in uploaded_files:
        file_name = str(upload.get("safe_name", "upload.txt"))
        file_sha = str(upload.get("file_sha", ""))
        raw = upload.get("raw_bytes")
        if not isinstance(raw, (bytes, bytearray)):
            continue
        suffix = Path(file_name).suffix.lower()
        base_path = f"uploaded/{file_name}"

        segments: list[tuple[str, int, list[str], str]] = []
        if suffix == ".pdf":
            pages = extract_pdf_pages(bytes(raw))
            for page_num, page_text in pages:
                page_lines = page_text.splitlines() or [page_text]
                segments.append((f"{base_path}#page-{page_num}", 1, page_lines, "pdf"))
        else:
            text = decode_upload_bytes(bytes(raw))
            if text.strip():
                lines = text.splitlines() or [text]
                language = "fortran" if suffix in FORTRAN_EXTENSIONS else "text"
                segments.append((base_path, 1, lines, language))

        if not segments:
            per_file_counts[file_sha] = 0
            continue

        chunk_counter = 0
        for segment_path, segment_start, lines, language in segments:
            chunk_triplets = split_lines_with_token_overlap(
                lines=lines,
                start_line=segment_start,
                min_tokens=UPLOAD_MIN_TOKENS,
                max_tokens=UPLOAD_MAX_TOKENS,
                overlap_tokens=UPLOAD_OVERLAP_TOKENS,
            )
            for line_start, line_end, chunk_text in chunk_triplets:
                metadata_chunks.append(
                    {
                        "file_path": segment_path,
                        "line_start": line_start,
                        "line_end": line_end,
                        "section_name": "upload",
                        "language": language,
                        "chunk_text": chunk_text,
                        "source_type": source_type,
                        "file_sha": file_sha,
                        "file_name": file_name,
                    }
                )
                chunk_counter += 1
        per_file_counts[file_sha] = chunk_counter

    return metadata_chunks, per_file_counts


async def extract_uploaded_files(files: list[UploadFile]) -> list[dict[str, Any]]:
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files attached. Max allowed is {UPLOAD_MAX_FILES}.",
        )

    total_bytes = 0
    uploaded: list[dict[str, Any]] = []
    for index, upload in enumerate(files, start=1):
        try:
            raw = await upload.read(UPLOAD_MAX_FILE_BYTES + 1)
        finally:
            await upload.close()

        if not raw:
            continue
        if len(raw) > UPLOAD_MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{upload.filename or f'upload_{index}'}' is too large. "
                    f"Max per file is {UPLOAD_MAX_FILE_BYTES} bytes."
                ),
            )

        total_bytes += len(raw)
        if total_bytes > UPLOAD_MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Attached files exceed total limit of {UPLOAD_MAX_TOTAL_BYTES} bytes.",
            )

        safe_name = safe_upload_name(upload.filename or "", index)
        uploaded.append(
            {
                "safe_name": safe_name,
                "original_name": upload.filename or safe_name,
                "raw_bytes": bytes(raw),
                "file_size": len(raw),
                "file_sha": upload_sha256(bytes(raw)),
            }
        )

    return uploaded


def upsert_attachment_chunks(
    project_id: str,
    uploaded_files: list[dict[str, Any]],
) -> list[UploadIngestStatus]:
    project = normalize_project_id(project_id)
    namespace = attachments_namespace(project)
    index = get_pinecone_index()

    statuses: list[UploadIngestStatus] = []
    for upload in uploaded_files:
        file_sha = str(upload.get("file_sha", ""))
        file_name = str(upload.get("safe_name", "upload.txt"))
        file_size = safe_int(upload.get("file_size")) or 0
        if not file_sha:
            continue

        if attachment_marker_exists(index=index, namespace=namespace, file_sha=file_sha):
            record = get_upload_record(project, file_sha)
            chunk_count = safe_int((record or {}).get("chunk_count")) or 0
            statuses.append(
                UploadIngestStatus(
                    file_sha=file_sha,
                    file_name=file_name,
                    file_size=file_size,
                    chunk_count=chunk_count,
                    status="skipped",
                    namespace=namespace,
                    reason="duplicate_sha",
                )
            )
            continue

        chunks, per_file = build_attachment_chunks([upload], source_type="upload")
        chunk_count = per_file.get(file_sha, 0)
        if not chunks or chunk_count <= 0:
            statuses.append(
                UploadIngestStatus(
                    file_sha=file_sha,
                    file_name=file_name,
                    file_size=file_size,
                    chunk_count=0,
                    status="skipped",
                    namespace=namespace,
                    reason="no_extractable_text",
                )
            )
            continue

        pending_vectors: list[tuple[str, list[float], dict[str, Any]]] = []
        next_chunk_index = 0
        for metadata_batch in batched(chunks, UPLOAD_EMBED_BATCH_SIZE):
            embeddings = embed_texts([metadata_chunk_text(metadata) for metadata in metadata_batch])
            for metadata, embedding in zip(metadata_batch, embeddings, strict=True):
                vector_id = attachment_chunk_id(file_sha, next_chunk_index)
                next_chunk_index += 1
                enriched = metadata.copy()
                enriched["_record_type"] = "attachment_chunk"
                enriched["project_id"] = project
                pending_vectors.append((vector_id, embedding, enriched))

            if len(pending_vectors) >= 100:
                vector_batch = pending_vectors[:100]
                pending_vectors = pending_vectors[100:]
                call_with_retries(
                    "pinecone attachment upsert",
                    lambda: index.upsert(vectors=vector_batch, namespace=namespace),
                )

        if pending_vectors:
            call_with_retries(
                "pinecone attachment upsert",
                lambda: index.upsert(vectors=pending_vectors, namespace=namespace),
            )

        marker_embedding = embed_texts([f"attachment marker {file_sha} {file_name}"])[0]
        marker_metadata = {
            "_record_type": "attachment_file",
            "source_type": "upload",
            "file_sha": file_sha,
            "file_name": file_name,
            "file_size": file_size,
            "chunk_count": next_chunk_index,
            "project_id": project,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "chunk_text": f"attachment file marker {file_name}",
        }
        call_with_retries(
            "pinecone marker upsert",
            lambda: index.upsert(
                vectors=[(attachment_marker_id(file_sha), marker_embedding, marker_metadata)],
                namespace=namespace,
            ),
        )

        upsert_upload_record(
            project,
            file_sha,
            {
                "file_name": file_name,
                "file_size": file_size,
                "chunk_count": next_chunk_index,
                "uploaded_at": marker_metadata["uploaded_at"],
                "pinned": False,
                "persisted": True,
            },
        )
        statuses.append(
            UploadIngestStatus(
                file_sha=file_sha,
                file_name=file_name,
                file_size=file_size,
                chunk_count=next_chunk_index,
                status="persisted",
                namespace=namespace,
            )
        )

    return statuses


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0
    dot = sum(vec_a[i] * vec_b[i] for i in range(length))
    norm_a = math.sqrt(sum(vec_a[i] * vec_a[i] for i in range(length)))
    norm_b = math.sqrt(sum(vec_b[i] * vec_b[i] for i in range(length)))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=max(settings.embedding_cache_size, 1))
def _cached_question_embedding(question: str) -> tuple[float, ...]:
    client = get_openai_client()
    response = call_with_retries(
        "openai embeddings",
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=question,
        ),
    )
    return tuple(response.data[0].embedding)


def embed_question(question: str) -> list[float]:
    normalized = question.strip()
    if not normalized:
        return []
    return list(_cached_question_embedding(normalized))


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


def normalize_semantic_score(score: float) -> float:
    # Cosine similarity can be in [-1, 1], map to [0, 1] for hybrid scoring.
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def tokenize_question(question: str) -> tuple[set[str], set[str]]:
    words = {token.lower() for token in WORD_PATTERN.findall(question) if len(token) >= 2}
    identifiers = {token.lower() for token in IDENTIFIER_PATTERN.findall(question)}
    return words, identifiers


def dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(item.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def rewrite_and_decompose_query(question: str) -> tuple[str, list[str]]:
    base = " ".join(question.split())
    lowered = base.lower()
    subqueries: list[str] = []

    locate_match = re.search(
        r"\bwhere\s+(?:is|are)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:implemented|defined|located)\b",
        base,
        re.IGNORECASE,
    )
    if locate_match:
        symbol = locate_match.group(1)
        subqueries.extend(
            [
                f"subroutine {symbol}",
                f"function {symbol}",
                f"module {symbol}",
                f"{symbol} implementation file",
            ]
        )

    broad_terms = ("workflow", "end-to-end", "pipeline", "reproduce", "runbook")
    if any(term in lowered for term in broad_terms):
        subqueries.extend(
            [
                f"{base} docs",
                f"{base} scripts",
                f"{base} conf",
                f"{base} src",
            ]
        )

    if "config" in lowered or "setting" in lowered or "parameter" in lowered:
        subqueries.extend([f"{base} conf", f"{base} Makefile", f"{base} scripts"])

    # Include identifier-focused probes for better lexical+semantic union in legacy code.
    _, identifiers = tokenize_question(base)
    for ident in sorted(identifiers)[:4]:
        subqueries.extend([f"{ident} subroutine", f"{ident} common block"])

    return base, dedupe_preserve_order(subqueries)


def parse_form_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def lexical_score(question_terms: set[str], identifiers: set[str], metadata: dict) -> float:
    if not question_terms and not identifiers:
        return 0.0
    file_path = str(metadata.get("file_path", "")).lower()
    section_name = str(metadata.get("section_name", "")).lower()
    source = f"{file_path}\n{section_name}\n{metadata_chunk_text(metadata)[:2500]}".lower()
    source_terms = {token for token in WORD_PATTERN.findall(source) if len(token) >= 2}

    overlap = len(question_terms & source_terms)
    term_coverage = (overlap / len(question_terms)) if question_terms else 0.0

    id_bonus = 0.0
    for ident in identifiers:
        if ident in file_path:
            id_bonus = max(id_bonus, 0.20)
        elif ident in section_name:
            id_bonus = max(id_bonus, 0.15)
        elif ident in source:
            id_bonus = max(id_bonus, 0.10)

    return min(1.0, term_coverage + id_bonus)


def rerank_matches(question: str, matches: list, top_k: int) -> tuple[list[tuple[dict, float]], list[dict[str, Any]]]:
    question_terms, identifiers = tokenize_question(question)
    lexical_weight = min(max(settings.retrieval_lexical_weight, 0.0), 1.0)

    rescored: list[tuple[float, float, float, dict, object]] = []
    for match in matches:
        metadata = normalize_metadata(match)
        semantic = normalize_semantic_score(match_score(match))
        lexical = lexical_score(question_terms, identifiers, metadata)
        hybrid = ((1.0 - lexical_weight) * semantic) + (lexical_weight * lexical)
        rescored.append((hybrid, semantic, lexical, metadata, match))

    rescored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    deduped: list[tuple[dict, float]] = []
    debug_candidates: list[dict[str, Any]] = []
    seen_ranges: set[tuple[str, int, int]] = set()
    for hybrid, semantic, lexical, metadata, _ in rescored:
        file_path = str(metadata.get("file_path", "unknown"))
        line_start = safe_int(metadata.get("line_start")) or 1
        line_end = safe_int(metadata.get("line_end")) or line_start
        key = (file_path, line_start, line_end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        deduped.append((metadata, hybrid))
        if len(debug_candidates) < 24:
            debug_candidates.append(
                {
                    "file_path": file_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "hybrid_score": round(hybrid, 4),
                    "semantic_score": round(semantic, 4),
                    "lexical_score": round(lexical, 4),
                    "query_used": str(metadata.get("query_used", "")),
                    "section_name": str(metadata.get("section_name", "")),
                }
            )
        if len(deduped) >= top_k:
            break

    min_score = min(max(settings.retrieval_min_hybrid_score, 0.0), 1.0)
    strong = [item for item in deduped if item[1] >= min_score]
    return (strong or deduped), debug_candidates


def extract_citation(metadata: dict, score: float) -> dict:
    try:
        line_start = int(metadata.get("line_start", 1))
    except (TypeError, ValueError):
        line_start = 1
    try:
        line_end = int(metadata.get("line_end", line_start))
    except (TypeError, ValueError):
        line_end = line_start

    return {
        "file_path": metadata.get("file_path", "unknown"),
        "line_start": line_start,
        "line_end": line_end,
        "score": round(score, 4),
        "source_type": str(metadata.get("source_type", "repo")),
        "file_sha": metadata.get("file_sha"),
        "snippet": metadata_chunk_text(metadata)[:550] or None,
    }


def metadata_chunk_text(metadata: dict) -> str:
    return (
        metadata.get("chunk_text")
        or metadata.get("text")
        or metadata.get("content")
        or metadata.get("code")
        or ""
    )


def build_context(citations: list[Citation], chunks: list[str]) -> str:
    parts = []
    max_context = max(settings.rag_max_context_chunks, 1)
    for idx, (citation, chunk) in enumerate(zip(citations[:max_context], chunks[:max_context]), start=1):
        parts.append(
            (
                f"[{idx}] {citation.file_path}:{citation.line_start}-{citation.line_end}\n"
                f"{chunk.strip()}"
            )
        )
    return "\n\n".join(parts)


def query_namespaces() -> list[str]:
    namespaces = [settings.pinecone_namespace]
    fallback = (settings.pinecone_fallback_namespace or "").strip()
    if fallback and fallback not in namespaces:
        namespaces.append(fallback)
    return namespaces


def retrieve_citations_and_chunks(
    question: str,
    top_k: int,
    retrieval_queries: list[str] | None = None,
    namespaces: list[str] | None = None,
    source_type: str = "repo",
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    queries = retrieval_queries or [question]
    target_namespaces = namespaces or query_namespaces()
    candidate_top_k = min(
        max(top_k * max(settings.retrieval_candidate_multiplier, 1), top_k),
        max(settings.retrieval_max_candidates, top_k),
    )
    index = get_pinecone_index()

    matches: list = []
    subquery_counts: list[dict[str, Any]] = []
    for query_text in queries:
        question_vector = embed_question(query_text)
        if not question_vector:
            subquery_counts.append({"query": query_text, "matches": 0})
            continue

        query_matches: list = []
        for namespace in target_namespaces:
            results = call_with_retries(
                "pinecone query",
                lambda: index.query(
                    vector=question_vector,
                    top_k=candidate_top_k,
                    include_metadata=True,
                    namespace=namespace,
                ),
            )
            query_matches = normalize_matches(results)
            if query_matches:
                break

        for match in query_matches:
            metadata = normalize_metadata(match).copy()
            if str(metadata.get("_record_type", "")) == "attachment_file":
                continue
            if not metadata_chunk_text(metadata).strip():
                continue
            metadata["query_used"] = query_text
            metadata.setdefault("source_type", source_type)
            matches.append({"score": match_score(match), "metadata": metadata})
        subquery_counts.append({"query": query_text, "matches": len(query_matches)})

    if not matches:
        return [], [], {"candidates": [], "subqueries": subquery_counts}

    reranked, debug_candidates = rerank_matches(question=question, matches=matches, top_k=top_k)
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    debug = {"candidates": debug_candidates, "subqueries": subquery_counts}
    return citations, chunks, debug


def retrieve_uploaded_citations_and_chunks(
    question: str,
    top_k: int,
    uploaded_files: list[dict[str, Any]],
    retrieval_queries: list[str] | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    if not uploaded_files:
        return [], [], {"candidates": [], "subqueries": []}

    metadata_chunks, _ = build_attachment_chunks(uploaded_files, source_type="temp-upload")
    if not metadata_chunks:
        return [], [], {"candidates": [], "subqueries": []}

    matches: list[dict] = []
    subquery_counts: list[dict[str, Any]] = []
    queries = retrieval_queries or [question]
    for query_text in queries:
        question_vector = embed_question(query_text)
        if not question_vector:
            subquery_counts.append({"query": query_text, "matches": 0})
            continue

        count = 0
        for metadata_batch in batched(metadata_chunks, UPLOAD_EMBED_BATCH_SIZE):
            embeddings = embed_texts([metadata_chunk_text(metadata) for metadata in metadata_batch])
            for metadata, embedding in zip(metadata_batch, embeddings, strict=True):
                scoped = metadata.copy()
                scoped["query_used"] = query_text
                matches.append(
                    {
                        "score": cosine_similarity(question_vector, embedding),
                        "metadata": scoped,
                    }
                )
                count += 1
        subquery_counts.append({"query": query_text, "matches": count})

    if not matches:
        return [], [], {"candidates": [], "subqueries": subquery_counts}

    reranked, debug_candidates = rerank_matches(question=question, matches=matches, top_k=top_k)
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    debug = {"candidates": debug_candidates, "subqueries": subquery_counts}
    return citations, chunks, debug


def merge_citation_sets(
    sets: list[tuple[list[Citation], list[str], float]],
    top_k: int,
) -> tuple[list[Citation], list[str]]:
    ranked_items: list[tuple[float, Citation, str]] = []
    for citations, chunks, bonus in sets:
        for citation, chunk in zip(citations, chunks, strict=False):
            rank_score = min(1.0, max(0.0, float(citation.score) + bonus))
            ranked_items.append((rank_score, citation, chunk))

    ranked_items.sort(key=lambda item: item[0], reverse=True)

    merged_citations: list[Citation] = []
    merged_chunks: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for _, citation, chunk in ranked_items:
        key = (citation.file_path, citation.line_start, citation.line_end)
        if key in seen:
            continue
        seen.add(key)
        merged_citations.append(citation)
        merged_chunks.append(chunk)
        if len(merged_citations) >= top_k:
            break

    return merged_citations, merged_chunks


def retrieve_with_optional_uploads(
    question: str,
    top_k: int,
    uploaded_files: list[dict[str, Any]],
    scope: str,
    project_id: str,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    include_repo = scope in {"repo", "both"}
    include_uploads = scope in {"uploads", "both"}
    rewritten_query, subqueries = rewrite_and_decompose_query(question)
    retrieval_queries = [rewritten_query, *subqueries]
    index_citations: list[Citation] = []
    index_chunks: list[str] = []
    index_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    persistent_upload_citations: list[Citation] = []
    persistent_upload_chunks: list[str] = []
    persistent_upload_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    temp_upload_citations: list[Citation] = []
    temp_upload_chunks: list[str] = []
    temp_upload_debug: dict[str, Any] = {"candidates": [], "subqueries": []}

    if include_repo:
        try:
            index_citations, index_chunks, index_debug = retrieve_citations_and_chunks(
                question=rewritten_query,
                top_k=top_k,
                retrieval_queries=retrieval_queries,
                namespaces=query_namespaces(),
                source_type="repo",
            )
        except HTTPException as exc:
            logger.warning("Indexed retrieval unavailable: %s", exc.detail)
        except Exception as exc:
            logger.warning("Indexed retrieval failed: %s", exc)

    if include_uploads:
        attachment_ns = attachments_namespace(project_id)
        try:
            persistent_upload_citations, persistent_upload_chunks, persistent_upload_debug = retrieve_citations_and_chunks(
                question=rewritten_query,
                top_k=top_k,
                retrieval_queries=retrieval_queries,
                namespaces=[attachment_ns],
                source_type="upload",
            )
        except HTTPException as exc:
            logger.warning("Attachment retrieval unavailable: %s", exc.detail)
        except Exception as exc:
            logger.warning("Attachment retrieval failed: %s", exc)

        if uploaded_files:
            temp_upload_citations, temp_upload_chunks, temp_upload_debug = retrieve_uploaded_citations_and_chunks(
                question=rewritten_query,
                top_k=top_k,
                uploaded_files=uploaded_files,
                retrieval_queries=retrieval_queries,
            )

    sets_to_merge: list[tuple[list[Citation], list[str], float]] = []
    if include_repo:
        sets_to_merge.append((index_citations, index_chunks, 0.0))
    if include_uploads:
        sets_to_merge.append((persistent_upload_citations, persistent_upload_chunks, UPLOAD_SOURCE_BONUS))
        sets_to_merge.append((temp_upload_citations, temp_upload_chunks, UPLOAD_PRIORITY_BONUS))
    citations, chunks = merge_citation_sets(sets=sets_to_merge, top_k=top_k)

    upload_debug = {
        "persistent": persistent_upload_debug,
        "temporary": temp_upload_debug,
        "subqueries": [
            *(persistent_upload_debug.get("subqueries", []) or []),
            *(temp_upload_debug.get("subqueries", []) or []),
        ],
        "candidates": [
            *(persistent_upload_debug.get("candidates", []) or []),
            *(temp_upload_debug.get("candidates", []) or []),
        ][:32],
    }
    debug = {
        "rewritten_query": rewritten_query,
        "subqueries": subqueries,
        "index": index_debug,
        "uploads": upload_debug,
    }
    return citations, chunks, debug


def compute_evidence_strength(
    question: str,
    citations: list[Citation],
    retrieval_debug: dict[str, Any],
) -> dict[str, Any]:
    if not citations:
        return {
            "label": "Low",
            "score": 0.0,
            "reason": "No relevant citations were retrieved.",
            "metrics": {
                "top_score": 0.0,
                "score_gap": 0.0,
                "distinct_files": 0,
                "symbol_match_count": 0,
                "subquery_coverage": 0.0,
            },
        }

    top_score = max(0.0, min(1.0, float(citations[0].score)))
    second = max(0.0, min(1.0, float(citations[1].score))) if len(citations) > 1 else 0.0
    gap = max(0.0, top_score - second)
    distinct_files = len({citation.file_path for citation in citations})

    _, identifiers = tokenize_question(question)
    symbol_match_count = 0
    for citation in citations:
        source = f"{citation.file_path}\n{citation.snippet or ''}".lower()
        for ident in identifiers:
            if ident and ident in source:
                symbol_match_count += 1
                break

    subquery_rows = []
    subquery_rows.extend(((retrieval_debug.get("index", {}) or {}).get("subqueries", []) or []))
    subquery_rows.extend(((retrieval_debug.get("uploads", {}) or {}).get("subqueries", []) or []))
    dedup_subquery: dict[str, int] = {}
    for row in subquery_rows:
        query_text = str(row.get("query", "")).strip().lower()
        if not query_text:
            continue
        matches = safe_int(row.get("matches")) or 0
        dedup_subquery[query_text] = max(dedup_subquery.get(query_text, 0), matches)

    total_subqueries = len(dedup_subquery)
    covered_subqueries = sum(1 for count in dedup_subquery.values() if count > 0)
    subquery_coverage = (covered_subqueries / total_subqueries) if total_subqueries > 0 else 1.0

    normalized_files = min(distinct_files / 4.0, 1.0)
    normalized_symbols = min(symbol_match_count / max(len(citations), 1), 1.0)
    confidence_score = (
        (0.45 * top_score)
        + (0.15 * min(gap * 2.0, 1.0))
        + (0.15 * normalized_files)
        + (0.15 * normalized_symbols)
        + (0.10 * subquery_coverage)
    )
    confidence_score = round(max(0.0, min(1.0, confidence_score)), 3)

    label = "Low"
    if confidence_score >= 0.72:
        label = "High"
    elif confidence_score >= 0.48:
        label = "Medium"

    reason = {
        "High": "Strong retrieval agreement across high-scoring evidence chunks.",
        "Medium": "Useful evidence found, but some ambiguity remains.",
        "Low": "Retrieved evidence is weak or fragmented for this question.",
    }[label]

    return {
        "label": label,
        "score": confidence_score,
        "reason": reason,
        "metrics": {
            "top_score": round(top_score, 3),
            "score_gap": round(gap, 3),
            "distinct_files": distinct_files,
            "symbol_match_count": symbol_match_count,
            "subquery_coverage": round(subquery_coverage, 3),
        },
    }


def suggest_next_investigation(question: str, citations: list[Citation]) -> dict[str, list[str]]:
    file_suggestions: list[str] = []
    seen_files: set[str] = set()
    for citation in citations:
        if citation.file_path in seen_files:
            continue
        seen_files.add(citation.file_path)
        file_suggestions.append(citation.file_path)
        if len(file_suggestions) >= 3:
            break

    terms = sorted(tokenize_question(question)[1])
    term_suggestions = terms[:4]
    if not term_suggestions:
        words = sorted(tokenize_question(question)[0])
        term_suggestions = words[:4]

    if not file_suggestions:
        file_suggestions = ["src/", "conf/", "scripts/"]
    if not term_suggestions:
        term_suggestions = ["subroutine", "module", "common block", "makefile"]

    return {"files": file_suggestions, "terms": term_suggestions}


def insufficient_evidence_answer(question: str, suggestions: dict[str, list[str]]) -> str:
    files = ", ".join(suggestions.get("files", []))
    terms = ", ".join(suggestions.get("terms", []))
    return (
        "I don't have enough evidence from the repo to answer confidently.\n"
        f"Question: {question}\n"
        f"Next files to inspect: {files}\n"
        f"Next search terms: {terms}"
    )


def build_debug_payload(
    retrieval_debug: dict[str, Any],
    context: str,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    payload = {
        "rewritten_query": retrieval_debug.get("rewritten_query", ""),
        "subqueries": retrieval_debug.get("subqueries", []),
        "retrieval": {
            "index": retrieval_debug.get("index", {}),
            "uploads": retrieval_debug.get("uploads", {}),
        },
        "final_context_preview": context[:3000],
        "context_token_estimate": token_count(context),
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    return payload


def is_weak_evidence(evidence: dict[str, Any]) -> bool:
    label = str(evidence.get("label", "")).lower()
    score = float(evidence.get("score", 0.0) or 0.0)
    return label == "low" or score < 0.42


def generate_answer(question: str, context: str) -> str:
    client = get_openai_client()
    completion = call_with_retries(
        "openai chat completion",
        lambda: client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legacy code assistant. Use only retrieved context. "
                        "If context is insufficient, say so clearly and suggest next queries."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        "Answer with concise technical detail and reference citation numbers like [1], [2]."
                    ),
                },
            ],
        ),
    )
    content = completion.choices[0].message.content
    return content.strip() if content else "No answer generated."


@app.post("/api/search", response_model=SearchResponse)
def search(payload: QueryRequest) -> SearchResponse:
    started = time.perf_counter()
    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=[],
            scope=normalize_scope(payload.scope),
            project_id=normalize_project_id(payload.project_id),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector search failed: {exc}") from exc

    evidence = compute_evidence_strength(payload.question, citations, retrieval_debug)
    debug_payload: dict[str, Any] = {}
    if payload.debug:
        context = build_context(citations, chunks)
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return SearchResponse(matches=citations, evidence_strength=evidence, debug=debug_payload)


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=[],
            scope=normalize_scope(payload.scope),
            project_id=normalize_project_id(payload.project_id),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector retrieval failed: {exc}") from exc

    evidence = compute_evidence_strength(payload.question, citations, retrieval_debug)
    if not citations:
        suggestions = suggest_next_investigation(payload.question, citations)
        context = ""
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions),
            citations=[],
            evidence_strength=evidence,
            debug=debug_payload,
        )

    context = build_context(citations, chunks)
    if is_weak_evidence(evidence):
        suggestions = suggest_next_investigation(payload.question, citations)
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions),
            citations=citations,
            evidence_strength=evidence,
            debug=debug_payload,
        )

    try:
        answer = generate_answer(payload.question, context)
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Response validation error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    debug_payload: dict[str, Any] = {}
    if payload.debug:
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return QueryResponse(answer=answer, citations=citations, evidence_strength=evidence, debug=debug_payload)


@app.post("/api/uploads/ingest", response_model=UploadIngestResponse)
async def ingest_uploads(
    files: list[UploadFile] | None = File(default=None),
    project_id: str | None = Form(default=None),
    persist_uploads: str | None = Form(default=None),
) -> UploadIngestResponse:
    project = normalize_project_id(project_id)
    persist = parse_form_bool(persist_uploads)
    uploaded_files = await extract_uploaded_files(files or [])
    namespace = attachments_namespace(project)

    if not persist:
        statuses = [
            UploadIngestStatus(
                file_sha=str(item.get("file_sha", "")),
                file_name=str(item.get("safe_name", "upload.txt")),
                file_size=safe_int(item.get("file_size")) or 0,
                chunk_count=0,
                status="skipped",
                namespace=namespace,
                reason="persist_opt_in_required",
            )
            for item in uploaded_files
        ]
        return UploadIngestResponse(project_id=project, namespace=namespace, persisted=False, files=statuses)

    statuses = upsert_attachment_chunks(project_id=project, uploaded_files=uploaded_files)
    return UploadIngestResponse(project_id=project, namespace=namespace, persisted=True, files=statuses)


@app.get("/api/uploads", response_model=UploadListResponse)
def list_uploads(project_id: str | None = None) -> UploadListResponse:
    project = normalize_project_id(project_id)
    namespace = attachments_namespace(project)
    return UploadListResponse(project_id=project, namespace=namespace, files=list_upload_records(project))


@app.post("/api/uploads/{file_sha}/pin", response_model=UploadFileRecord)
def pin_upload(file_sha: str, payload: UploadPinRequest) -> UploadFileRecord:
    project = normalize_project_id(payload.project_id)
    record = get_upload_record(project, file_sha)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    record["pinned"] = bool(payload.pinned)
    upsert_upload_record(project, file_sha, record)
    return UploadFileRecord(
        project_id=project,
        namespace=attachments_namespace(project),
        file_sha=file_sha,
        file_name=str(record.get("file_name", "unknown")),
        file_size=safe_int(record.get("file_size")) or 0,
        chunk_count=safe_int(record.get("chunk_count")) or 0,
        uploaded_at=str(record.get("uploaded_at", "")),
        pinned=bool(record.get("pinned", False)),
        persisted=bool(record.get("persisted", True)),
    )


@app.delete("/api/uploads/{file_sha}", response_model=UploadDeleteResponse)
def delete_upload(file_sha: str, project_id: str | None = None) -> UploadDeleteResponse:
    project = normalize_project_id(project_id)
    record = get_upload_record(project, file_sha)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")

    chunk_count = safe_int(record.get("chunk_count")) or 0
    delete_attachment_vectors(project_id=project, file_sha=file_sha, chunk_count=chunk_count)
    remove_upload_record(project, file_sha)

    return UploadDeleteResponse(
        project_id=project,
        namespace=attachments_namespace(project),
        file_sha=file_sha,
        deleted=True,
    )


@app.post("/api/search/upload", response_model=SearchResponse)
async def search_with_uploads(
    question: str = Form(...),
    top_k: int = Form(5),
    files: list[UploadFile] | None = File(default=None),
    debug: str | None = Form(default=None),
    scope: str = Form("both"),
    project_id: str | None = Form(default=None),
    persist_uploads: str | None = Form(default=None),
) -> SearchResponse:
    payload = validate_query_request(
        question=question,
        top_k=top_k,
        debug=parse_form_bool(debug),
        scope=scope,
        project_id=project_id,
    )
    uploaded_files = await extract_uploaded_files(files or [])
    persist = parse_form_bool(persist_uploads)
    if persist and uploaded_files:
        upsert_attachment_chunks(project_id=payload.project_id, uploaded_files=uploaded_files)
    started = time.perf_counter()

    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=uploaded_files,
            scope=payload.scope,
            project_id=payload.project_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload retrieval failed: {exc}") from exc

    evidence = compute_evidence_strength(payload.question, citations, retrieval_debug)
    debug_payload: dict[str, Any] = {}
    if payload.debug:
        context = build_context(citations, chunks)
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return SearchResponse(matches=citations, evidence_strength=evidence, debug=debug_payload)


@app.post("/api/query/upload", response_model=QueryResponse)
async def query_with_uploads(
    question: str = Form(...),
    top_k: int = Form(5),
    files: list[UploadFile] | None = File(default=None),
    debug: str | None = Form(default=None),
    scope: str = Form("both"),
    project_id: str | None = Form(default=None),
    persist_uploads: str | None = Form(default=None),
) -> QueryResponse:
    payload = validate_query_request(
        question=question,
        top_k=top_k,
        debug=parse_form_bool(debug),
        scope=scope,
        project_id=project_id,
    )
    uploaded_files = await extract_uploaded_files(files or [])
    persist = parse_form_bool(persist_uploads)
    if persist and uploaded_files:
        upsert_attachment_chunks(project_id=payload.project_id, uploaded_files=uploaded_files)
    started = time.perf_counter()

    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=uploaded_files,
            scope=payload.scope,
            project_id=payload.project_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload retrieval failed: {exc}") from exc

    evidence = compute_evidence_strength(payload.question, citations, retrieval_debug)
    if not citations:
        suggestions = suggest_next_investigation(payload.question, citations)
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions),
            citations=[],
            evidence_strength=evidence,
            debug=debug_payload,
        )

    context = build_context(citations, chunks)
    if is_weak_evidence(evidence):
        suggestions = suggest_next_investigation(payload.question, citations)
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions),
            citations=citations,
            evidence_strength=evidence,
            debug=debug_payload,
        )

    try:
        answer = generate_answer(payload.question, context)
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Response validation error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    debug_payload: dict[str, Any] = {}
    if payload.debug:
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return QueryResponse(answer=answer, citations=citations, evidence_strength=evidence, debug=debug_payload)
