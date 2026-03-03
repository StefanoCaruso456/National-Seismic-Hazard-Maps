import logging
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    split_into_sections,
    split_large_text,
    token_count,
)

logger = logging.getLogger("legacylens.api")
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
SAFE_UPLOAD_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

UPLOAD_MAX_FILES = 8
UPLOAD_MAX_FILE_BYTES = 1_500_000
UPLOAD_MAX_TOTAL_BYTES = 6_000_000
UPLOAD_MIN_TOKENS = 140
UPLOAD_MAX_TOKENS = 420
UPLOAD_EMBED_BATCH_SIZE = 64
UPLOAD_PRIORITY_BONUS = 0.04
DEFAULT_REPO_URL = "https://github.com/StefanoCaruso456/National-Seismic-Hazard-Maps"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    debug: bool = False


class Citation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    score: float
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
    repo_url: str
    query_top_k_default: int = 5
    query_top_k_min: int = 1
    query_top_k_max: int = 20


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
    return RetrievalInfoResponse(
        lexical_weight=min(max(settings.retrieval_lexical_weight, 0.0), 1.0),
        min_hybrid_score=min(max(settings.retrieval_min_hybrid_score, 0.0), 1.0),
        candidate_multiplier=max(settings.retrieval_candidate_multiplier, 1),
        max_candidates=max(settings.retrieval_max_candidates, 1),
        rag_max_context_chunks=max(settings.rag_max_context_chunks, 1),
        upload_max_files=UPLOAD_MAX_FILES,
        upload_max_file_bytes=UPLOAD_MAX_FILE_BYTES,
        upload_max_total_bytes=UPLOAD_MAX_TOTAL_BYTES,
        repo_url=DEFAULT_REPO_URL,
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


def validate_query_request(question: str, top_k: int, debug: bool = False) -> QueryRequest:
    try:
        return QueryRequest(question=question, top_k=top_k, debug=debug)
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


async def extract_uploaded_files(files: list[UploadFile]) -> list[tuple[str, str]]:
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files attached. Max allowed is {UPLOAD_MAX_FILES}.",
        )

    total_bytes = 0
    uploaded: list[tuple[str, str]] = []
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

        text = decode_upload_bytes(raw)
        if not text.strip():
            continue

        uploaded.append((safe_upload_name(upload.filename or "", index), text))

    return uploaded


def build_upload_chunk_metadata(uploaded_files: list[tuple[str, str]]) -> list[dict]:
    metadata_chunks: list[dict] = []
    for filename, text in uploaded_files:
        file_path = f"uploaded/{filename}"
        suffix = Path(filename).suffix.lower()
        is_fortran = suffix in FORTRAN_EXTENSIONS

        if is_fortran:
            sections = split_into_sections(text)
            for section in sections:
                section_chunks = split_large_text(
                    lines=section.lines,
                    start_line=section.start_line,
                    section_name=section.name,
                    min_tokens=UPLOAD_MIN_TOKENS,
                    max_tokens=UPLOAD_MAX_TOKENS,
                )
                for chunk in section_chunks:
                    chunk.file_path = file_path
                    metadata_chunks.append(
                        {
                            "file_path": chunk.file_path,
                            "line_start": chunk.line_start,
                            "line_end": chunk.line_end,
                            "section_name": chunk.section_name,
                            "language": "fortran",
                            "chunk_text": chunk.chunk_text,
                        }
                    )
            continue

        lines = text.splitlines() or [text]
        section_chunks = split_large_text(
            lines=lines,
            start_line=1,
            section_name="file",
            min_tokens=UPLOAD_MIN_TOKENS,
            max_tokens=UPLOAD_MAX_TOKENS,
        )
        for chunk in section_chunks:
            chunk.file_path = file_path
            metadata_chunks.append(
                {
                    "file_path": chunk.file_path,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "section_name": chunk.section_name,
                    "language": "text",
                    "chunk_text": chunk.chunk_text,
                }
            )

    return metadata_chunks


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
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    queries = retrieval_queries or [question]
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
        for namespace in query_namespaces():
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
            metadata["query_used"] = query_text
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
    uploaded_files: list[tuple[str, str]],
    retrieval_queries: list[str] | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    if not uploaded_files:
        return [], [], {"candidates": [], "subqueries": []}

    metadata_chunks = build_upload_chunk_metadata(uploaded_files)
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
    index_citations: list[Citation],
    index_chunks: list[str],
    upload_citations: list[Citation],
    upload_chunks: list[str],
    top_k: int,
) -> tuple[list[Citation], list[str]]:
    ranked_items: list[tuple[float, Citation, str]] = []

    for citation, chunk in zip(index_citations, index_chunks, strict=False):
        ranked_items.append((float(citation.score), citation, chunk))

    for citation, chunk in zip(upload_citations, upload_chunks, strict=False):
        ranked_items.append((min(1.0, float(citation.score) + UPLOAD_PRIORITY_BONUS), citation, chunk))

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
    uploaded_files: list[tuple[str, str]],
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    index_citations: list[Citation] = []
    index_chunks: list[str] = []
    index_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    upload_citations: list[Citation] = []
    upload_chunks: list[str] = []
    upload_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    rewritten_query, subqueries = rewrite_and_decompose_query(question)
    retrieval_queries = [rewritten_query, *subqueries]

    if uploaded_files:
        try:
            index_citations, index_chunks, index_debug = retrieve_citations_and_chunks(
                question=rewritten_query,
                top_k=top_k,
                retrieval_queries=retrieval_queries,
            )
        except HTTPException as exc:
            logger.warning("Indexed retrieval unavailable while using upload scope: %s", exc.detail)
        except Exception as exc:
            logger.warning("Indexed retrieval failed while using upload scope: %s", exc)

        upload_citations, upload_chunks, upload_debug = retrieve_uploaded_citations_and_chunks(
            question=rewritten_query,
            top_k=top_k,
            uploaded_files=uploaded_files,
            retrieval_queries=retrieval_queries,
        )
        citations, chunks = merge_citation_sets(
            index_citations=index_citations,
            index_chunks=index_chunks,
            upload_citations=upload_citations,
            upload_chunks=upload_chunks,
            top_k=top_k,
        )
        debug = {
            "rewritten_query": rewritten_query,
            "subqueries": subqueries,
            "index": index_debug,
            "uploads": upload_debug,
        }
        return citations, chunks, debug

    citations, chunks, index_debug = retrieve_citations_and_chunks(
        question=rewritten_query,
        top_k=top_k,
        retrieval_queries=retrieval_queries,
    )
    debug = {
        "rewritten_query": rewritten_query,
        "subqueries": subqueries,
        "index": index_debug,
        "uploads": {"candidates": [], "subqueries": []},
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


@app.post("/api/search/upload", response_model=SearchResponse)
async def search_with_uploads(
    question: str = Form(...),
    top_k: int = Form(5),
    files: list[UploadFile] | None = File(default=None),
    debug: str | None = Form(default=None),
) -> SearchResponse:
    payload = validate_query_request(question=question, top_k=top_k, debug=parse_form_bool(debug))
    uploaded_files = await extract_uploaded_files(files or [])
    started = time.perf_counter()

    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=uploaded_files,
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
) -> QueryResponse:
    payload = validate_query_request(question=question, top_k=top_k, debug=parse_form_bool(debug))
    uploaded_files = await extract_uploaded_files(files or [])
    started = time.perf_counter()

    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=uploaded_files,
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
