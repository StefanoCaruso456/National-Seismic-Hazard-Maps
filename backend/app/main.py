import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.ingest import FORTRAN_EXTENSIONS, chunk_fortran_file, discover_fortran_files, token_count

logger = logging.getLogger("legacylens.api")
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    score: float
    snippet: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    matches: list[Citation] = Field(default_factory=list)


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


def rerank_matches(question: str, matches: list, top_k: int) -> list[tuple[dict, float]]:
    question_terms, identifiers = tokenize_question(question)
    lexical_weight = min(max(settings.retrieval_lexical_weight, 0.0), 1.0)

    rescored: list[tuple[float, float, dict, object]] = []
    for match in matches:
        metadata = normalize_metadata(match)
        semantic = normalize_semantic_score(match_score(match))
        lexical = lexical_score(question_terms, identifiers, metadata)
        hybrid = ((1.0 - lexical_weight) * semantic) + (lexical_weight * lexical)
        rescored.append((hybrid, semantic, metadata, match))

    rescored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    deduped: list[tuple[dict, float]] = []
    seen_ranges: set[tuple[str, int, int]] = set()
    for hybrid, _, metadata, _ in rescored:
        file_path = str(metadata.get("file_path", "unknown"))
        line_start = safe_int(metadata.get("line_start")) or 1
        line_end = safe_int(metadata.get("line_end")) or line_start
        key = (file_path, line_start, line_end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        deduped.append((metadata, hybrid))
        if len(deduped) >= top_k:
            break

    min_score = min(max(settings.retrieval_min_hybrid_score, 0.0), 1.0)
    strong = [item for item in deduped if item[1] >= min_score]
    return strong or deduped


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


def retrieve_citations_and_chunks(question: str, top_k: int) -> tuple[list[Citation], list[str]]:
    question_vector = embed_question(question)
    if not question_vector:
        return [], []

    candidate_top_k = min(
        max(top_k * max(settings.retrieval_candidate_multiplier, 1), top_k),
        max(settings.retrieval_max_candidates, top_k),
    )
    index = get_pinecone_index()

    matches: list = []
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
        matches = normalize_matches(results)
        if matches:
            break

    if not matches:
        return [], []

    reranked = rerank_matches(question=question, matches=matches, top_k=top_k)
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    return citations, chunks


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
    try:
        citations, _ = retrieve_citations_and_chunks(payload.question, payload.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector search failed: {exc}") from exc

    return SearchResponse(matches=citations)


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        citations, chunks = retrieve_citations_and_chunks(payload.question, payload.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector retrieval failed: {exc}") from exc

    if not citations:
        return QueryResponse(
            answer=(
                "I could not find relevant indexed code for that question. "
                "Try a more specific function name, file name, or keyword. "
                "If this keeps happening, your Pinecone namespace may be empty (run ingestion first)."
            ),
            citations=[],
        )

    context = build_context(citations, chunks)

    try:
        answer = generate_answer(payload.question, context)
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Response validation error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    return QueryResponse(answer=answer, citations=citations)
