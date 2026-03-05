import logging
import math
import re
import time
import hashlib
import io
import json
import subprocess
import threading
from collections import defaultdict
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
from app.gitnexus_client import GitNexusClient, GitNexusClientError
from app.hybrid import extract_ranked_candidate_files, normalize_file_path, should_run_impact
from app.ingest import (
    FORTRAN_EXTENSIONS,
    chunk_fortran_file,
    discover_fortran_files,
    token_count,
)
from app.router import (
    PLAN_GRAPH_ONLY,
    PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR,
    PLAN_GRAPH_PLUS_VECTOR,
    PLAN_KEYWORD_ONLY,
    PLAN_KEYWORD_PLUS_VECTOR,
    PLAN_VECTOR_ONLY,
    ROUTER_PINECONE_TOP_K,
    detect_route_signals,
    escalated_plan,
    low_confidence_reason,
    route_debug_template,
    select_retrieval_plan,
)

logger = logging.getLogger("legacylens.api")
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
IDENTIFIER_HEAVY_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}\b|\b[a-z][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*\b|\b[A-Za-z_]*\d+[A-Za-z0-9_]*\b|\b[A-Za-z_]+_[A-Za-z0-9_]+\b|\b[A-Za-z][A-Za-z0-9_]*-[A-Za-z0-9_-]+\b"
)
CALL_TARGET_PATTERN = re.compile(r"\bcalls?\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
FILE_TOKEN_PATTERN = re.compile(r"\b([a-zA-Z0-9_.-]+\.(?:f|for|f90|f95|f03|f08|inc|py|sh|txt|cfg|conf))\b", re.IGNORECASE)
SAFE_UPLOAD_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
SAFE_PROJECT_ID_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
FOCUS_TERM_QUOTED_PATTERN = re.compile(r"[\"'`]([^\"'`]{3,140})[\"'`]")
FOCUS_TERM_HYPHENATED_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*-[a-zA-Z0-9_-]+\b")
FOCUS_TERM_UNDERSCORED_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")
FORTRAN_DEF_PATTERN = re.compile(
    r"^\s*(program|subroutine|function|module(?!\s+procedure)|block\s+data|interface)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
FORTRAN_CALL_PATTERN = re.compile(r"\bcall\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
FORTRAN_USE_PATTERN = re.compile(
    r"^\s*use\b(?:\s*,\s*(?:intrinsic|non_intrinsic)\s*::|\s*,\s*only\s*:\s*|\s*::|\s+)?\s*([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
FORTRAN_INCLUDE_PATTERN = re.compile(r"^\s*include\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
FORTRAN_COMMON_PATTERN = re.compile(r"\bcommon\s*/\s*([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
FORTRAN_LOOP_START_PATTERN = re.compile(
    r"^\s*(?:\d+\s+)?do\b(?:\s+\d+|\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=)?",
    re.IGNORECASE,
)
FORTRAN_LOOP_END_PATTERN = re.compile(r"^\s*(?:end\s*do|enddo|\d+\s+continue)\b", re.IGNORECASE)
FORTRAN_IO_PATTERN = re.compile(r"\b(open|read|write|inquire|rewind|backspace|close)\b", re.IGNORECASE)
FORTRAN_ACTION_PATTERN = re.compile(
    r"\b(call|read|write|open|allocate|deallocate|stop|error|inquire|rewind|backspace|close)\b",
    re.IGNORECASE,
)
CONFIG_DIR_HINTS = ("conf/", "etc/", "scripts/", "makefile", "readme", "run_", "run.")
MODE_VALUES = {"chat", "search", "patterns", "dependencies", "hybrid", "graph"}
FILTERABLE_LANGUAGES = {"fortran", "text", "pdf"}
FILTERABLE_SOURCE_TYPES = {"repo", "upload", "temp-upload"}
STARTUP_SMOKE_MODES = {"off", "warn", "strict"}
QUERY_MAX_CHARS = 8000
IDENTIFIER_STOPWORDS = {
    "where",
    "which",
    "what",
    "when",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "the",
    "this",
    "that",
    "from",
    "with",
    "about",
    "into",
    "for",
    "and",
    "or",
    "show",
    "find",
    "list",
    "tell",
    "explain",
    "main",
    "entry",
    "point",
    "program",
    "code",
    "file",
    "files",
    "dependency",
    "dependencies",
    "impact",
    "impacts",
    "downstream",
    "upstream",
    "chain",
    "calls",
    "called",
    "routine",
    "routines",
    "citation",
    "citations",
    "function",
    "functions",
    "subroutine",
    "subroutines",
    "module",
    "modules",
    "error",
}
DEBUG_CANDIDATE_LIMIT = 64

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
EMBEDDING_METADATA_SCHEMA_VERSION = "v1"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=QUERY_MAX_CHARS)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="chat")
    scope: str = Field(default="both")
    project_id: str = Field(default="nshmp-main", min_length=1, max_length=80)
    path_prefix: str | None = Field(default=None, max_length=260)
    language: str | None = Field(default=None, max_length=40)
    source_type: str | None = Field(default=None, max_length=40)
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
    graph: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_strength: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class FileResult(BaseModel):
    file_path: str
    match_count: int
    max_score: float
    source_types: list[str] = Field(default_factory=list)


class DependencyEdge(BaseModel):
    edge_type: str
    caller: str
    caller_path: str
    caller_line: int
    target: str
    resolved: bool
    target_kind: str | None = None
    target_path: str | None = None
    target_line: int | None = None


class PatternExample(BaseModel):
    pattern_type: str
    summary: str
    file_path: str
    line_start: int
    line_end: int
    snippet: str


class SearchResponse(BaseModel):
    matches: list[Citation] = Field(default_factory=list)
    evidence_strength: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
    result_type: str = "Ranked Chunks"
    summary: str | None = None
    follow_ups: list[str] = Field(default_factory=list)
    file_results: list[FileResult] = Field(default_factory=list)
    graph_edges: list[DependencyEdge] = Field(default_factory=list)
    pattern_examples: list[PatternExample] = Field(default_factory=list)


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
    focus_term_guardrail_enabled: bool
    focus_term_absent_cap: float
    focus_term_partial_coverage_cap: float
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
    hybrid_top_k_default: int = 12
    hybrid_max_candidate_files: int = 50
    gitnexus_enabled: bool = True


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
gitnexus_client: GitNexusClient | None = None
startup_smoke_state: dict[str, Any] = {
    "mode": "off",
    "checked_at": None,
    "ok": None,
    "errors": [],
    "checks": {},
}
query_result_cache_lock = threading.Lock()
query_result_cache: dict[str, dict[str, Any]] = {}
lexical_candidate_cache_lock = threading.Lock()
lexical_candidate_cache: dict[str, dict[str, Any]] = {}


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def normalize_startup_smoke_mode() -> str:
    mode = (settings.startup_smoke_mode or "off").strip().lower()
    if mode in STARTUP_SMOKE_MODES:
        return mode
    logger.warning("Invalid STARTUP_SMOKE_MODE=%r. Falling back to 'off'.", settings.startup_smoke_mode)
    return "off"


def startup_smoke_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:280] if len(text) > 280 else text


def run_startup_smoke_probe() -> dict[str, Any]:
    mode = normalize_startup_smoke_mode()
    checked_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "mode": mode,
        "checked_at": checked_at,
        "ok": None,
        "errors": [],
        "checks": {},
    }
    if mode == "off":
        result["ok"] = True
        result["checks"] = {"startup_gate": {"ok": True, "skipped": True}}
        return result

    errors: list[str] = []
    checks: dict[str, Any] = {}
    probe_vector: list[float] | None = None

    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is not configured")
        checks["openai_embeddings"] = {"ok": False, "error": "missing_api_key"}
    else:
        try:
            openai_probe = OpenAI(api_key=settings.openai_api_key)
            embedding_response = call_with_retries(
                "startup openai embeddings",
                lambda: openai_probe.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=settings.startup_smoke_query,
                ),
            )
            probe_vector = list(embedding_response.data[0].embedding)
            checks["openai_embeddings"] = {
                "ok": True,
                "model": settings.openai_embedding_model,
                "dimension": len(probe_vector),
            }
        except Exception as exc:
            message = startup_smoke_error(exc)
            errors.append(f"OpenAI probe failed: {message}")
            checks["openai_embeddings"] = {
                "ok": False,
                "error": message,
                "model": settings.openai_embedding_model,
            }

    if not settings.pinecone_api_key:
        errors.append("PINECONE_API_KEY is not configured")
        checks["pinecone_index"] = {"ok": False, "error": "missing_api_key"}
    else:
        try:
            pinecone_probe = Pinecone(api_key=settings.pinecone_api_key)
            index = pinecone_probe.Index(settings.pinecone_index_name)
            description_obj = call_with_retries(
                "startup pinecone describe index",
                lambda: pinecone_probe.describe_index(settings.pinecone_index_name),
            )
            index_dimension = index_dimension_from_description(description_obj)
            stats_obj = call_with_retries("startup pinecone stats", lambda: index.describe_index_stats())
            stats = stats_obj if isinstance(stats_obj, dict) else getattr(stats_obj, "to_dict", lambda: {})()
            namespace_vector_count, _ = extract_vector_counts(stats, settings.pinecone_namespace)
            checks["pinecone_index"] = {
                "ok": True,
                "index": settings.pinecone_index_name,
                "namespace": settings.pinecone_namespace,
                "namespace_vector_count": namespace_vector_count,
                "dimension": index_dimension,
            }
            if probe_vector:
                call_with_retries(
                    "startup pinecone query",
                    lambda: index.query(
                        vector=probe_vector,
                        top_k=max(settings.startup_smoke_top_k, 1),
                        include_metadata=False,
                        namespace=settings.pinecone_namespace,
                    ),
                )
                checks["pinecone_query"] = {
                    "ok": True,
                    "top_k": max(settings.startup_smoke_top_k, 1),
                    "namespace": settings.pinecone_namespace,
                }
        except Exception as exc:
            message = startup_smoke_error(exc)
            errors.append(f"Pinecone probe failed: {message}")
            checks["pinecone_index"] = {
                "ok": False,
                "error": message,
                "index": settings.pinecone_index_name,
            }

    result["checks"] = checks
    result["errors"] = errors
    result["ok"] = len(errors) == 0
    return result


def apply_startup_smoke_probe() -> None:
    global startup_smoke_state
    startup_smoke_state = run_startup_smoke_probe()
    if startup_smoke_state.get("ok"):
        logger.info("Startup smoke probe passed (mode=%s).", startup_smoke_state.get("mode"))
        return

    errors = startup_smoke_state.get("errors", [])
    message = "; ".join(str(item) for item in errors if item) or "unknown error"
    if startup_smoke_state.get("mode") == "strict":
        raise RuntimeError(f"Startup smoke gate failed: {message}")
    logger.warning("Startup smoke probe failed (mode=%s): %s", startup_smoke_state.get("mode"), message)


def enforce_startup_smoke_gate() -> None:
    if startup_smoke_state.get("mode") != "strict":
        return
    if startup_smoke_state.get("ok") is False:
        message = "; ".join(str(item) for item in startup_smoke_state.get("errors", [])) or "startup smoke probe failed"
        raise HTTPException(status_code=503, detail=f"Startup smoke gate failed: {message}")


@app.on_event("startup")
def startup_event() -> None:
    apply_startup_smoke_probe()


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_gitnexus_client()


@app.get("/health")
def health() -> dict:
    mode = str(startup_smoke_state.get("mode", "off"))
    smoke_ok = startup_smoke_state.get("ok")
    health_status = "ok"
    if mode != "off" and smoke_ok is False:
        health_status = "degraded"
    return {
        "status": health_status,
        "env": settings.app_env,
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
        "startup_smoke": startup_smoke_state,
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
        focus_term_guardrail_enabled=bool(settings.retrieval_focus_term_guardrail_enabled),
        focus_term_absent_cap=min(max(float(settings.retrieval_focus_term_absent_cap), 0.0), 1.0),
        focus_term_partial_coverage_cap=min(max(float(settings.retrieval_focus_term_partial_coverage_cap), 0.0), 1.0),
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
        hybrid_top_k_default=max(settings.hybrid_top_k_default, 1),
        hybrid_max_candidate_files=max(settings.hybrid_max_candidate_files, 1),
        gitnexus_enabled=bool(settings.gitnexus_enabled),
    )


def get_openai_client() -> OpenAI:
    global openai_client
    enforce_startup_smoke_gate()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    if openai_client is None:
        openai_client = OpenAI(api_key=settings.openai_api_key)
    return openai_client


def get_pinecone_index():
    global pinecone_client
    enforce_startup_smoke_gate()
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not configured")
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return pinecone_client.Index(settings.pinecone_index_name)


def default_gitnexus_repo() -> str:
    configured = (settings.gitnexus_default_repo or "").strip()
    if configured:
        return configured
    return repo_root_path().name


def get_gitnexus_client() -> GitNexusClient:
    global gitnexus_client
    if not settings.gitnexus_enabled:
        raise HTTPException(status_code=503, detail="GitNexus is disabled by configuration")
    if gitnexus_client is None:
        gitnexus_client = GitNexusClient(
            command=settings.gitnexus_mcp_command,
            default_repo=default_gitnexus_repo(),
            startup_timeout_seconds=settings.gitnexus_startup_timeout_seconds,
            call_timeout_seconds=settings.gitnexus_call_timeout_seconds,
        )
    return gitnexus_client


def close_gitnexus_client() -> None:
    global gitnexus_client
    if gitnexus_client is None:
        return
    try:
        gitnexus_client.close()
    except Exception:
        pass
    finally:
        gitnexus_client = None


def infer_hybrid_target(question: str) -> str | None:
    for value in CALL_TARGET_PATTERN.findall(question):
        normalized = normalize_focus_term(str(value))
        lowered = normalized.lower()
        if normalized and lowered not in IDENTIFIER_STOPWORDS:
            return normalized

    focus_terms = extract_focus_terms(question)
    if focus_terms:
        return str(focus_terms[0]).strip()

    identifier_hints = extract_identifier_hints(question)
    for ident in identifier_hints:
        lowered = str(ident).lower().strip()
        if lowered and lowered not in IDENTIFIER_STOPWORDS:
            return ident

    _, identifiers = tokenize_question(question)
    if not identifiers:
        return None
    ranked = sorted(identifiers, key=lambda item: (-len(item), item))
    for ident in ranked:
        if ident in IDENTIFIER_STOPWORDS:
            continue
        return ident
    return None


def merge_impact_by_depth(
    upstream: dict[str, Any] | None,
    downstream: dict[str, Any] | None,
) -> dict[str, Any]:
    merged_rows: dict[str, list[dict[str, Any]]] = {}
    for payload in (upstream, downstream):
        if not isinstance(payload, dict):
            continue
        by_depth = payload.get("byDepth", {})
        if not isinstance(by_depth, dict):
            continue
        for depth_key, values in by_depth.items():
            if not isinstance(values, list):
                continue
            bucket = merged_rows.setdefault(str(depth_key), [])
            for row in values:
                if isinstance(row, dict):
                    bucket.append(row)
    target = None
    if isinstance(upstream, dict):
        target = upstream.get("target")
    if target is None and isinstance(downstream, dict):
        target = downstream.get("target")
    return {"target": target, "byDepth": merged_rows}


def graph_entrypoints(query_result: dict[str, Any]) -> list[str]:
    processes = query_result.get("processes", []) if isinstance(query_result, dict) else []
    if not isinstance(processes, list):
        return []
    labels: list[str] = []
    for row in processes:
        if not isinstance(row, dict):
            continue
        summary = str(row.get("summary") or row.get("id") or "").strip()
        if not summary:
            continue
        labels.append(summary)
        if len(labels) >= 8:
            break
    return dedupe_preserve_order(labels)


def run_gitnexus_graph(question: str, repo_name: str | None = None) -> dict[str, Any]:
    selected_repo = (repo_name or default_gitnexus_repo()).strip()
    graph: dict[str, Any] = {
        "repo": selected_repo,
        "processes": [],
        "entrypoints": [],
        "impact": {},
        "candidate_files": [],
        "candidate_file_ranking": [],
        "target_symbol": None,
        "errors": [],
    }
    if not settings.gitnexus_enabled:
        graph["errors"] = ["gitnexus_disabled"]
        return graph

    try:
        client = get_gitnexus_client()
        query_result = client.query(
            query=question,
            repo=selected_repo,
            limit=8,
            max_symbols=20,
            include_content=False,
        )
        graph["processes"] = list(query_result.get("processes", [])) if isinstance(query_result, dict) else []
        graph["entrypoints"] = graph_entrypoints(query_result if isinstance(query_result, dict) else {})

        target = infer_hybrid_target(question)
        graph["target_symbol"] = target

        context_result: dict[str, Any] = {}
        if target:
            context_result = client.context(name=target, repo=selected_repo, include_content=False)
            if isinstance(context_result, dict) and context_result.get("error"):
                graph["errors"].append(str(context_result.get("error")))

        upstream_impact: dict[str, Any] = {}
        downstream_impact: dict[str, Any] = {}
        if target and should_run_impact(question):
            upstream_impact = client.impact(
                target=target,
                direction="upstream",
                repo=selected_repo,
                max_depth=3,
                min_confidence=0.75,
                include_tests=False,
            )
            downstream_impact = client.impact(
                target=target,
                direction="downstream",
                repo=selected_repo,
                max_depth=3,
                min_confidence=0.75,
                include_tests=False,
            )
        graph["impact"] = {"upstream": upstream_impact, "downstream": downstream_impact}

        merged_impact = merge_impact_by_depth(upstream_impact, downstream_impact)
        candidate_files, candidate_ranking = extract_ranked_candidate_files(
            query_result=query_result if isinstance(query_result, dict) else {},
            context_result=context_result if isinstance(context_result, dict) else {},
            impact_result=merged_impact,
            max_candidate_files=max(settings.hybrid_max_candidate_files, 1),
        )
        graph["candidate_files"] = candidate_files
        graph["candidate_file_ranking"] = candidate_ranking
    except (GitNexusClientError, HTTPException) as exc:
        graph["errors"] = [str(exc)]
    except Exception as exc:
        graph["errors"] = [f"gitnexus_unavailable: {exc}"]

    return graph


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


def normalize_index_description_payload(description: object) -> dict[str, Any]:
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


def index_dimension_from_description(description: object) -> int | None:
    attr_dimension = safe_int(getattr(description, "dimension", None))
    if attr_dimension and attr_dimension > 0:
        return attr_dimension
    payload = normalize_index_description_payload(description)
    for key in ("dimension", "vector_dimension"):
        value = safe_int(payload.get(key))
        if value and value > 0:
            return value
    return None


@lru_cache(maxsize=4)
def cached_index_dimension(index_name: str) -> int | None:
    if not settings.pinecone_api_key:
        return None
    pinecone_probe = Pinecone(api_key=settings.pinecone_api_key)
    if not hasattr(pinecone_probe, "describe_index"):
        return None
    description = call_with_retries(
        "pinecone describe index",
        lambda: pinecone_probe.describe_index(index_name),
    )
    return index_dimension_from_description(description)


def ensure_embedding_dimension_matches_index(embedding_dim: int, index_dim: int | None = None) -> None:
    if not settings.enforce_embedding_dimension:
        return
    expected = index_dim if index_dim is not None else cached_index_dimension(settings.pinecone_index_name)
    if expected is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to determine Pinecone index dimension for compatibility check. "
                "Set ENFORCE_EMBEDDING_DIMENSION=false to bypass."
            ),
        )
    if embedding_dim != expected:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embedding dimension mismatch: embedding={embedding_dim}, index={expected}. "
                "Update embedding model or recreate the Pinecone index."
            ),
        )


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


def normalize_mode(value: str | None) -> str:
    mode = (value or "chat").strip().lower()
    if mode in MODE_VALUES:
        return mode
    raise HTTPException(
        status_code=422,
        detail="mode must be one of: chat, search, patterns, dependencies, graph, hybrid",
    )


def normalize_path_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace("\\", "/")
    cleaned = cleaned.lstrip("/")
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    if not cleaned:
        return None
    if ".." in cleaned.split("/"):
        raise HTTPException(status_code=422, detail="path_prefix cannot contain parent traversal segments")
    return cleaned[:260]


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    language = str(value).strip().lower()
    if not language:
        return None
    if language not in FILTERABLE_LANGUAGES:
        supported = ", ".join(sorted(FILTERABLE_LANGUAGES))
        raise HTTPException(status_code=422, detail=f"language must be one of: {supported}")
    return language


def normalize_source_type(value: str | None) -> str | None:
    if value is None:
        return None
    source_type = str(value).strip().lower()
    if not source_type:
        return None
    if source_type not in FILTERABLE_SOURCE_TYPES:
        supported = ", ".join(sorted(FILTERABLE_SOURCE_TYPES))
        raise HTTPException(status_code=422, detail=f"source_type must be one of: {supported}")
    return source_type


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
    mode: str = "chat",
    scope: str = "both",
    project_id: str | None = None,
    path_prefix: str | None = None,
    language: str | None = None,
    source_type: str | None = None,
) -> QueryRequest:
    try:
        return QueryRequest(
            question=question,
            top_k=top_k,
            debug=debug,
            mode=normalize_mode(mode),
            scope=normalize_scope(scope),
            project_id=normalize_project_id(project_id),
            path_prefix=normalize_path_prefix(path_prefix),
            language=normalize_language(language),
            source_type=normalize_source_type(source_type),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def batched(items: list[Any], size: int):
    if size <= 0:
        size = 1
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(
    texts: list[str],
    check_index_dimension: bool = False,
    index_dim: int | None = None,
) -> list[list[float]]:
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
    embeddings = [list(item.embedding) for item in response.data]
    if check_index_dimension and embeddings:
        ensure_embedding_dimension_matches_index(len(embeddings[0]), index_dim=index_dim)
    return embeddings


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
    index_dim: int | None = None
    if settings.enforce_embedding_dimension:
        try:
            index_dim = cached_index_dimension(settings.pinecone_index_name)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to resolve Pinecone index dimension: {startup_smoke_error(exc)}",
            ) from exc

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
            embeddings = embed_texts(
                [metadata_chunk_text(metadata) for metadata in metadata_batch],
                check_index_dimension=True,
                index_dim=index_dim,
            )
            for metadata, embedding in zip(metadata_batch, embeddings, strict=True):
                vector_id = attachment_chunk_id(file_sha, next_chunk_index)
                next_chunk_index += 1
                enriched = metadata.copy()
                enriched["_record_type"] = "attachment_chunk"
                enriched["project_id"] = project
                enriched["embedding_model"] = settings.openai_embedding_model
                enriched["embedding_provider"] = "openai"
                enriched["embedding_dimension"] = len(embedding)
                enriched["embedding_schema_version"] = EMBEDDING_METADATA_SCHEMA_VERSION
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

        marker_embedding = embed_texts(
            [f"attachment marker {file_sha} {file_name}"],
            check_index_dimension=True,
            index_dim=index_dim,
        )[0]
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
            "embedding_model": settings.openai_embedding_model,
            "embedding_provider": "openai",
            "embedding_dimension": len(marker_embedding),
            "embedding_schema_version": EMBEDDING_METADATA_SCHEMA_VERSION,
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


def normalize_focus_term(term: str) -> str:
    trimmed = " ".join(term.strip().strip("`'\"").split())
    return trimmed.strip(".,:;()[]{}")


def is_likely_code_focus_term(term: str) -> bool:
    normalized = normalize_focus_term(term)
    if len(normalized) < 4:
        return False
    if "_" in normalized:
        return True
    if "-" in normalized:
        if any(char.isupper() for char in normalized):
            return True
        if any(char.isdigit() for char in normalized):
            return True
        return len(normalized) >= 12
    if " " in normalized:
        return "_" in normalized or "-" in normalized
    return False


def extract_focus_terms(question: str) -> list[str]:
    terms: list[str] = []
    for value in FOCUS_TERM_QUOTED_PATTERN.findall(question):
        normalized = normalize_focus_term(value)
        if normalized and is_likely_code_focus_term(normalized):
            terms.append(normalized.lower())
    for value in FOCUS_TERM_HYPHENATED_PATTERN.findall(question):
        normalized = normalize_focus_term(value)
        if normalized and is_likely_code_focus_term(normalized):
            terms.append(normalized.lower())
    for value in FOCUS_TERM_UNDERSCORED_PATTERN.findall(question):
        normalized = normalize_focus_term(value)
        if normalized and is_likely_code_focus_term(normalized):
            terms.append(normalized.lower())
    return dedupe_preserve_order(terms)


def focus_term_present(term: str, source_text: str) -> bool:
    lookup = term.lower().strip()
    if not lookup:
        return False
    lower_source = source_text.lower()
    if lookup in lower_source:
        return True
    compact_lookup = re.sub(r"[-_\s]+", "", lookup)
    if len(compact_lookup) < 4:
        return False
    compact_source = re.sub(r"[-_\s]+", "", lower_source)
    return compact_lookup in compact_source


def focus_term_match_count(required_terms: list[str], source_text: str) -> int:
    if not required_terms:
        return 0
    return sum(1 for term in required_terms if focus_term_present(term, source_text))


def analyze_focus_term_alignment(
    question: str,
    citations: list[Citation],
    chunks: list[str],
) -> dict[str, Any]:
    required_terms = extract_focus_terms(question)
    if not required_terms:
        return {
            "enabled": bool(settings.retrieval_focus_term_guardrail_enabled),
            "required_terms": [],
            "matched_terms": [],
            "unmatched_terms": [],
            "coverage": 1.0,
            "matched_citation_count": 0,
            "action": "none",
        }

    matched_terms: set[str] = set()
    matched_citation_count = 0
    for citation, chunk in zip(citations, chunks, strict=False):
        source = f"{citation.file_path}\n{citation.snippet or ''}\n{chunk}"
        local_hits = 0
        for term in required_terms:
            if focus_term_present(term, source):
                matched_terms.add(term)
                local_hits += 1
        if local_hits > 0:
            matched_citation_count += 1

    matched_list = sorted(matched_terms)
    unmatched_list = [term for term in required_terms if term not in matched_terms]
    coverage = (len(matched_list) / len(required_terms)) if required_terms else 1.0

    return {
        "enabled": bool(settings.retrieval_focus_term_guardrail_enabled),
        "required_terms": required_terms,
        "matched_terms": matched_list,
        "unmatched_terms": unmatched_list,
        "coverage": round(coverage, 3),
        "matched_citation_count": matched_citation_count,
        "action": "none",
    }


def missing_focus_terms_from_debug(retrieval_debug: dict[str, Any]) -> list[str]:
    guardrail = retrieval_debug.get("focus_guardrail", {}) if isinstance(retrieval_debug, dict) else {}
    unmatched = guardrail.get("unmatched_terms", []) if isinstance(guardrail, dict) else []
    if not isinstance(unmatched, list):
        return []
    terms = [str(item).strip().lower() for item in unmatched if str(item).strip()]
    return dedupe_preserve_order(terms)


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


def is_exhaustive_file_query(question: str) -> bool:
    lowered = question.lower()
    return (
        lowered.startswith("find all files")
        or "all references" in lowered
        or "all files that" in lowered
        or "all files where" in lowered
        or ("find all" in lowered and "files" in lowered)
    )


def is_config_query(question: str) -> bool:
    lowered = question.lower()
    hints = (
        "config",
        "configuration",
        "parameter file",
        "input deck",
        "control file",
        "conf/",
        "etc/",
        "run script",
    )
    return any(term in lowered for term in hints)


def is_dependency_query(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("depend", "dependency", "call graph", "who calls", "callers"))


def search_expansion_terms(question: str) -> list[str]:
    lowered = question.lower()
    expansions: list[str] = []
    if "ground motion" in lowered or "gmpe" in lowered:
        expansions.extend(
            [
                "gmpe",
                "ground motion prediction equation",
                "attenuation model",
                "nga",
                "pga",
                "sigma",
                "spectral acceleration",
            ]
        )
    if "hazard curve" in lowered:
        expansions.extend(["hazcurv", "hazpoint", "hazall", "deagg", "probability"])
    if is_config_query(question):
        expansions.extend(["conf/", "etc/", "scripts/", "Makefile", "README", "input"])
    return expansions


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

    if is_config_query(base):
        subqueries.extend(
            [
                f"{base} conf/",
                f"{base} etc/",
                f"{base} scripts/",
                f"{base} Makefile",
                f"{base} README",
                f"{base} input deck",
            ]
        )

    if is_dependency_query(base):
        subqueries.extend(
            [
                f"{base} call",
                f"{base} use",
                f"{base} include",
                f"{base} common block",
            ]
        )

    if is_exhaustive_file_query(base):
        subqueries.extend([f"{base} file path", f"{base} implementation", f"{base} source file"])

    for expansion in search_expansion_terms(base):
        subqueries.extend([expansion, f"{base} {expansion}"])

    # Include identifier-focused probes for better lexical+semantic union in legacy code.
    _, identifiers = tokenize_question(base)
    for ident in sorted(identifiers)[:4]:
        subqueries.extend([f"{ident} subroutine", f"{ident} common block"])

    return base, dedupe_preserve_order(subqueries)


def parse_form_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalized_request_filters(
    path_prefix: str | None,
    language: str | None,
    source_type: str | None,
) -> tuple[str | None, str | None, str | None]:
    return (
        normalize_path_prefix(path_prefix),
        normalize_language(language),
        normalize_source_type(source_type),
    )


def extract_identifier_hints(question: str) -> list[str]:
    hints: list[str] = []
    for token in extract_focus_terms(question):
        normalized = token.lower().strip()
        if normalized and normalized not in IDENTIFIER_STOPWORDS:
            hints.append(normalized)
    for token in IDENTIFIER_HEAVY_PATTERN.findall(question):
        normalized = str(token).strip("`'\"").lower()
        if len(normalized) >= 3 and normalized not in IDENTIFIER_STOPWORDS:
            hints.append(normalized)
    for token in CALL_TARGET_PATTERN.findall(question):
        normalized = str(token).strip().lower()
        if len(normalized) >= 3 and normalized not in IDENTIFIER_STOPWORDS:
            hints.append(normalized)
    for token in FILE_TOKEN_PATTERN.findall(question):
        normalized = str(token).strip().lower()
        if len(normalized) >= 3 and normalized not in IDENTIFIER_STOPWORDS:
            hints.append(normalized)
    _, identifiers = tokenize_question(question)
    for token in identifiers:
        if (
            ("_" in token or any(ch.isdigit() for ch in token) or token.upper() == token)
            and token not in IDENTIFIER_STOPWORDS
        ):
            hints.append(token.lower())
    return dedupe_preserve_order([item for item in hints if item and len(item) >= 3])[:8]


def lexical_cache_get(key: str) -> dict[str, Any] | None:
    ttl = max(float(settings.retrieval_query_cache_ttl_seconds), 0.0)
    if ttl <= 0:
        return None
    now = time.time()
    with lexical_candidate_cache_lock:
        row = lexical_candidate_cache.get(key)
        if not isinstance(row, dict):
            return None
        expires_at = float(row.get("expires_at", 0.0) or 0.0)
        if expires_at < now:
            lexical_candidate_cache.pop(key, None)
            return None
        payload = row.get("payload")
        return payload if isinstance(payload, dict) else None


def lexical_cache_put(key: str, payload: dict[str, Any]) -> None:
    ttl = max(float(settings.retrieval_query_cache_ttl_seconds), 0.0)
    if ttl <= 0:
        return
    now = time.time()
    with lexical_candidate_cache_lock:
        lexical_candidate_cache[key] = {"expires_at": now + ttl, "payload": payload}
        if len(lexical_candidate_cache) > max(int(settings.retrieval_query_cache_max_entries), 32):
            stale = sorted(
                lexical_candidate_cache.items(),
                key=lambda item: float((item[1] or {}).get("expires_at", 0.0)),
            )
            for stale_key, _ in stale[: max(len(stale) - int(settings.retrieval_query_cache_max_entries), 1)]:
                lexical_candidate_cache.pop(stale_key, None)


def lexical_candidate_files(question: str) -> dict[str, Any]:
    if not settings.retrieval_identifier_lexical_enabled:
        return {
            "enabled": False,
            "identifiers": [],
            "candidate_files": [],
            "file_scores": {},
            "hits": [],
            "errors": [],
        }

    identifiers = extract_identifier_hints(question)
    payload: dict[str, Any] = {
        "enabled": bool(identifiers),
        "identifiers": identifiers,
        "candidate_files": [],
        "file_scores": {},
        "hits": [],
        "errors": [],
    }
    if not identifiers:
        return payload

    cache_key = hashlib.sha1(f"lexical:{question.strip().lower()}".encode("utf-8")).hexdigest()
    cached = lexical_cache_get(cache_key)
    if cached:
        return cached

    root = repo_root_path()
    score_map: dict[str, float] = {}
    hits: list[dict[str, Any]] = []
    errors: list[str] = []
    globs = ["*.f", "*.for", "*.f90", "*.f95", "*.f03", "*.f08", "*.inc", "*.sh", "*.py", "*.txt", "*.conf", "*.cfg"]

    for token in identifiers:
        token_pattern = rf"\b{re.escape(token)}\b" if re.fullmatch(r"[a-zA-Z0-9_]+", token) else re.escape(token)
        args = ["rg", "--no-heading", "--line-number", "--color", "never", "-S", "-i", "-m", "120", "-e", token_pattern]
        for glob in globs:
            args.extend(["-g", glob])
        args.append(".")
        try:
            proc = subprocess.run(
                args,
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                timeout=1.5,
            )
        except FileNotFoundError:
            errors.append("rg_not_available")
            break
        except Exception as exc:
            errors.append(f"rg_error:{exc.__class__.__name__}")
            continue
        if proc.returncode not in {0, 1}:
            continue

        for raw_line in proc.stdout.splitlines():
            if raw_line.count(":") < 2:
                continue
            path_part, line_part, snippet = raw_line.split(":", 2)
            normalized_path = normalize_file_path(path_part)
            if not normalized_path:
                continue
            line_no = safe_int(line_part) or 1
            snippet_lower = snippet.lower()
            weight = 1.0
            if re.search(rf"\b(subroutine|function|module)\s+{re.escape(token)}\b", snippet_lower):
                weight = 3.2
            elif re.search(rf"\bcall\s+{re.escape(token)}\b", snippet_lower):
                weight = 2.7
            elif token in Path(normalized_path).name.lower():
                weight = 2.0
            score_map[normalized_path] = score_map.get(normalized_path, 0.0) + weight
            if len(hits) < 80:
                hits.append({"token": token, "file_path": normalized_path, "line": line_no, "weight": round(weight, 3)})

    ranked = sorted(score_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    file_limit = max(int(settings.retrieval_identifier_file_limit), 1)
    candidate_files = [path for path, _ in ranked[:file_limit]]
    max_score = ranked[0][1] if ranked else 1.0
    file_scores = {path: round(score / max(max_score, 0.001), 4) for path, score in ranked[:file_limit]}

    payload = {
        "enabled": True,
        "identifiers": identifiers,
        "candidate_files": candidate_files,
        "file_scores": file_scores,
        "hits": hits,
        "errors": dedupe_preserve_order(errors)[:3],
    }
    lexical_cache_put(cache_key, payload)
    return payload


def token_overlap_score(query_terms: set[str], source_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & source_terms) / max(len(query_terms), 1)


def count_exact_identifier_hits(identifiers: list[str], source: str) -> int:
    if not identifiers:
        return 0
    lowered = source.lower()
    hits = 0
    for ident in identifiers:
        probe = ident.lower()
        if not probe:
            continue
        if re.search(rf"(?<![a-zA-Z0-9_]){re.escape(probe)}(?![a-zA-Z0-9_])", lowered):
            hits += 1
    return hits


def action_statement_score(text: str) -> float:
    if not text:
        return 0.0
    count = len(FORTRAN_ACTION_PATTERN.findall(text))
    return min(1.0, count / 6.0)


def comment_density_penalty(text: str) -> float:
    lines = [line for line in text.splitlines()]
    if not lines:
        return 0.0
    blank_or_comment = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_or_comment += 1
            continue
        if stripped.startswith("!") or re.match(r"^[cC*]\b", stripped):
            blank_or_comment += 1
    ratio = blank_or_comment / max(len(lines), 1)
    return min(1.0, ratio)


def rerank_matches(question: str, matches: list, top_k: int) -> tuple[list[tuple[dict, float]], list[dict[str, Any]]]:
    if not settings.retrieval_deterministic_rerank_enabled:
        ranked = sorted(matches, key=match_score, reverse=True)
        deduped: list[tuple[dict, float]] = []
        debug_candidates: list[dict[str, Any]] = []
        seen_ranges: set[tuple[str, int, int]] = set()
        for rank, match in enumerate(ranked, start=1):
            metadata = normalize_metadata(match)
            file_path = str(metadata.get("file_path", "unknown"))
            line_start = safe_int(metadata.get("line_start")) or safe_int(metadata.get("start_line")) or 1
            line_end = safe_int(metadata.get("line_end")) or safe_int(metadata.get("end_line")) or line_start
            key = (file_path, line_start, line_end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            semantic = normalize_semantic_score(match_score(match))
            if len(deduped) < max(top_k, 1):
                deduped.append((metadata, float(semantic)))
            if len(debug_candidates) < max(DEBUG_CANDIDATE_LIMIT, max(top_k, 1)):
                snippet = " ".join(metadata_chunk_text(metadata).split())[:260]
                debug_candidates.append(
                    {
                        "file_path": file_path,
                        "line_start": line_start,
                        "line_end": line_end,
                        "query_used": str(metadata.get("query_used", "")),
                        "section_name": str(metadata.get("section_name", "")),
                        "semantic_rank": rank,
                        "rerank_rank": rank,
                        "semantic_score": round(float(semantic), 4),
                        "hybrid_score": round(float(semantic), 4),
                        "token_overlap": 0.0,
                        "exact_identifier_hits": 0,
                        "symbol_module_score": 0.0,
                        "action_score": 0.0,
                        "comment_penalty": 0.0,
                        "lexical_file_boost": 0.0,
                        "snippet": snippet,
                    }
                )
            if len(deduped) >= max(top_k, 1) and len(debug_candidates) >= max(DEBUG_CANDIDATE_LIMIT, max(top_k, 1)):
                break
        return deduped, debug_candidates

    question_terms, _ = tokenize_question(question)
    identifiers = extract_identifier_hints(question)
    focus_terms = extract_focus_terms(question)
    lexical_weight = min(max(settings.retrieval_lexical_weight, 0.0), 1.0)
    semantic_weight = max(0.2, min(0.9, 1.0 - lexical_weight))
    lexical_mix_weight = 1.0 - semantic_weight

    rows: list[dict[str, Any]] = []
    for ordinal, match in enumerate(matches, start=1):
        metadata = normalize_metadata(match)
        chunk_text = metadata_chunk_text(metadata)
        file_path = str(metadata.get("file_path", "unknown"))
        section_name = str(metadata.get("section_name", ""))
        symbol_name = str(metadata.get("symbol_name", "")).lower()
        module_name = str(metadata.get("module_name", "")).lower()
        source = f"{file_path}\n{section_name}\n{chunk_text}"
        source_terms = {token for token in WORD_PATTERN.findall(source.lower()) if len(token) >= 2}

        semantic = normalize_semantic_score(match_score(match))
        overlap = token_overlap_score(question_terms, source_terms)
        exact_hits = count_exact_identifier_hits(identifiers, source)
        exact_score = min(1.0, exact_hits / max(len(identifiers), 1)) if identifiers else 0.0
        symbol_module_score = 0.0
        if symbol_name and symbol_name in identifiers:
            symbol_module_score = max(symbol_module_score, 1.0)
        if module_name and module_name in identifiers:
            symbol_module_score = max(symbol_module_score, 0.8)
        if symbol_name and symbol_name in question_terms:
            symbol_module_score = max(symbol_module_score, 0.6)
        if module_name and module_name in question_terms:
            symbol_module_score = max(symbol_module_score, 0.5)

        action_score = action_statement_score(chunk_text)
        comment_penalty = comment_density_penalty(chunk_text)
        lexical_file_boost = min(1.0, max(0.0, float(metadata.get("_lexical_file_score", 0.0) or 0.0)))

        hybrid = (semantic_weight * semantic) + (lexical_mix_weight * ((0.55 * overlap) + (0.45 * exact_score)))
        hybrid += (0.08 * symbol_module_score) + (0.06 * action_score) + (0.06 * lexical_file_boost)
        hybrid -= 0.08 * comment_penalty

        focus_hits = focus_term_match_count(focus_terms, source)
        if focus_terms:
            if focus_hits == 0:
                hybrid *= 0.70
            else:
                hybrid = min(1.0, hybrid + (0.10 * (focus_hits / len(focus_terms))))
        hybrid = max(0.0, min(1.0, hybrid))

        rows.append(
            {
                "metadata": metadata,
                "semantic_score": semantic,
                "token_overlap": overlap,
                "exact_hits": exact_hits,
                "exact_score": exact_score,
                "symbol_module_score": symbol_module_score,
                "action_score": action_score,
                "comment_penalty": comment_penalty,
                "lexical_file_boost": lexical_file_boost,
                "hybrid_score": hybrid,
                "ordinal": ordinal,
            }
        )

    semantic_sorted = sorted(rows, key=lambda item: item["semantic_score"], reverse=True)
    for idx, row in enumerate(semantic_sorted, start=1):
        row["semantic_rank"] = idx

    reranked = sorted(
        rows,
        key=lambda item: (item["hybrid_score"], item["semantic_score"], item["exact_hits"]),
        reverse=True,
    )

    deduped: list[tuple[dict, float]] = []
    debug_candidates: list[dict[str, Any]] = []
    seen_ranges: set[tuple[str, int, int]] = set()
    for rank, row in enumerate(reranked, start=1):
        metadata = row["metadata"]
        file_path = str(metadata.get("file_path", "unknown"))
        line_start = safe_int(metadata.get("line_start")) or safe_int(metadata.get("start_line")) or 1
        line_end = safe_int(metadata.get("line_end")) or safe_int(metadata.get("end_line")) or line_start
        key = (file_path, line_start, line_end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        if len(deduped) < max(top_k, 1):
            deduped.append((metadata, float(row["hybrid_score"])))
        if len(debug_candidates) < max(DEBUG_CANDIDATE_LIMIT, max(top_k, 1)):
            snippet = " ".join(metadata_chunk_text(metadata).split())[:260]
            debug_candidates.append(
                {
                    "file_path": file_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "query_used": str(metadata.get("query_used", "")),
                    "section_name": str(metadata.get("section_name", "")),
                    "semantic_rank": int(row.get("semantic_rank", rank)),
                    "rerank_rank": rank,
                    "semantic_score": round(float(row["semantic_score"]), 4),
                    "hybrid_score": round(float(row["hybrid_score"]), 4),
                    "token_overlap": round(float(row["token_overlap"]), 4),
                    "exact_identifier_hits": int(row["exact_hits"]),
                    "symbol_module_score": round(float(row["symbol_module_score"]), 4),
                    "action_score": round(float(row["action_score"]), 4),
                    "comment_penalty": round(float(row["comment_penalty"]), 4),
                    "lexical_file_boost": round(float(row["lexical_file_boost"]), 4),
                    "snippet": snippet,
                }
            )
        if len(deduped) >= max(top_k, 1) and len(debug_candidates) >= max(DEBUG_CANDIDATE_LIMIT, max(top_k, 1)):
            break

    min_score = min(max(settings.retrieval_min_hybrid_score, 0.0), 1.0)
    strong = [item for item in deduped if item[1] >= min_score]
    return (strong or deduped), debug_candidates


def extract_citation(metadata: dict, score: float) -> dict:
    try:
        line_start = int(metadata.get("line_start", metadata.get("start_line", 1)))
    except (TypeError, ValueError):
        line_start = 1
    try:
        line_end = int(metadata.get("line_end", metadata.get("end_line", line_start)))
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


def repo_root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def read_repo_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return path.read_text(encoding="utf-8", errors="ignore")


@lru_cache(maxsize=4096)
def repo_file_lines(file_path: str) -> tuple[str, ...]:
    rel = str(file_path or "").replace("\\", "/").lstrip("/")
    if not rel:
        return tuple()
    root = repo_root_path()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        return tuple()
    if not target.is_file():
        return tuple()
    return tuple(read_repo_text(target).splitlines())


@lru_cache(maxsize=1)
def repo_symbol_index() -> dict[str, Any]:
    root = repo_root_path()
    files = discover_fortran_files(repo_root=root, extensions=set(FORTRAN_EXTENSIONS))
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files_by_basename: dict[str, list[str]] = defaultdict(list)

    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        files_by_basename[path.name.lower()].append(rel)
        lines = read_repo_text(path).splitlines()
        starts: list[tuple[int, str, str, str]] = []
        for lineno, line in enumerate(lines, start=1):
            match = FORTRAN_DEF_PATTERN.match(line)
            if not match:
                continue
            starts.append((lineno, match.group(1).lower(), match.group(2).lower(), line.strip()))

        for idx, (line_start, kind, symbol, signature) in enumerate(starts):
            line_end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
            definition = {
                "symbol": symbol,
                "kind": kind,
                "file_path": rel,
                "line_start": line_start,
                "line_end": line_end,
                "signature": signature,
            }
            by_symbol[symbol].append(definition)
            by_file[rel].append(definition)

    for definitions in by_symbol.values():
        definitions.sort(key=lambda item: (item["file_path"], item["line_start"]))
    for definitions in by_file.values():
        definitions.sort(key=lambda item: item["line_start"])
    for file_paths in files_by_basename.values():
        file_paths.sort()

    return {
        "by_symbol": dict(by_symbol),
        "by_file": dict(by_file),
        "files_by_basename": dict(files_by_basename),
    }


def resolve_symbol_definition(symbol: str) -> dict[str, Any] | None:
    lookup = symbol.strip().lower()
    if not lookup:
        return None
    definitions = (repo_symbol_index().get("by_symbol", {}) or {}).get(lookup, [])
    return definitions[0] if definitions else None


def resolve_include_definition(include_target: str) -> dict[str, Any] | None:
    normalized = include_target.strip().lower().split("/")[-1]
    if not normalized:
        return None
    files = (repo_symbol_index().get("files_by_basename", {}) or {}).get(normalized, [])
    if not files:
        return None
    return {"file_path": files[0], "line_start": 1, "line_end": 1, "kind": "include"}


def find_enclosing_definition(file_path: str, line_number: int) -> dict[str, Any] | None:
    definitions = (repo_symbol_index().get("by_file", {}) or {}).get(file_path, [])
    for definition in definitions:
        if definition["line_start"] <= line_number <= definition["line_end"]:
            return definition
    return None


def file_snippet(file_path: str, line_start: int, line_end: int, pad_after: int = 0) -> str:
    lines = repo_file_lines(file_path)
    if not lines:
        return ""
    start = max(1, line_start)
    end = max(start, min(len(lines), line_end + max(pad_after, 0)))
    return "\n".join(lines[start - 1 : end]).strip()


def citation_key(citation: Citation) -> tuple[str, int, int]:
    return citation.file_path, citation.line_start, citation.line_end


def dedupe_citation_pairs(citations: list[Citation], chunks: list[str], limit: int) -> tuple[list[Citation], list[str]]:
    deduped_citations: list[Citation] = []
    deduped_chunks: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for citation, chunk in zip(citations, chunks, strict=False):
        key = citation_key(citation)
        if key in seen:
            continue
        seen.add(key)
        deduped_citations.append(citation)
        deduped_chunks.append(chunk)
        if len(deduped_citations) >= limit:
            break
    return deduped_citations, deduped_chunks


def config_priority_bonus(citation: Citation) -> float:
    file_path = citation.file_path.lower()
    snippet = (citation.snippet or "").lower()
    bonus = 0.0
    if any(token in file_path for token in CONFIG_DIR_HINTS):
        bonus += 0.45
    if any(token in snippet for token in ("config", "parameter", "input", "option", "args", "argv", "setting")):
        bonus += 0.20
    if "open(" in snippet or "read(" in snippet or "inquire(" in snippet:
        bonus += 0.05
    return min(0.75, bonus)


def apply_config_intent_priority(
    question: str,
    citations: list[Citation],
    chunks: list[str],
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    if not is_config_query(question) or not citations:
        return citations, chunks, {"intent": "default", "boosted_hits": 0}

    ranked: list[tuple[float, Citation, str]] = []
    boosted_hits = 0
    for citation, chunk in zip(citations, chunks, strict=False):
        bonus = config_priority_bonus(citation)
        if bonus >= 0.2:
            boosted_hits += 1
        ranked.append((min(1.0, float(citation.score) + bonus), citation, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if boosted_hits < 2:
        return citations, chunks, {"intent": "config", "boosted_hits": boosted_hits}

    reranked_citations = [item[1] for item in ranked]
    reranked_chunks = [item[2] for item in ranked]
    return reranked_citations, reranked_chunks, {"intent": "config", "boosted_hits": boosted_hits}


def aggregate_file_results(citations: list[Citation]) -> list[FileResult]:
    grouped: dict[str, dict[str, Any]] = {}
    for citation in citations:
        path = citation.file_path
        if not path:
            continue
        row = grouped.setdefault(
            path,
            {"match_count": 0, "max_score": 0.0, "source_types": set()},
        )
        row["match_count"] += 1
        row["max_score"] = max(float(row["max_score"]), float(citation.score))
        row["source_types"].add(citation.source_type)

    ranked = sorted(
        grouped.items(),
        key=lambda item: (item[1]["match_count"], item[1]["max_score"], item[0]),
        reverse=True,
    )
    return [
        FileResult(
            file_path=path,
            match_count=int(payload["match_count"]),
            max_score=round(float(payload["max_score"]), 4),
            source_types=sorted(str(source) for source in payload["source_types"]),
        )
        for path, payload in ranked
    ]


def exhaustive_file_summary(question: str, file_results: list[FileResult], limit: int = 20) -> str:
    if not file_results:
        return f"No files matched for exhaustive scan: {question}"
    top = file_results[: max(limit, 1)]
    lines = [f"File List ({len(file_results)} files) for: {question}"]
    for idx, row in enumerate(top, start=1):
        score = int(max(0.0, min(1.0, row.max_score)) * 100)
        lines.append(f"{idx}. {row.file_path}  ({row.match_count} hits, max score {score}%)")
    return "\n".join(lines)


def dependency_summary(question: str, edges: list[DependencyEdge], resolved_edges: int) -> str:
    if not edges:
        return (
            "Dependency Graph\n"
            "No caller->target edges were resolved from the retrieved evidence. "
            "Try a narrower routine name and rerun dependencies mode."
        )
    lines = [
        f"Dependency Graph ({resolved_edges}/{len(edges)} resolved) for: {question}",
    ]
    for idx, edge in enumerate(edges[:20], start=1):
        if edge.resolved and edge.target_path and edge.target_line:
            lines.append(
                f"{idx}. {edge.caller} --{edge.edge_type}--> {edge.target} "
                f"(defined at {edge.target_path}:{edge.target_line})"
            )
        else:
            lines.append(f"{idx}. {edge.caller} --{edge.edge_type}--> {edge.target} (definition unresolved)")
    return "\n".join(lines)


def pattern_summary(question: str, examples: list[PatternExample]) -> str:
    if not examples:
        return (
            "Pattern Examples\n"
            "No loop or IO patterns were extracted from the retrieved evidence. "
            "Try targeting a specific routine or workflow stage."
        )
    lines = [f"Pattern Examples ({len(examples)}) for: {question}"]
    for idx, example in enumerate(examples[:6], start=1):
        lines.append(
            f"{idx}. {example.pattern_type}: {example.summary} "
            f"({example.file_path}:{example.line_start}-{example.line_end})"
        )
    return "\n".join(lines)


def dependency_followups(edges: list[DependencyEdge]) -> list[str]:
    followups: list[str] = []
    seen_targets: set[str] = set()
    for edge in edges:
        target = edge.target.strip()
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)
        followups.append(f"Find definition of {target}")
        followups.append(f"Show all callers of {target}")
        if len(followups) >= 6:
            break
    if not followups:
        followups = ["Find definition of hazpoint", "Show all callers of hazpoint"]
    return followups[:6]


def pattern_followups(examples: list[PatternExample]) -> list[str]:
    followups: list[str] = []
    seen_files: set[str] = set()
    for example in examples:
        file_path = example.file_path
        if file_path in seen_files:
            continue
        seen_files.add(file_path)
        followups.append(f"Show more loops in {file_path}")
        if len(followups) >= 4:
            break
    if not followups:
        followups = ["Show loops over seismic sources", "Show IO blocks that read input files"]
    return followups[:6]


def search_followups(question: str, file_results: list[FileResult]) -> list[str]:
    followups: list[str] = []
    if file_results:
        head = file_results[0].file_path
        followups.append(f"Open {head}")
        followups.append(f"Show snippets from {head}")
    if is_config_query(question):
        followups.append("Search conf/ and scripts for configuration parsing")
    if "ground motion" in question.lower() or "gmpe" in question.lower():
        followups.append("Find definition of GMPE selection logic")
    return dedupe_preserve_order(followups)[:6]


def extract_dependency_graph(
    citations: list[Citation],
    chunks: list[str],
    max_edges: int = 30,
) -> tuple[list[DependencyEdge], list[Citation], list[str], dict[str, Any]]:
    edge_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for citation, chunk in zip(citations, chunks, strict=False):
        lines = chunk.splitlines()
        if not lines:
            continue
        caller_def = find_enclosing_definition(citation.file_path, citation.line_start)
        caller = str((caller_def or {}).get("symbol", "")).strip().lower() or Path(citation.file_path).stem.lower()

        for offset, line in enumerate(lines):
            abs_line = citation.line_start + offset
            stripped = line.strip()
            if not stripped:
                continue

            for match in FORTRAN_CALL_PATTERN.finditer(line):
                target = match.group(1).lower()
                key = ("CALL", caller, target, citation.file_path)
                row = edge_map.setdefault(
                    key,
                    {
                        "edge_type": "CALL",
                        "caller": caller,
                        "caller_path": citation.file_path,
                        "caller_line": abs_line,
                        "target": target,
                        "score": float(citation.score),
                        "count": 0,
                    },
                )
                row["count"] += 1
                row["score"] = max(float(row["score"]), float(citation.score))
                row["caller_line"] = min(int(row["caller_line"]), abs_line)

            use_match = FORTRAN_USE_PATTERN.match(line)
            if use_match:
                target = use_match.group(1).lower()
                key = ("USE", caller, target, citation.file_path)
                row = edge_map.setdefault(
                    key,
                    {
                        "edge_type": "USE",
                        "caller": caller,
                        "caller_path": citation.file_path,
                        "caller_line": abs_line,
                        "target": target,
                        "score": float(citation.score),
                        "count": 0,
                    },
                )
                row["count"] += 1
                row["score"] = max(float(row["score"]), float(citation.score))
                row["caller_line"] = min(int(row["caller_line"]), abs_line)

            include_match = FORTRAN_INCLUDE_PATTERN.match(line)
            if include_match:
                target = include_match.group(1).lower()
                key = ("INCLUDE", caller, target, citation.file_path)
                row = edge_map.setdefault(
                    key,
                    {
                        "edge_type": "INCLUDE",
                        "caller": caller,
                        "caller_path": citation.file_path,
                        "caller_line": abs_line,
                        "target": target,
                        "score": float(citation.score),
                        "count": 0,
                    },
                )
                row["count"] += 1
                row["score"] = max(float(row["score"]), float(citation.score))
                row["caller_line"] = min(int(row["caller_line"]), abs_line)

            for match in FORTRAN_COMMON_PATTERN.finditer(line):
                target = match.group(1).lower()
                key = ("COMMON", caller, target, citation.file_path)
                row = edge_map.setdefault(
                    key,
                    {
                        "edge_type": "COMMON",
                        "caller": caller,
                        "caller_path": citation.file_path,
                        "caller_line": abs_line,
                        "target": target,
                        "score": float(citation.score),
                        "count": 0,
                    },
                )
                row["count"] += 1
                row["score"] = max(float(row["score"]), float(citation.score))
                row["caller_line"] = min(int(row["caller_line"]), abs_line)

    ranked = sorted(
        edge_map.values(),
        key=lambda row: (row["count"], row["score"], row["edge_type"], row["caller"]),
        reverse=True,
    )

    selected: list[DependencyEdge] = []
    citations_out: list[Citation] = []
    chunks_out: list[str] = []
    seen_citations: set[tuple[str, int, int]] = set()
    resolved_edges = 0

    for row in ranked[: max(max_edges, 1)]:
        target_definition: dict[str, Any] | None = None
        if row["edge_type"] in {"CALL", "USE"}:
            target_definition = resolve_symbol_definition(row["target"])
        elif row["edge_type"] == "INCLUDE":
            target_definition = resolve_include_definition(row["target"])

        resolved = bool(target_definition)
        if resolved:
            resolved_edges += 1

        selected.append(
            DependencyEdge(
                edge_type=str(row["edge_type"]),
                caller=str(row["caller"]),
                caller_path=str(row["caller_path"]),
                caller_line=int(row["caller_line"]),
                target=str(row["target"]),
                resolved=resolved,
                target_kind=(target_definition or {}).get("kind"),
                target_path=(target_definition or {}).get("file_path"),
                target_line=safe_int((target_definition or {}).get("line_start")),
            )
        )

        caller_citation = Citation(
            file_path=str(row["caller_path"]),
            line_start=int(row["caller_line"]),
            line_end=int(row["caller_line"]),
            score=round(float(row["score"]), 4),
            source_type="repo",
            snippet=file_snippet(str(row["caller_path"]), int(row["caller_line"]), int(row["caller_line"]), pad_after=5)[:550] or None,
        )
        caller_key = citation_key(caller_citation)
        if caller_key not in seen_citations:
            seen_citations.add(caller_key)
            citations_out.append(caller_citation)
            chunks_out.append(caller_citation.snippet or "")

        if target_definition:
            target_start = safe_int(target_definition.get("line_start")) or 1
            target_end = safe_int(target_definition.get("line_end")) or target_start
            snippet_end = min(target_end, target_start + 24)
            target_citation = Citation(
                file_path=str(target_definition.get("file_path", "unknown")),
                line_start=target_start,
                line_end=snippet_end,
                score=round(max(0.0, float(row["score"]) - 0.03), 4),
                source_type="repo",
                snippet=file_snippet(str(target_definition.get("file_path", "unknown")), target_start, snippet_end, pad_after=0)[:550]
                or None,
            )
            target_key = citation_key(target_citation)
            if target_key not in seen_citations:
                seen_citations.add(target_key)
                citations_out.append(target_citation)
                chunks_out.append(target_citation.snippet or "")

    followups = dependency_followups(selected)
    return selected, citations_out, chunks_out, {"resolved_edges": resolved_edges, "total_edges": len(selected), "followups": followups}


def extract_pattern_examples(
    citations: list[Citation],
    chunks: list[str],
    max_examples: int = 6,
) -> tuple[list[PatternExample], list[Citation], list[str], dict[str, Any]]:
    candidates: list[tuple[int, PatternExample, Citation, str]] = []

    for citation, chunk in zip(citations, chunks, strict=False):
        lines = chunk.splitlines()
        if not lines:
            continue

        for idx, line in enumerate(lines):
            if FORTRAN_LOOP_START_PATTERN.match(line):
                end_idx = idx
                for probe in range(idx + 1, min(idx + 26, len(lines))):
                    if FORTRAN_LOOP_END_PATTERN.match(lines[probe]):
                        end_idx = probe
                        break
                if end_idx == idx:
                    end_idx = min(idx + 8, len(lines) - 1)
                block = "\n".join(lines[idx : end_idx + 1]).strip()
                if not block:
                    continue

                lower = block.lower()
                pattern_type = "Loop Block"
                summary = "General DO-loop iteration."
                priority = 2
                if any(token in lower for token in ("isrc", "source", "nsrc", "imag", "idist", "mag", "distance")):
                    pattern_type = "Source Iteration Loop"
                    summary = "Loop iterates over seismic source or distance/magnitude dimensions."
                    priority = 0
                elif FORTRAN_IO_PATTERN.search(lower):
                    pattern_type = "I/O Loop"
                    summary = "Loop combines iteration with file I/O operations."
                    priority = 1

                abs_start = citation.line_start + idx
                abs_end = citation.line_start + end_idx
                example = PatternExample(
                    pattern_type=pattern_type,
                    summary=summary,
                    file_path=citation.file_path,
                    line_start=abs_start,
                    line_end=abs_end,
                    snippet=block[:550],
                )
                example_citation = Citation(
                    file_path=citation.file_path,
                    line_start=abs_start,
                    line_end=abs_end,
                    score=citation.score,
                    source_type=citation.source_type,
                    file_sha=citation.file_sha,
                    snippet=block[:550],
                )
                candidates.append((priority, example, example_citation, block))

            if FORTRAN_IO_PATTERN.search(line):
                start_idx = max(0, idx - 2)
                end_idx = min(len(lines) - 1, idx + 2)
                block = "\n".join(lines[start_idx : end_idx + 1]).strip()
                if not block:
                    continue
                abs_start = citation.line_start + start_idx
                abs_end = citation.line_start + end_idx
                example = PatternExample(
                    pattern_type="I/O Block",
                    summary="File unit operation (OPEN/READ/INQUIRE/WRITE).",
                    file_path=citation.file_path,
                    line_start=abs_start,
                    line_end=abs_end,
                    snippet=block[:550],
                )
                example_citation = Citation(
                    file_path=citation.file_path,
                    line_start=abs_start,
                    line_end=abs_end,
                    score=max(0.0, citation.score - 0.01),
                    source_type=citation.source_type,
                    file_sha=citation.file_sha,
                    snippet=block[:550],
                )
                candidates.append((3, example, example_citation, block))

    candidates.sort(key=lambda item: (item[0], -float(item[2].score)))
    selected_examples: list[PatternExample] = []
    selected_citations: list[Citation] = []
    selected_chunks: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for _, example, example_citation, block in candidates:
        key = (example.file_path, example.line_start, example.line_end)
        if key in seen:
            continue
        seen.add(key)
        selected_examples.append(example)
        selected_citations.append(example_citation)
        selected_chunks.append(block)
        if len(selected_examples) >= max(max_examples, 1):
            break

    followups = pattern_followups(selected_examples)
    return selected_examples, selected_citations, selected_chunks, {"pattern_examples": len(selected_examples), "followups": followups}


def apply_mode_analysis(
    mode: str,
    question: str,
    citations: list[Citation],
    chunks: list[str],
) -> dict[str, Any]:
    normalized_mode = normalize_mode(mode)
    result: dict[str, Any] = {
        "result_type": "Ranked Chunks",
        "summary": None,
        "follow_ups": [],
        "file_results": [],
        "graph_edges": [],
        "pattern_examples": [],
        "citations": citations,
        "chunks": chunks,
        "mode_metrics": {},
    }

    if normalized_mode == "dependencies":
        edges, dep_citations, dep_chunks, metrics = extract_dependency_graph(citations, chunks, max_edges=30)
        if dep_citations:
            dep_citations, dep_chunks = dedupe_citation_pairs(dep_citations, dep_chunks, limit=24)
            result["citations"] = dep_citations
            result["chunks"] = dep_chunks
        result["result_type"] = "Dependency Graph"
        result["summary"] = dependency_summary(question, edges, safe_int(metrics.get("resolved_edges")) or 0)
        result["follow_ups"] = list(metrics.get("followups", []))
        result["graph_edges"] = edges
        result["mode_metrics"] = metrics
        return result

    if normalized_mode == "patterns":
        examples, pattern_citations, pattern_chunks, metrics = extract_pattern_examples(citations, chunks, max_examples=6)
        if pattern_citations:
            pattern_citations, pattern_chunks = dedupe_citation_pairs(pattern_citations, pattern_chunks, limit=18)
            result["citations"] = pattern_citations
            result["chunks"] = pattern_chunks
        result["result_type"] = "Pattern Examples"
        result["summary"] = pattern_summary(question, examples)
        result["follow_ups"] = list(metrics.get("followups", []))
        result["pattern_examples"] = examples
        result["mode_metrics"] = metrics
        return result

    if normalized_mode == "search" and is_exhaustive_file_query(question):
        file_results = aggregate_file_results(citations)
        config_hits = sum(
            1 for row in file_results if any(token in row.file_path.lower() for token in ("conf/", "etc/", "scripts/", "makefile"))
        )
        result["result_type"] = "File List"
        result["summary"] = exhaustive_file_summary(question, file_results)
        follow_ups = search_followups(question, file_results)
        if is_config_query(question) and config_hits == 0:
            follow_ups.insert(0, "Clarify: do you mean conf/etc/scripts config, or GMPE data-table reads?")
        result["follow_ups"] = dedupe_preserve_order(follow_ups)[:6]
        result["file_results"] = file_results[:60]
        result["mode_metrics"] = {"unique_files": len(file_results)}
        return result

    if normalized_mode == "search":
        result["result_type"] = "Ranked Chunks"
        file_results = aggregate_file_results(citations)
        config_hits = sum(
            1 for row in file_results if any(token in row.file_path.lower() for token in ("conf/", "etc/", "scripts/", "makefile"))
        )
        follow_ups = search_followups(question, file_results)
        if is_config_query(question) and config_hits == 0:
            follow_ups.insert(0, "Clarify: do you mean conf/etc/scripts config, or GMPE data-table reads?")
        result["follow_ups"] = dedupe_preserve_order(follow_ups)[:6]
        result["mode_metrics"] = {"unique_files": len({citation.file_path for citation in citations})}

    return result


def query_namespaces() -> list[str]:
    namespaces = [settings.pinecone_namespace]
    fallback = (settings.pinecone_fallback_namespace or "").strip()
    if fallback and fallback not in namespaces:
        namespaces.append(fallback)
    return namespaces


def query_filter_cache_key(filter_payload: dict[str, Any] | None) -> str:
    if not filter_payload:
        return ""
    try:
        return json.dumps(filter_payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(filter_payload)


def pinecone_query_cache_key(query_text: str, namespace: str, top_k: int, filter_payload: dict[str, Any] | None) -> str:
    material = "\n".join(
        [
            settings.pinecone_index_name,
            namespace,
            str(top_k),
            query_text.strip().lower(),
            query_filter_cache_key(filter_payload),
        ]
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def query_cache_get(key: str) -> list[dict[str, Any]] | None:
    ttl = max(float(settings.retrieval_query_cache_ttl_seconds), 0.0)
    if ttl <= 0:
        return None
    now = time.time()
    with query_result_cache_lock:
        row = query_result_cache.get(key)
        if not isinstance(row, dict):
            return None
        expires_at = float(row.get("expires_at", 0.0) or 0.0)
        if expires_at < now:
            query_result_cache.pop(key, None)
            return None
        payload = row.get("payload")
        if not isinstance(payload, list):
            return None
        return payload


def query_cache_put(key: str, payload: list[dict[str, Any]]) -> None:
    ttl = max(float(settings.retrieval_query_cache_ttl_seconds), 0.0)
    if ttl <= 0:
        return
    now = time.time()
    with query_result_cache_lock:
        query_result_cache[key] = {"expires_at": now + ttl, "payload": payload}
        max_entries = max(int(settings.retrieval_query_cache_max_entries), 64)
        if len(query_result_cache) > max_entries:
            stale = sorted(query_result_cache.items(), key=lambda item: float((item[1] or {}).get("expires_at", 0.0)))
            for stale_key, _ in stale[: max(len(stale) - max_entries, 1)]:
                query_result_cache.pop(stale_key, None)


def normalize_cached_matches(matches: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for match in matches:
        normalized.append(
            {
                "score": match_score(match),
                "metadata": normalize_metadata(match),
            }
        )
    return normalized


def query_index_with_cache(
    index: Any,
    query_text: str,
    question_vector: list[float],
    candidate_top_k: int,
    namespace: str,
    pinecone_filter: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], bool]:
    key = pinecone_query_cache_key(query_text, namespace, candidate_top_k, pinecone_filter)
    cached = query_cache_get(key)
    if cached is not None:
        return cached, True

    query_kwargs: dict[str, Any] = {
        "vector": question_vector,
        "top_k": candidate_top_k,
        "include_metadata": True,
        "namespace": namespace,
    }
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter
    results = call_with_retries("pinecone query", lambda: index.query(**query_kwargs))
    query_matches = normalize_cached_matches(normalize_matches(results))
    query_cache_put(key, query_matches)
    return query_matches, False


def build_pinecone_filter(
    language: str | None = None,
    source_type: str | None = None,
    file_paths: list[str] | None = None,
    repo: str | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if language:
        clauses.append({"language": {"$eq": language}})
    if source_type:
        clauses.append({"source_type": {"$eq": source_type}})
    normalized_paths = [path for path in (normalize_file_path(item) for item in (file_paths or [])) if path]
    if normalized_paths:
        clauses.append({"file_path": {"$in": dedupe_preserve_order(normalized_paths)}})
    if repo:
        clauses.append({"repo": {"$eq": repo}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def metadata_matches_filters(
    metadata: dict[str, Any],
    path_prefix: str | None = None,
    language: str | None = None,
    source_type: str | None = None,
    candidate_files: list[str] | None = None,
    repo_name: str | None = None,
) -> bool:
    file_path = str(metadata.get("file_path", "")).replace("\\", "/").lstrip("/")
    meta_language = str(metadata.get("language", "")).strip().lower()
    meta_source_type = str(metadata.get("source_type", "")).strip().lower()
    meta_repo = str(metadata.get("repo", "")).strip()

    if path_prefix:
        normalized_prefix = path_prefix.replace("\\", "/").lstrip("/").lower()
        if not file_path.lower().startswith(normalized_prefix):
            return False
    if language and meta_language != language:
        return False
    if source_type and meta_source_type != source_type:
        return False
    if candidate_files:
        normalized_candidates = {item.lower() for item in candidate_files if item}
        if file_path.lower() not in normalized_candidates:
            return False
    if repo_name and meta_repo and meta_repo != repo_name:
        return False
    return True


def retrieve_citations_and_chunks(
    question: str,
    top_k: int,
    retrieval_queries: list[str] | None = None,
    namespaces: list[str] | None = None,
    default_source_type: str = "repo",
    path_prefix: str | None = None,
    language: str | None = None,
    source_type_filter: str | None = None,
    candidate_files: list[str] | None = None,
    repo_name: str | None = None,
    lexical_file_scores: dict[str, float] | None = None,
    candidate_top_k_override: int | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    queries = retrieval_queries or [question]
    target_namespaces = namespaces or query_namespaces()
    if candidate_top_k_override is not None:
        candidate_top_k = max(int(candidate_top_k_override), 1)
    else:
        candidate_top_k = min(
            max(top_k * max(settings.retrieval_candidate_multiplier, 1), max(top_k, 20)),
            max(settings.retrieval_max_candidates, max(top_k, 20)),
        )
    index = get_pinecone_index()
    index_dim: int | None = None
    if settings.enforce_embedding_dimension:
        try:
            index_dim = cached_index_dimension(settings.pinecone_index_name)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to resolve Pinecone index dimension: {startup_smoke_error(exc)}",
            ) from exc
    normalized_candidate_files = dedupe_preserve_order(
        [path for path in (normalize_file_path(item) for item in (candidate_files or [])) if path]
    )
    lexical_scores = {
        str(normalize_file_path(path) or ""): max(0.0, min(1.0, float(score)))
        for path, score in (lexical_file_scores or {}).items()
        if normalize_file_path(path)
    }
    pinecone_filter = build_pinecone_filter(
        language=language,
        source_type=source_type_filter,
        file_paths=normalized_candidate_files,
        repo=repo_name,
    )

    matches: list = []
    subquery_counts: list[dict[str, Any]] = []
    embed_ms = 0.0
    pinecone_ms = 0.0
    cache_hits = 0
    cache_misses = 0
    for query_text in queries:
        embed_started = time.perf_counter()
        question_vector = embed_question(query_text)
        embed_ms += (time.perf_counter() - embed_started) * 1000.0
        if not question_vector:
            subquery_counts.append({"query": query_text, "matches": 0})
            continue
        ensure_embedding_dimension_matches_index(len(question_vector), index_dim=index_dim)

        query_matches: list = []
        for namespace in target_namespaces:
            pinecone_started = time.perf_counter()
            query_matches, from_cache = query_index_with_cache(
                index=index,
                query_text=query_text,
                question_vector=question_vector,
                candidate_top_k=candidate_top_k,
                namespace=namespace,
                pinecone_filter=pinecone_filter,
            )
            pinecone_ms += (time.perf_counter() - pinecone_started) * 1000.0
            if from_cache:
                cache_hits += 1
            else:
                cache_misses += 1
            if query_matches:
                break

        accepted_count = 0
        for match in query_matches:
            metadata = normalize_metadata(match).copy()
            if str(metadata.get("_record_type", "")) == "attachment_file":
                continue
            if not metadata_chunk_text(metadata).strip():
                continue
            metadata["query_used"] = query_text
            metadata.setdefault("source_type", default_source_type)
            file_path = normalize_file_path(metadata.get("file_path"))
            if file_path and file_path in lexical_scores:
                metadata["_lexical_file_score"] = lexical_scores[file_path]
            if not metadata_matches_filters(
                metadata,
                path_prefix=path_prefix,
                language=language,
                source_type=source_type_filter,
                candidate_files=normalized_candidate_files,
                repo_name=repo_name,
            ):
                continue
            matches.append({"score": match_score(match), "metadata": metadata})
            accepted_count += 1
        subquery_counts.append({"query": query_text, "matches": accepted_count})

    if not matches:
        return (
            [],
            [],
            {
                "candidates": [],
                "subqueries": subquery_counts,
                "timings_ms": {
                    "embed": round(embed_ms, 2),
                    "pinecone_query": round(pinecone_ms, 2),
                    "rerank": 0.0,
                },
                "cache": {"hits": cache_hits, "misses": cache_misses},
            },
        )

    rerank_started = time.perf_counter()
    reranked, debug_candidates = rerank_matches(question=question, matches=matches, top_k=top_k)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    debug = {
        "candidates": debug_candidates,
        "subqueries": subquery_counts,
        "timings_ms": {
            "embed": round(embed_ms, 2),
            "pinecone_query": round(pinecone_ms, 2),
            "rerank": round(rerank_ms, 2),
        },
        "cache": {"hits": cache_hits, "misses": cache_misses},
    }
    return citations, chunks, debug


def retrieve_uploaded_citations_and_chunks(
    question: str,
    top_k: int,
    uploaded_files: list[dict[str, Any]],
    retrieval_queries: list[str] | None = None,
    path_prefix: str | None = None,
    language: str | None = None,
    source_type_filter: str | None = None,
    candidate_files: list[str] | None = None,
    repo_name: str | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    if not uploaded_files:
        return [], [], {"candidates": [], "subqueries": []}

    metadata_chunks, _ = build_attachment_chunks(uploaded_files, source_type="temp-upload")
    if not metadata_chunks:
        return [], [], {"candidates": [], "subqueries": []}

    normalized_candidate_files = dedupe_preserve_order(
        [path for path in (normalize_file_path(item) for item in (candidate_files or [])) if path]
    )
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
                if not metadata_matches_filters(
                    scoped,
                    path_prefix=path_prefix,
                    language=language,
                    source_type=source_type_filter,
                    candidate_files=normalized_candidate_files,
                    repo_name=repo_name,
                ):
                    continue
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


def should_include_header_context(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("module", "imports", "use ", "dependency", "entry point", "entrypoint"))


def expand_repo_citation_context(question: str, citation: Citation, chunk: str) -> tuple[Citation, str]:
    file_path = str(citation.file_path or "")
    if not file_path or citation.source_type != "repo":
        return citation, chunk

    lines = repo_file_lines(file_path)
    if not lines:
        return citation, chunk

    line_start = max(1, int(citation.line_start))
    line_end = max(line_start, int(citation.line_end))
    neighbor = max(int(settings.retrieval_context_neighbor_lines), 0)
    parent_max = max(int(settings.retrieval_context_parent_max_lines), 1)
    header_lines = max(int(settings.retrieval_context_header_lines), 0)

    parent = find_enclosing_definition(file_path, line_start)
    expanded_start = max(1, line_start - neighbor)
    expanded_end = min(len(lines), line_end + neighbor)
    if parent:
        parent_start = max(1, safe_int(parent.get("line_start")) or expanded_start)
        parent_end = min(len(lines), safe_int(parent.get("line_end")) or expanded_end)
        parent_span = parent_end - parent_start + 1
        if parent_span <= parent_max:
            expanded_start = parent_start
            expanded_end = parent_end
        else:
            expanded_start = max(parent_start, expanded_start)
            expanded_end = min(parent_end, expanded_end)

    snippet_parts: list[str] = []
    if should_include_header_context(question) and header_lines > 0 and expanded_start > 1:
        header_end = min(header_lines, len(lines))
        header_text = "\n".join(lines[:header_end]).strip()
        if header_text:
            snippet_parts.append(header_text)
    body_text = file_snippet(file_path, expanded_start, expanded_end, pad_after=0).strip()
    if body_text:
        snippet_parts.append(body_text)
    expanded_text = "\n\n".join(part for part in snippet_parts if part).strip() or (chunk or "")

    return (
        Citation(
            file_path=file_path,
            line_start=expanded_start,
            line_end=expanded_end,
            score=float(citation.score),
            source_type=citation.source_type,
            file_sha=citation.file_sha,
            snippet=expanded_text[:550] if expanded_text else citation.snippet,
        ),
        expanded_text,
    )


def expand_context_for_citations(
    question: str,
    citations: list[Citation],
    chunks: list[str],
    limit: int,
) -> tuple[list[Citation], list[str]]:
    if not citations or not chunks:
        return citations, chunks
    max_items = max(limit, 1)
    expanded_citations: list[Citation] = []
    expanded_chunks: list[str] = []
    for citation, chunk in zip(citations, chunks, strict=False):
        if citation.source_type != "repo":
            expanded_citations.append(citation)
            expanded_chunks.append(chunk)
        else:
            next_citation, expanded_text = expand_repo_citation_context(question, citation, chunk)
            expanded_citations.append(next_citation)
            expanded_chunks.append(expanded_text)
        if len(expanded_citations) >= max_items:
            break
    return expanded_citations, expanded_chunks


def retrieve_with_optional_uploads(
    question: str,
    top_k: int,
    uploaded_files: list[dict[str, Any]],
    scope: str,
    project_id: str,
    mode: str = "chat",
    path_prefix: str | None = None,
    language: str | None = None,
    source_type_filter: str | None = None,
    candidate_files: list[str] | None = None,
    repo_name: str | None = None,
    disable_internal_lexical: bool = False,
    candidate_top_k_override: int | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    started = time.perf_counter()
    normalized_mode = normalize_mode(mode)
    effective_top_k = max(top_k, 1)
    if normalized_mode == "search" and is_exhaustive_file_query(question):
        effective_top_k = max(effective_top_k * 4, 40)

    include_repo = scope in {"repo", "both"}
    include_uploads = scope in {"uploads", "both"}
    rewrite_started = time.perf_counter()
    rewritten_query, subqueries = rewrite_and_decompose_query(question)
    rewrite_ms = (time.perf_counter() - rewrite_started) * 1000.0
    retrieval_queries = [rewritten_query, *subqueries]
    index_citations: list[Citation] = []
    index_chunks: list[str] = []
    index_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    lexical_index_citations: list[Citation] = []
    lexical_index_chunks: list[str] = []
    lexical_index_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    persistent_upload_citations: list[Citation] = []
    persistent_upload_chunks: list[str] = []
    persistent_upload_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    temp_upload_citations: list[Citation] = []
    temp_upload_chunks: list[str] = []
    temp_upload_debug: dict[str, Any] = {"candidates": [], "subqueries": []}
    lexical_started = time.perf_counter()
    lexical_enabled = include_repo and settings.retrieval_identifier_lexical_enabled and not disable_internal_lexical
    lexical_debug = (
        lexical_candidate_files(question)
        if lexical_enabled
        else {"enabled": False, "identifiers": [], "candidate_files": [], "file_scores": {}, "hits": []}
    )
    lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
    lexical_files = [path for path in (normalize_file_path(item) for item in lexical_debug.get("candidate_files", []) or []) if path]
    lexical_scores = lexical_debug.get("file_scores", {}) if isinstance(lexical_debug, dict) else {}
    provided_candidates = [path for path in (normalize_file_path(item) for item in (candidate_files or [])) if path]
    primary_candidate_files = dedupe_preserve_order(
        [*provided_candidates, *(lexical_files if provided_candidates else [])]
    )

    if include_repo:
        try:
            index_citations, index_chunks, index_debug = retrieve_citations_and_chunks(
                question=rewritten_query,
                top_k=effective_top_k,
                retrieval_queries=retrieval_queries,
                namespaces=query_namespaces(),
                default_source_type="repo",
                path_prefix=path_prefix,
                language=language,
                source_type_filter=source_type_filter,
                candidate_files=primary_candidate_files,
                repo_name=repo_name,
                lexical_file_scores=lexical_scores,
                candidate_top_k_override=candidate_top_k_override,
            )
            # For identifier-heavy queries, run a focused lexical pass and merge with a small bonus.
            if lexical_enabled and lexical_files and not provided_candidates:
                lexical_index_citations, lexical_index_chunks, lexical_index_debug = retrieve_citations_and_chunks(
                    question=rewritten_query,
                    top_k=effective_top_k,
                    retrieval_queries=retrieval_queries,
                    namespaces=query_namespaces(),
                    default_source_type="repo",
                    path_prefix=path_prefix,
                    language=language,
                    source_type_filter=source_type_filter,
                    candidate_files=lexical_files,
                    repo_name=repo_name,
                    lexical_file_scores=lexical_scores,
                    candidate_top_k_override=candidate_top_k_override,
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
                top_k=effective_top_k,
                retrieval_queries=retrieval_queries,
                namespaces=[attachment_ns],
                default_source_type="upload",
                path_prefix=path_prefix,
                language=language,
                source_type_filter=source_type_filter,
                candidate_files=provided_candidates,
                repo_name=repo_name,
                candidate_top_k_override=candidate_top_k_override,
            )
        except HTTPException as exc:
            logger.warning("Attachment retrieval unavailable: %s", exc.detail)
        except Exception as exc:
            logger.warning("Attachment retrieval failed: %s", exc)

        if uploaded_files:
            temp_upload_citations, temp_upload_chunks, temp_upload_debug = retrieve_uploaded_citations_and_chunks(
                question=rewritten_query,
                top_k=effective_top_k,
                uploaded_files=uploaded_files,
                retrieval_queries=retrieval_queries,
                path_prefix=path_prefix,
                language=language,
                source_type_filter=source_type_filter,
                candidate_files=provided_candidates,
                repo_name=repo_name,
            )

    sets_to_merge: list[tuple[list[Citation], list[str], float]] = []
    if include_repo:
        sets_to_merge.append((index_citations, index_chunks, 0.0))
        if lexical_index_citations:
            sets_to_merge.append((lexical_index_citations, lexical_index_chunks, 0.06))
    if include_uploads:
        sets_to_merge.append((persistent_upload_citations, persistent_upload_chunks, UPLOAD_SOURCE_BONUS))
        sets_to_merge.append((temp_upload_citations, temp_upload_chunks, UPLOAD_PRIORITY_BONUS))
    citations, chunks = merge_citation_sets(sets=sets_to_merge, top_k=effective_top_k)
    citations, chunks = dedupe_citation_pairs(citations, chunks, limit=effective_top_k)
    citations, chunks, intent_debug = apply_config_intent_priority(question, citations, chunks)
    focus_guardrail = analyze_focus_term_alignment(question, citations, chunks)
    if (
        settings.retrieval_focus_term_guardrail_enabled
        and focus_guardrail.get("required_terms")
        and not focus_guardrail.get("matched_terms")
    ):
        citations = []
        chunks = []
        focus_guardrail["action"] = "drop_all_without_exact_term_match"
    elif focus_guardrail.get("required_terms"):
        focus_guardrail["action"] = "retain_exact_term_matches_found"

    context_started = time.perf_counter()
    if settings.retrieval_context_expansion_enabled:
        citations, chunks = expand_context_for_citations(question, citations, chunks, limit=effective_top_k)
        citations, chunks = dedupe_citation_pairs(citations, chunks, limit=effective_top_k)
    context_ms = (time.perf_counter() - context_started) * 1000.0

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
        "index_lexical": lexical_index_debug,
        "uploads": upload_debug,
        "lexical_candidates": lexical_debug,
        "feature_flags": {
            "identifier_lexical_enabled": bool(settings.retrieval_identifier_lexical_enabled),
            "deterministic_rerank_enabled": bool(settings.retrieval_deterministic_rerank_enabled),
            "context_expansion_enabled": bool(settings.retrieval_context_expansion_enabled),
        },
        "intent_router": intent_debug,
        "focus_guardrail": focus_guardrail,
        "retrieval_top_k": effective_top_k,
        "timings_ms": {
            "rewrite": round(rewrite_ms, 2),
            "lexical": round(lexical_ms, 2),
            "context_assembly": round(context_ms, 2),
            "total": round((time.perf_counter() - started) * 1000.0, 2),
        },
        "filters": {
            "path_prefix": path_prefix,
            "language": language,
            "source_type": source_type_filter,
            "candidate_files_count": len(provided_candidates),
            "lexical_candidate_files_count": len(lexical_files),
            "repo": repo_name,
        },
    }
    return citations, chunks, debug


def _normalized_candidate_paths(paths: list[str] | None) -> list[str]:
    return dedupe_preserve_order([path for path in (normalize_file_path(item) for item in (paths or [])) if path])


def run_routed_retrieval_plan(
    question: str,
    mode: str,
    top_k: int,
    scope: str,
    project_id: str,
    path_prefix: str | None = None,
    language: str | None = None,
    source_type_filter: str | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    selected_plan = select_retrieval_plan(question=question, mode=mode)
    signals = detect_route_signals(question)
    route_debug = route_debug_template(route=selected_plan, signals=signals)
    route_debug["initial_route"] = selected_plan

    final_top_k = min(max(int(top_k), 1), 5)
    target_repo = default_gitnexus_repo()
    graph_payload: dict[str, Any] = {}
    graph_files: list[str] = []
    keyword_files: list[str] = []
    retrieval_debug: dict[str, Any] = {"subqueries": [], "candidates": []}
    citations: list[Citation] = []
    chunks: list[str] = []

    graph_done = False
    keyword_done = False

    def step_graph() -> None:
        nonlocal graph_payload, graph_files, graph_done
        if graph_done:
            return
        started = time.perf_counter()
        graph_payload = run_gitnexus_graph(question, repo_name=target_repo)
        graph_files = _normalized_candidate_paths(graph_payload.get("candidate_files", []) if isinstance(graph_payload, dict) else [])
        processes = graph_payload.get("processes", []) if isinstance(graph_payload, dict) else []
        entrypoints = graph_payload.get("entrypoints", []) if isinstance(graph_payload, dict) else []
        graph_errors = graph_payload.get("errors", []) if isinstance(graph_payload, dict) else []
        graph_step: dict[str, Any] = {
            "name": "graph",
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
            "candidates": {
                "files": len(graph_files),
                "symbols": len(entrypoints) + (len(processes) if isinstance(processes, list) else 0),
            },
        }
        if isinstance(graph_errors, list):
            condensed_errors = [str(item).strip() for item in graph_errors if str(item).strip()]
            if condensed_errors:
                graph_step["errors"] = condensed_errors[:3]
        route_debug["steps"].append(graph_step)
        graph_done = True

    def step_keyword() -> None:
        nonlocal keyword_files, keyword_done
        if keyword_done:
            return
        started = time.perf_counter()
        lexical_payload = lexical_candidate_files(question)
        keyword_files = _normalized_candidate_paths(
            lexical_payload.get("candidate_files", []) if isinstance(lexical_payload, dict) else []
        )
        hits = lexical_payload.get("hits", []) if isinstance(lexical_payload, dict) else []
        lexical_errors = lexical_payload.get("errors", []) if isinstance(lexical_payload, dict) else []
        keyword_step: dict[str, Any] = {
            "name": "keyword",
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
            "candidates": {"files": len(keyword_files), "symbols": len(hits) if isinstance(hits, list) else 0},
        }
        if isinstance(lexical_errors, list):
            condensed_errors = [str(item).strip() for item in lexical_errors if str(item).strip()]
            if condensed_errors:
                keyword_step["errors"] = condensed_errors[:3]
        route_debug["steps"].append(keyword_step)
        keyword_done = True

    def step_vector(candidate_files: list[str] | None) -> tuple[list[Citation], list[str], dict[str, Any]]:
        started = time.perf_counter()
        local_citations, local_chunks, local_debug = retrieve_with_optional_uploads(
            question=question,
            top_k=final_top_k,
            uploaded_files=[],
            scope=scope,
            project_id=project_id,
            mode="chat",
            path_prefix=path_prefix,
            language=language,
            source_type_filter=source_type_filter,
            candidate_files=candidate_files,
            repo_name=target_repo if candidate_files else None,
            disable_internal_lexical=True,
            candidate_top_k_override=ROUTER_PINECONE_TOP_K,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        index_debug = (local_debug.get("index", {}) if isinstance(local_debug, dict) else {}) or {}
        index_timing = (index_debug.get("timings_ms", {}) if isinstance(index_debug, dict) else {}) or {}
        pinecone_ms = float(index_timing.get("pinecone_query") or 0.0)
        rerank_ms = float(index_timing.get("rerank") or 0.0)

        route_debug["steps"].append(
            {
                "name": "pinecone",
                "ms": round(pinecone_ms if pinecone_ms > 0 else elapsed_ms, 2),
                "top_k": ROUTER_PINECONE_TOP_K,
                "filtered": bool(candidate_files),
            }
        )
        route_debug["steps"].append(
            {
                "name": "rerank",
                "ms": round(max(rerank_ms, 0.0), 2),
                "kept": len(local_citations),
            }
        )
        return local_citations, local_chunks, local_debug

    if selected_plan == PLAN_GRAPH_ONLY:
        step_graph()
        return citations, chunks, retrieval_debug, graph_payload, route_debug, selected_plan

    if selected_plan in {PLAN_GRAPH_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
        step_graph()
    if selected_plan in {PLAN_KEYWORD_ONLY, PLAN_KEYWORD_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
        step_keyword()

    candidate_files: list[str] = []
    if selected_plan == PLAN_GRAPH_PLUS_VECTOR:
        candidate_files = graph_files
    elif selected_plan == PLAN_KEYWORD_PLUS_VECTOR:
        candidate_files = keyword_files
    elif selected_plan == PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR:
        candidate_files = _normalized_candidate_paths([*graph_files, *keyword_files])

    citations, chunks, retrieval_debug = step_vector(candidate_files or None)

    low_conf_reason = low_confidence_reason(citations)
    if not low_conf_reason and selected_plan in {PLAN_GRAPH_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
        graph_errors = graph_payload.get("errors", []) if isinstance(graph_payload, dict) else []
        if isinstance(graph_errors, list) and any(str(item).strip() for item in graph_errors):
            low_conf_reason = "graph_unavailable"
        elif bool(signals.get("structure_intent")) and not graph_files:
            low_conf_reason = "graph_no_candidates"
    next_plan = escalated_plan(current_plan=selected_plan, did_escalate=False) if low_conf_reason else None
    if next_plan:
        route_debug["escalation"] = {"did_escalate": True, "reason": low_conf_reason}
        selected_plan = next_plan
        route_debug["route"] = selected_plan

        if selected_plan == PLAN_KEYWORD_PLUS_VECTOR:
            step_keyword()
            escalation_candidates = keyword_files
        elif selected_plan == PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR:
            step_graph()
            step_keyword()
            escalation_candidates = _normalized_candidate_paths([*graph_files, *keyword_files])
        else:
            escalation_candidates = []

        citations, chunks, retrieval_debug = step_vector(escalation_candidates or None)
    else:
        route_debug["escalation"] = {"did_escalate": False, "reason": None}

    return citations, chunks, retrieval_debug, graph_payload, route_debug, selected_plan


def compute_evidence_strength(
    question: str,
    citations: list[Citation],
    retrieval_debug: dict[str, Any],
    mode: str = "chat",
    mode_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_mode = normalize_mode(mode)
    metrics_payload: dict[str, Any] = mode_metrics or {}

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
                "mode": normalized_mode,
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

    focus_guardrail = retrieval_debug.get("focus_guardrail", {}) if isinstance(retrieval_debug, dict) else {}
    required_focus_terms = focus_guardrail.get("required_terms", []) if isinstance(focus_guardrail, dict) else []
    matched_focus_terms = focus_guardrail.get("matched_terms", []) if isinstance(focus_guardrail, dict) else []
    unmatched_focus_terms = focus_guardrail.get("unmatched_terms", []) if isinstance(focus_guardrail, dict) else []
    if not isinstance(required_focus_terms, list):
        required_focus_terms = []
    if not isinstance(matched_focus_terms, list):
        matched_focus_terms = []
    if not isinstance(unmatched_focus_terms, list):
        unmatched_focus_terms = []
    focus_coverage = (len(matched_focus_terms) / len(required_focus_terms)) if required_focus_terms else 1.0

    normalized_files = min(distinct_files / 4.0, 1.0)
    normalized_symbols = min(symbol_match_count / max(len(citations), 1), 1.0)
    confidence_score = (
        (0.45 * top_score)
        + (0.15 * min(gap * 2.0, 1.0))
        + (0.15 * normalized_files)
        + (0.15 * normalized_symbols)
        + (0.10 * subquery_coverage)
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    if normalized_mode == "dependencies":
        resolved_edges = safe_int(metrics_payload.get("resolved_edges")) or 0
        total_edges = safe_int(metrics_payload.get("total_edges")) or 0
        edge_ratio = (resolved_edges / total_edges) if total_edges > 0 else 0.0
        confidence_score = (
            (0.55 * confidence_score)
            + (0.25 * min(edge_ratio, 1.0))
            + (0.20 * min(total_edges / 12.0, 1.0))
        )
        if resolved_edges < 3:
            confidence_score = min(confidence_score, 0.66)
        metrics_payload["resolved_edges"] = resolved_edges
        metrics_payload["total_edges"] = total_edges
        metrics_payload["resolved_edge_ratio"] = round(edge_ratio, 3)

    elif normalized_mode == "search":
        unique_files = safe_int(metrics_payload.get("unique_files")) or len({c.file_path for c in citations})
        coverage = min(unique_files / 10.0, 1.0)
        confidence_score = (0.70 * confidence_score) + (0.30 * coverage)
        metrics_payload["unique_files"] = unique_files
        if is_exhaustive_file_query(question) and unique_files < 10:
            confidence_score = min(confidence_score, 0.68)

    elif normalized_mode == "patterns":
        pattern_examples = safe_int(metrics_payload.get("pattern_examples")) or 0
        coverage = min(pattern_examples / 6.0, 1.0)
        confidence_score = (0.70 * confidence_score) + (0.30 * coverage)
        metrics_payload["pattern_examples"] = pattern_examples
        if pattern_examples < 3:
            confidence_score = min(confidence_score, 0.63)

    else:  # chat
        if len(citations) < 3:
            confidence_score = min(confidence_score, 0.70)

    if settings.retrieval_focus_term_guardrail_enabled and required_focus_terms:
        absent_cap = max(0.0, min(1.0, float(settings.retrieval_focus_term_absent_cap)))
        partial_cap = max(0.0, min(1.0, float(settings.retrieval_focus_term_partial_coverage_cap)))
        if not matched_focus_terms:
            confidence_score = min(confidence_score, absent_cap)
        elif focus_coverage < 0.5:
            confidence_score = min(confidence_score, partial_cap)

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
    if settings.retrieval_focus_term_guardrail_enabled and required_focus_terms and not matched_focus_terms:
        reason = "Requested identifier/phrase was not found in retrieved evidence."
    elif settings.retrieval_focus_term_guardrail_enabled and required_focus_terms and focus_coverage < 1.0:
        reason = "Evidence only partially covers the requested identifier/phrase."

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
            "focus_term_count": len(required_focus_terms),
            "focus_term_matches": len(matched_focus_terms),
            "focus_term_unmatched": len(unmatched_focus_terms),
            "focus_term_coverage": round(focus_coverage, 3),
            "mode": normalized_mode,
            **metrics_payload,
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


def insufficient_evidence_answer(
    question: str,
    suggestions: dict[str, list[str]],
    missing_terms: list[str] | None = None,
) -> str:
    files = ", ".join(suggestions.get("files", []))
    terms = ", ".join(suggestions.get("terms", []))
    missing = ", ".join((missing_terms or [])[:4])
    missing_line = f"Missing exact terms: {missing}\n" if missing else ""
    return (
        "I don't have enough evidence from the repo to answer confidently.\n"
        f"Question: {question}\n"
        f"{missing_line}"
        f"Next files to inspect: {files}\n"
        f"Next search terms: {terms}"
    )


def build_hybrid_evidence_rows(citations: list[Citation], chunks: list[str], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (citation, chunk) in enumerate(zip(citations, chunks, strict=False), start=1):
        snippet = " ".join((chunk or citation.snippet or "").split())
        summary = snippet[:220].strip()
        if len(snippet) > 220:
            summary = f"{summary}..."
        rows.append(
            {
                "citation_index": idx,
                "file_path": citation.file_path,
                "line_start": citation.line_start,
                "line_end": citation.line_end,
                "score": round(float(citation.score), 4),
                "source_type": citation.source_type,
                "snippet": summary,
            }
        )
        if len(rows) >= max(limit, 1):
            break
    return rows


def compose_hybrid_answer(
    question: str,
    graph: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    used_fallback: bool,
) -> str:
    lines: list[str] = []
    lines.append("I. System Map (graph)")
    processes = graph.get("processes", []) if isinstance(graph, dict) else []
    if isinstance(processes, list) and processes:
        lines.append(f"- GitNexus identified {len(processes)} relevant process flow(s).")
        for process in processes[:5]:
            if not isinstance(process, dict):
                continue
            label = str(process.get("summary") or process.get("id") or "Unnamed process").strip()
            priority = process.get("priority")
            if priority is None:
                lines.append(f"- {label}")
            else:
                lines.append(f"- {label} (priority {priority})")
    else:
        lines.append("- No process-level graph flow matched strongly for this question.")

    graph_errors = graph.get("errors", []) if isinstance(graph, dict) else []
    if isinstance(graph_errors, list):
        condensed_errors = [str(item).strip() for item in graph_errors if str(item).strip()]
        if condensed_errors:
            lines.append(f"- Graph errors: {'; '.join(condensed_errors[:3])}")

    entrypoints = graph.get("entrypoints", []) if isinstance(graph, dict) else []
    if isinstance(entrypoints, list) and entrypoints:
        lines.append(f"- Entrypoints/signals: {', '.join(str(item) for item in entrypoints[:8])}")

    candidate_files = graph.get("candidate_files", []) if isinstance(graph, dict) else []
    if isinstance(candidate_files, list) and candidate_files:
        preview = ", ".join(str(item) for item in candidate_files[:8])
        suffix = " ..." if len(candidate_files) > 8 else ""
        lines.append(
            f"- Candidate files constraining retrieval ({len(candidate_files)}): {preview}{suffix}"
        )
    if used_fallback:
        lines.append("- Hybrid fallback applied: graph-constrained retrieval had no chunks, so standard Pinecone retrieval was used.")

    impact = graph.get("impact", {}) if isinstance(graph, dict) else {}
    if isinstance(impact, dict):
        upstream = impact.get("upstream", {}) if isinstance(impact.get("upstream"), dict) else {}
        downstream = impact.get("downstream", {}) if isinstance(impact.get("downstream"), dict) else {}
        upstream_count = safe_int(upstream.get("impactedCount")) or 0
        downstream_count = safe_int(downstream.get("impactedCount")) or 0
        if upstream_count or downstream_count:
            lines.append(
                f"- Impact snapshot for target {graph.get('target_symbol') or 'n/a'}: upstream={upstream_count}, downstream={downstream_count}"
            )

    lines.append("")
    lines.append("II. Evidence-backed explanation (with citations)")
    if evidence_rows:
        lines.append(f'- Evidence for "{question}":')
        for row in evidence_rows[:8]:
            lines.append(
                f"- [{row.get('citation_index')}] {row.get('file_path')}:{row.get('line_start')}-{row.get('line_end')} {row.get('snippet')}"
            )
    else:
        lines.append("- No Pinecone evidence was retrieved for this question.")

    lines.append("")
    lines.append("III. Next actions")
    if evidence_rows:
        lines.append("- Open the top cited chunks to confirm control/data-flow assumptions before making changes.")
    else:
        lines.append("- Refine the target symbol/file name and rerun Hybrid mode.")
    if not candidate_files:
        lines.append("- Graph candidate files were empty; broaden query terms or run a direct graph query first.")
        if isinstance(graph_errors, list) and any(str(item).strip() for item in graph_errors):
            lines.append("- Graph engine reported an error; verify GitNexus runtime/index availability for this deployment.")
    else:
        lines.append("- If needed, run focused follow-up on one candidate file with mode=search for exhaustive references.")
    return "\n".join(lines)


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
            "index_lexical": retrieval_debug.get("index_lexical", {}),
            "uploads": retrieval_debug.get("uploads", {}),
            "lexical_candidates": retrieval_debug.get("lexical_candidates", {}),
        },
        "feature_flags": retrieval_debug.get("feature_flags", {}),
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
    path_prefix, language, source_type_filter = normalized_request_filters(
        payload.path_prefix,
        payload.language,
        payload.source_type,
    )
    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=[],
            mode=normalize_mode(payload.mode),
            scope=normalize_scope(payload.scope),
            project_id=normalize_project_id(payload.project_id),
            path_prefix=path_prefix,
            language=language,
            source_type_filter=source_type_filter,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector search failed: {exc}") from exc

    analysis = apply_mode_analysis(
        mode=payload.mode,
        question=payload.question,
        citations=citations,
        chunks=chunks,
    )
    citations = analysis["citations"]
    chunks = analysis["chunks"]
    mode_metrics = analysis.get("mode_metrics", {})
    evidence = compute_evidence_strength(
        payload.question,
        citations,
        retrieval_debug,
        mode=payload.mode,
        mode_metrics=mode_metrics,
    )
    debug_payload: dict[str, Any] = {}
    if payload.debug:
        context = build_context(citations, chunks)
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return SearchResponse(
        matches=citations,
        evidence_strength=evidence,
        debug=debug_payload,
        result_type=str(analysis.get("result_type", "Ranked Chunks")),
        summary=analysis.get("summary"),
        follow_ups=list(analysis.get("follow_ups", [])),
        file_results=list(analysis.get("file_results", [])),
        graph_edges=list(analysis.get("graph_edges", [])),
        pattern_examples=list(analysis.get("pattern_examples", [])),
    )


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    normalized_mode = normalize_mode(payload.mode)
    normalized_scope = normalize_scope(payload.scope)
    normalized_project_id = normalize_project_id(payload.project_id)
    path_prefix, language, source_type_filter = normalized_request_filters(
        payload.path_prefix,
        payload.language,
        payload.source_type,
    )
    try:
        citations, chunks, retrieval_debug, graph_payload, route_debug, routed_plan = run_routed_retrieval_plan(
            question=payload.question,
            mode=normalized_mode,
            top_k=payload.top_k,
            scope=normalized_scope,
            project_id=normalized_project_id,
            path_prefix=path_prefix,
            language=language,
            source_type_filter=source_type_filter,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Routed retrieval failed: {exc}") from exc

    debug_payload: dict[str, Any] = {"route_debug": route_debug}

    if routed_plan == PLAN_GRAPH_ONLY:
        answer = compose_hybrid_answer(
            question=payload.question,
            graph=graph_payload,
            evidence_rows=[],
            used_fallback=False,
        )
        if payload.debug:
            debug_payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            debug_payload["graph"] = graph_payload
        return QueryResponse(
            answer=answer,
            citations=[],
            graph=graph_payload,
            evidence=[],
            evidence_strength={
                "label": "Low",
                "score": 0.0,
                "reason": "Graph-only mode does not retrieve Pinecone evidence.",
                "metrics": {"mode": "graph"},
            },
            debug=debug_payload,
        )

    evidence = compute_evidence_strength(
        payload.question,
        citations,
        retrieval_debug,
        mode=payload.mode,
        mode_metrics={},
    )
    evidence_rows = build_hybrid_evidence_rows(citations, chunks, limit=8)

    if normalized_mode == "hybrid":
        answer = compose_hybrid_answer(
            question=payload.question,
            graph=graph_payload,
            evidence_rows=evidence_rows,
            used_fallback=bool(route_debug.get("escalation", {}).get("did_escalate")),
        )
        if payload.debug:
            context = build_context(citations, chunks) if citations else ""
            verbose_debug = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            verbose_debug["graph"] = graph_payload
            verbose_debug["route_debug"] = route_debug
            debug_payload = verbose_debug
        return QueryResponse(
            answer=answer,
            citations=citations,
            graph=graph_payload,
            evidence=evidence_rows,
            evidence_strength=evidence,
            debug=debug_payload,
        )

    if not citations:
        suggestions = suggest_next_investigation(payload.question, citations)
        missing_terms = missing_focus_terms_from_debug(retrieval_debug)
        context = ""
        if payload.debug:
            verbose_debug = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            verbose_debug["route_debug"] = route_debug
            debug_payload = verbose_debug
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms),
            citations=[],
            graph=graph_payload,
            evidence=[],
            evidence_strength=evidence,
            debug=debug_payload,
        )

    context = build_context(citations, chunks)
    if is_weak_evidence(evidence):
        suggestions = suggest_next_investigation(payload.question, citations)
        missing_terms = missing_focus_terms_from_debug(retrieval_debug)
        response_citations = [] if missing_terms else citations
        if payload.debug:
            verbose_debug = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            verbose_debug["route_debug"] = route_debug
            debug_payload = verbose_debug
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms),
            citations=response_citations,
            graph=graph_payload,
            evidence=build_hybrid_evidence_rows(response_citations, chunks, limit=8),
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

    if payload.debug:
        verbose_debug = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        verbose_debug["route_debug"] = route_debug
        debug_payload = verbose_debug
    return QueryResponse(
        answer=answer,
        citations=citations,
        graph=graph_payload,
        evidence=evidence_rows,
        evidence_strength=evidence,
        debug=debug_payload,
    )


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
    chunk_count = safe_int((record or {}).get("chunk_count")) or 0
    delete_attachment_vectors(project_id=project, file_sha=file_sha, chunk_count=chunk_count)
    if record:
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
    mode: str = Form("chat"),
    scope: str = Form("both"),
    project_id: str | None = Form(default=None),
    path_prefix: str | None = Form(default=None),
    language: str | None = Form(default=None),
    source_type: str | None = Form(default=None),
    persist_uploads: str | None = Form(default=None),
) -> SearchResponse:
    payload = validate_query_request(
        question=question,
        top_k=top_k,
        debug=parse_form_bool(debug),
        mode=mode,
        scope=scope,
        project_id=project_id,
        path_prefix=path_prefix,
        language=language,
        source_type=source_type,
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
            mode=payload.mode,
            scope=payload.scope,
            project_id=payload.project_id,
            path_prefix=payload.path_prefix,
            language=payload.language,
            source_type_filter=payload.source_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload retrieval failed: {exc}") from exc

    analysis = apply_mode_analysis(
        mode=payload.mode,
        question=payload.question,
        citations=citations,
        chunks=chunks,
    )
    citations = analysis["citations"]
    chunks = analysis["chunks"]
    evidence = compute_evidence_strength(
        payload.question,
        citations,
        retrieval_debug,
        mode=payload.mode,
        mode_metrics=analysis.get("mode_metrics", {}),
    )
    debug_payload: dict[str, Any] = {}
    if payload.debug:
        context = build_context(citations, chunks)
        debug_payload = build_debug_payload(
            retrieval_debug=retrieval_debug,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return SearchResponse(
        matches=citations,
        evidence_strength=evidence,
        debug=debug_payload,
        result_type=str(analysis.get("result_type", "Ranked Chunks")),
        summary=analysis.get("summary"),
        follow_ups=list(analysis.get("follow_ups", [])),
        file_results=list(analysis.get("file_results", [])),
        graph_edges=list(analysis.get("graph_edges", [])),
        pattern_examples=list(analysis.get("pattern_examples", [])),
    )


@app.post("/api/query/upload", response_model=QueryResponse)
async def query_with_uploads(
    question: str = Form(...),
    top_k: int = Form(5),
    files: list[UploadFile] | None = File(default=None),
    debug: str | None = Form(default=None),
    mode: str = Form("chat"),
    scope: str = Form("both"),
    project_id: str | None = Form(default=None),
    path_prefix: str | None = Form(default=None),
    language: str | None = Form(default=None),
    source_type: str | None = Form(default=None),
    persist_uploads: str | None = Form(default=None),
) -> QueryResponse:
    payload = validate_query_request(
        question=question,
        top_k=top_k,
        debug=parse_form_bool(debug),
        mode=mode,
        scope=scope,
        project_id=project_id,
        path_prefix=path_prefix,
        language=language,
        source_type=source_type,
    )
    uploaded_files = await extract_uploaded_files(files or [])
    persist = parse_form_bool(persist_uploads)
    if persist and uploaded_files:
        upsert_attachment_chunks(project_id=payload.project_id, uploaded_files=uploaded_files)
    normalized_mode = normalize_mode(payload.mode)
    if normalized_mode in {"hybrid", "graph"}:
        # Hybrid/graph paths are repo-structure-first and currently ignore transient uploads.
        return query(payload)
    started = time.perf_counter()

    try:
        citations, chunks, retrieval_debug = retrieve_with_optional_uploads(
            question=payload.question,
            top_k=payload.top_k,
            uploaded_files=uploaded_files,
            mode=payload.mode,
            scope=payload.scope,
            project_id=payload.project_id,
            path_prefix=payload.path_prefix,
            language=payload.language,
            source_type_filter=payload.source_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload retrieval failed: {exc}") from exc

    evidence = compute_evidence_strength(
        payload.question,
        citations,
        retrieval_debug,
        mode=payload.mode,
        mode_metrics={},
    )
    if not citations:
        suggestions = suggest_next_investigation(payload.question, citations)
        missing_terms = missing_focus_terms_from_debug(retrieval_debug)
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms),
            citations=[],
            graph={},
            evidence=[],
            evidence_strength=evidence,
            debug=debug_payload,
        )

    context = build_context(citations, chunks)
    if is_weak_evidence(evidence):
        suggestions = suggest_next_investigation(payload.question, citations)
        missing_terms = missing_focus_terms_from_debug(retrieval_debug)
        response_citations = [] if missing_terms else citations
        debug_payload: dict[str, Any] = {}
        if payload.debug:
            debug_payload = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        return QueryResponse(
            answer=insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms),
            citations=response_citations,
            graph={},
            evidence=build_hybrid_evidence_rows(response_citations, chunks, limit=8),
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
    return QueryResponse(
        answer=answer,
        citations=citations,
        graph={},
        evidence=build_hybrid_evidence_rows(citations, chunks, limit=8),
        evidence_strength=evidence,
        debug=debug_payload,
    )
