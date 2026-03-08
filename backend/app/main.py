import logging
import math
import re
import time
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit, urlunsplit

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
from app.telemetry import (
    RagRequestTelemetry,
    create_request_telemetry,
    emit_telemetry_log,
    get_telemetry_store,
    parse_embedding_usage,
    parse_openai_usage,
    parse_pinecone_usage,
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
TEST_QUERY_HINTS = (
    "test",
    "tests",
    "pytest",
    "spec",
    "failing test",
    "regression",
    "coverage",
    "assert",
    "assertion",
    "mock",
    "fixture",
    "expected behavior",
    "unit test",
    "integration test",
)
TEST_PATH_SEGMENT_PATTERN = re.compile(r"(^|/)(tests?|__tests__)(/|$)", re.IGNORECASE)
TEST_FILE_NAME_PATTERN = re.compile(
    r"(^|/)(test_[^/]+\.py|[^/]+_test\.py|[^/]+\.spec\.[^/]+|[^/]+\.test\.[^/]+)$",
    re.IGNORECASE,
)
MODE_VALUES = {"chat", "search", "patterns", "dependencies", "hybrid", "graph"}
FILTERABLE_LANGUAGES = {"fortran", "text", "pdf"}
FILTERABLE_SOURCE_TYPES = {"repo", "upload", "temp-upload"}
STARTUP_SMOKE_MODES = {"off", "warn", "strict"}
QUERY_MAX_CHARS = 8000
DIRECT_UI_MODES = {"audit", "diagrams"}
DIRECT_DIAGRAM_TYPES = {
    "systemArchitecture",
    "executionPipeline",
    "retrievalFlow",
    "dataFlow",
    "dependencyGraph",
    "buildRuntime",
    "auditDiagram",
}
DIRECT_CONTEXT_MAX_FILES = 12
DIRECT_CONTEXT_MAX_CHARS = 18_000
DIRECT_SOURCE_PROMPT_CHARS = 1_800
DIRECT_UPLOAD_PROMPT_CHARS = 1_400
DIRECT_SOURCE_SNIPPET_CHARS = 550
DIRECT_FILE_PREVIEW_LINES = 80
DIRECT_SYMBOL_PREVIEW_LIMIT = 3
DIRECT_DIAGRAM_MAX_LANES = 6
DIRECT_DIAGRAM_MAX_NODES = 14
DIRECT_DIAGRAM_MAX_EDGES = 20
TEXTUAL_CONTEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".conf",
    ".f",
    ".f03",
    ".f08",
    ".f90",
    ".f95",
    ".for",
    ".h",
    ".inc",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IMPORTANT_TEXT_FILE_NAMES = {
    ".env.example",
    "dockerfile",
    "makefile",
    "package.json",
    "procfile",
    "pyproject.toml",
    "readme",
    "readme.md",
    "requirements.txt",
}
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
GRAPH_PROCESS_PRIORITY_THRESHOLD = 0.20
GENERIC_REPO_NAMES = {"app", "workspace", "repo"}

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
    ui_mode: str | None = Field(default=None, max_length=40)
    diagram_type: str | None = Field(default=None, max_length=80)
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
    type: str = "text"
    format: str = "text"
    content: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_strength: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)


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
    telemetry: dict[str, Any] = Field(default_factory=dict)
    result_type: str = "Ranked Chunks"
    summary: str | None = None
    follow_ups: list[str] = Field(default_factory=list)
    file_results: list[FileResult] = Field(default_factory=list)
    graph_edges: list[DependencyEdge] = Field(default_factory=list)
    pattern_examples: list[PatternExample] = Field(default_factory=list)


class TelemetrySummaryResponse(BaseModel):
    request_count: int
    avg_cost_usd_est: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    failure_rate: float
    average_retrieved_chunks: float
    average_selected_chunks: float
    cost_by_mode: list[dict[str, Any]] = Field(default_factory=list)
    cost_by_repo: list[dict[str, Any]] = Field(default_factory=list)
    cost_by_model: list[dict[str, Any]] = Field(default_factory=list)


class TelemetryRequestsResponse(BaseModel):
    requests: list[dict[str, Any]] = Field(default_factory=list)


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
gitnexus_bootstrap_lock = threading.Lock()
gitnexus_bootstrap_state: dict[str, Any] = {
    "attempted": False,
    "ok": None,
    "repo_id": None,
    "repo_path": None,
    "error": None,
}
runtime_repo_root_override: Path | None = None
runtime_gitnexus_repo_override: str | None = None


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
    if settings.telemetry_enabled:
        try:
            get_telemetry_store()
        except Exception as exc:
            logger.warning("Telemetry store initialization failed: %s", exc)
    if settings.gitnexus_enabled and settings.gitnexus_bootstrap_enabled:
        try:
            ensure_gitnexus_bootstrap_index(force=False)
        except Exception as exc:
            logger.warning("GitNexus bootstrap on startup failed: %s", exc)
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


@app.get("/api/telemetry/summary", response_model=TelemetrySummaryResponse)
def telemetry_summary() -> TelemetrySummaryResponse:
    if not settings.telemetry_enabled:
        return TelemetrySummaryResponse(
            request_count=0,
            avg_cost_usd_est=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            failure_rate=0.0,
            average_retrieved_chunks=0.0,
            average_selected_chunks=0.0,
            cost_by_mode=[],
            cost_by_repo=[],
            cost_by_model=[],
        )
    return TelemetrySummaryResponse(**get_telemetry_store().summary())


@app.get("/api/telemetry/requests", response_model=TelemetryRequestsResponse)
def telemetry_requests(limit: int = 25) -> TelemetryRequestsResponse:
    if not settings.telemetry_enabled:
        return TelemetryRequestsResponse(requests=[])
    safe_limit = max(1, min(int(limit or settings.telemetry_recent_limit), 200))
    return TelemetryRequestsResponse(requests=get_telemetry_store().recent(safe_limit))


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


def gitnexus_bootstrap_repo_path() -> Path:
    raw = str(settings.gitnexus_bootstrap_repo_path or "").strip() or "/tmp/nshmp-main"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return (Path.cwd() / path).resolve()
    return path.resolve()


def _tail_text(value: str, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    secret = str(settings.gitnexus_bootstrap_git_token or "").strip()
    if secret:
        text = text.replace(secret, "***")
    if len(text) <= limit:
        return text
    return f"...{text[-limit:]}"


def _available_repo_names(rows: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = _repo_name_from_row(row)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return dedupe_preserve_order(names)


def gitnexus_bootstrap_clone_url(repo_url: str) -> str:
    url = str(repo_url or "").strip()
    token = str(settings.gitnexus_bootstrap_git_token or "").strip()
    if not url or not token:
        return url
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return url
    if "@" in parsed.netloc:
        return url
    token_part = quote(token, safe="")
    auth_netloc = f"x-access-token:{token_part}@{parsed.netloc}"
    return urlunsplit((parsed.scheme, auth_netloc, parsed.path, parsed.query, parsed.fragment))


def ensure_gitnexus_source_checkout() -> Path | None:
    repo_url = str(settings.gitnexus_bootstrap_repo_url or "").strip()
    if not repo_url:
        return None

    target = gitnexus_bootstrap_repo_path()
    target_git = target / ".git"
    timeout_seconds = max(float(settings.gitnexus_analyze_timeout_seconds), 30.0)
    branch = str(settings.gitnexus_bootstrap_repo_ref or "").strip()
    safe_tmp_path = str(target).startswith("/tmp/") or str(target).startswith("/var/tmp/")

    if target_git.is_dir():
        if not branch:
            return target
        fetch_cmd = ["git", "-C", str(target), "fetch", "--depth", "1", "origin", branch]
        checkout_cmd = ["git", "-C", str(target), "checkout", "--force", "FETCH_HEAD"]
        try:
            fetch_proc = subprocess.run(
                fetch_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            logger.warning("GitNexus bootstrap git fetch failed: %s", exc)
            return None
        if fetch_proc.returncode != 0:
            logger.warning(
                "GitNexus bootstrap git fetch failed for ref %s (rc=%s): %s",
                branch,
                fetch_proc.returncode,
                _tail_text((fetch_proc.stderr or "") + " " + (fetch_proc.stdout or "")),
            )
            if safe_tmp_path:
                try:
                    shutil.rmtree(target)
                except Exception:
                    return None
            return None
        try:
            checkout_proc = subprocess.run(
                checkout_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            logger.warning("GitNexus bootstrap git checkout failed: %s", exc)
            return None
        if checkout_proc.returncode != 0:
            logger.warning(
                "GitNexus bootstrap git checkout failed for ref %s (rc=%s): %s",
                branch,
                checkout_proc.returncode,
                _tail_text((checkout_proc.stderr or "") + " " + (checkout_proc.stdout or "")),
            )
            return None
        return target

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("GitNexus bootstrap could not create parent directory (%s): %s", target.parent, exc)
        return None

    if target.exists() and any(target.iterdir()):
        if safe_tmp_path:
            try:
                shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("GitNexus bootstrap failed to reset stale target %s: %s", target, exc)
                return None
        else:
            logger.warning("GitNexus bootstrap target exists and is not empty: %s", target)
            return None

    clone_cmd = ["git", "clone", "--depth", "1"]
    if branch:
        clone_cmd.extend(["--branch", branch])
    clone_cmd.extend([gitnexus_bootstrap_clone_url(repo_url), str(target)])

    timeout_seconds = max(float(settings.gitnexus_analyze_timeout_seconds), 30.0)
    try:
        proc = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        logger.warning("GitNexus bootstrap git clone failed: %s", exc)
        return None

    if proc.returncode != 0:
        logger.warning(
            "GitNexus bootstrap git clone failed (rc=%s): %s",
            proc.returncode,
            _tail_text((proc.stderr or "") + " " + (proc.stdout or "")),
        )
        return None

    return target if target_git.is_dir() else None


def ensure_gitnexus_bootstrap_index(force: bool = False) -> dict[str, Any]:
    global gitnexus_bootstrap_state, runtime_repo_root_override, runtime_gitnexus_repo_override
    snapshot = dict(gitnexus_bootstrap_state)
    if not settings.gitnexus_enabled or not settings.gitnexus_bootstrap_enabled:
        return snapshot

    with gitnexus_bootstrap_lock:
        current = dict(gitnexus_bootstrap_state)
        if not force and current.get("attempted") and current.get("ok"):
            return current

        next_state: dict[str, Any] = {
            "attempted": True,
            "ok": False,
            "repo_id": None,
            "repo_path": None,
            "error": None,
        }
        source_path = ensure_gitnexus_source_checkout()
        if source_path is None:
            next_state["error"] = "source_checkout_unavailable"
            gitnexus_bootstrap_state = next_state
            return dict(next_state)

        analyze_prefix = str(settings.gitnexus_analyze_command or "").strip() or "gitnexus analyze"
        analyze_cmd = shlex.split(analyze_prefix) + [str(source_path)]
        timeout_seconds = max(float(settings.gitnexus_analyze_timeout_seconds), 30.0)
        try:
            proc = subprocess.run(
                analyze_cmd,
                capture_output=True,
                text=True,
                cwd=str(source_path),
                check=False,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            next_state["error"] = f"analyze_exception:{_tail_text(exc)}"
            gitnexus_bootstrap_state = next_state
            return dict(next_state)

        if proc.returncode != 0:
            next_state["error"] = f"analyze_failed:{_tail_text((proc.stderr or '') + ' ' + (proc.stdout or ''))}"
            gitnexus_bootstrap_state = next_state
            return dict(next_state)

        repo_id = source_path.name
        runtime_repo_root_override = source_path
        runtime_gitnexus_repo_override = repo_id
        next_state.update(
            {
                "ok": True,
                "repo_id": repo_id,
                "repo_path": str(source_path),
                "error": None,
            }
        )
        gitnexus_bootstrap_state = next_state
        logger.info("GitNexus bootstrap index ready for repo=%s path=%s", repo_id, source_path)
        return dict(next_state)


def gitnexus_bootstrap_debug_state(state: dict[str, Any] | None = None) -> dict[str, Any]:
    row = state or gitnexus_bootstrap_state
    payload: dict[str, Any] = {
        "attempted": bool(row.get("attempted")),
        "ok": row.get("ok"),
        "repo_id": row.get("repo_id"),
        "repo_path": str(row.get("repo_path") or "") or None,
        "error": row.get("error"),
    }
    return payload


def default_gitnexus_repo() -> str:
    if isinstance(runtime_gitnexus_repo_override, str) and runtime_gitnexus_repo_override.strip():
        return runtime_gitnexus_repo_override.strip()

    configured = (settings.gitnexus_default_repo or "").strip()
    if configured and configured.lower() not in GENERIC_REPO_NAMES:
        return configured

    namespace_hint = str(settings.pinecone_namespace or "").split(":", 1)[0].strip()
    if namespace_hint:
        return namespace_hint

    bootstrap_hint = gitnexus_bootstrap_repo_path().name.strip()
    if settings.gitnexus_bootstrap_enabled and bootstrap_hint and bootstrap_hint.lower() not in GENERIC_REPO_NAMES:
        return bootstrap_hint

    if configured:
        return configured

    inferred = repo_root_path().name.strip()
    if inferred and inferred.lower() not in GENERIC_REPO_NAMES:
        return inferred
    if inferred:
        return inferred
    return "unknown-repo"


@lru_cache(maxsize=1)
def repo_commit_short() -> str | None:
    for key in ("RAILWAY_GIT_COMMIT_SHA", "GIT_COMMIT", "SOURCE_COMMIT"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value[:12]
    root = repo_root_path()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.2,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    commit = (proc.stdout or "").strip()
    return commit or None


def _repo_name_from_row(row: dict[str, Any]) -> str | None:
    for key in ("repo", "name", "id", "repo_id", "slug"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _repo_metric_from_row(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = row.get(key)
        parsed = safe_int(value)
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _repo_index_details(
    selected_repo: str,
    list_repos_result: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    rows = list_repos_result if isinstance(list_repos_result, list) else []
    names: list[str] = []
    repo_row: dict[str, Any] | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _repo_name_from_row(row)
        if not name:
            continue
        names.append(name)
        if name == selected_repo and repo_row is None:
            repo_row = row
    names = dedupe_preserve_order(names)

    build_timestamp = None
    node_count = None
    edge_count = None
    if isinstance(repo_row, dict):
        for key in ("built_at", "build_timestamp", "updated_at", "created_at"):
            value = repo_row.get(key)
            if isinstance(value, str) and value.strip():
                build_timestamp = value.strip()
                break
        node_count = _repo_metric_from_row(repo_row, ("nodes", "node_count", "symbols"))
        edge_count = _repo_metric_from_row(repo_row, ("edges", "edge_count", "relationships"))

    return {
        "repo_id": selected_repo,
        "available_repos": names[:40],
        "index_present": bool(selected_repo and selected_repo in names) if names else None,
        "build_timestamp": build_timestamp,
        "node_count": node_count,
        "edge_count": edge_count,
    }


def resolve_gitnexus_repo_name(selected_repo: str, available_repos: list[str]) -> str:
    names = [str(item).strip() for item in available_repos if str(item).strip()]
    if not names:
        return selected_repo

    namespace_hint = str(settings.pinecone_namespace or "").split(":", 1)[0].strip()
    configured = str(settings.gitnexus_default_repo or "").strip()
    bootstrap_hint = str((gitnexus_bootstrap_repo_path().name if settings.gitnexus_bootstrap_enabled else "")).strip()

    if selected_repo in names:
        if selected_repo.lower() in GENERIC_REPO_NAMES and namespace_hint and namespace_hint in names:
            return namespace_hint
        return selected_repo
    if namespace_hint and namespace_hint in names:
        return namespace_hint
    if configured and configured in names:
        return configured
    if isinstance(runtime_gitnexus_repo_override, str) and runtime_gitnexus_repo_override in names:
        return runtime_gitnexus_repo_override
    if bootstrap_hint and bootstrap_hint in names:
        return bootstrap_hint
    if len(names) == 1:
        return names[0]
    return selected_repo


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


def _graph_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _graph_node_id(kind: str, value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.:/-]+", "_", str(value or "").strip().lower()).strip("_")
    return f"{kind}:{slug or 'unknown'}"


def _graph_impact_rows(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    by_depth = payload.get("byDepth", {}) if isinstance(payload, dict) else {}
    if not isinstance(by_depth, dict):
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for depth_key, values in by_depth.items():
        try:
            depth = int(depth_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, dict):
                rows.append((depth, row))
    return rows


def build_hybrid_graph_canvas(
    question: str,
    query_result: dict[str, Any],
    context_result: dict[str, Any],
    impact_result: dict[str, Any],
    candidate_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    edge_ids: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, label: str, kind: str, size: float = 1.0, path: str | None = None) -> None:
        if not node_id or node_id in node_ids:
            return
        node_ids.add(node_id)
        payload = {
            "id": node_id,
            "label": (label or node_id)[:80],
            "kind": kind,
            "size": round(max(float(size), 0.5), 3),
        }
        if path:
            payload["path"] = path
        nodes.append(payload)

    def add_edge(source: str, target: str, kind: str) -> None:
        if not source or not target:
            return
        edge_key = (source, target, kind)
        if edge_key in edge_ids:
            return
        edge_ids.add(edge_key)
        edges.append({"source": source, "target": target, "kind": kind})

    query_label = "Hybrid Query"
    query_node = _graph_node_id("query", query_label)
    add_node(query_node, query_label, "query", size=1.6)

    target = infer_hybrid_target(question)
    target_node = ""
    if target:
        target_node = _graph_node_id("symbol", target)
        add_node(target_node, target, "symbol", size=1.2)
        add_edge(query_node, target_node, "focus")

    process_nodes: dict[str, str] = {}
    for process in (query_result.get("processes", []) if isinstance(query_result, dict) else []):
        if not isinstance(process, dict):
            continue
        pid = _graph_value(process, ("id",))
        label = _graph_value(process, ("summary", "id")) or "Process"
        node_id = _graph_node_id("process", pid or label)
        priority = float(process.get("priority", 0.0) or 0.0) if isinstance(process.get("priority"), (int, float)) else 0.0
        add_node(node_id, label, "process", size=1.0 + min(max(priority, 0.0), 1.0))
        add_edge(query_node, node_id, "process")
        if pid:
            process_nodes[pid] = node_id
        if len(process_nodes) >= 24:
            break

    for row in (query_result.get("process_symbols", []) if isinstance(query_result, dict) else []):
        if not isinstance(row, dict):
            continue
        symbol_label = _graph_value(row, ("name", "symbol", "id", "uid"))
        file_path = normalize_file_path(_graph_value(row, ("filePath", "file_path")))
        process_id = _graph_value(row, ("process_id",))
        if not symbol_label and not file_path:
            continue

        symbol_node = _graph_node_id("symbol", symbol_label or file_path or "symbol")
        add_node(symbol_node, symbol_label or "Symbol", "symbol", size=1.0)
        if process_id and process_id in process_nodes:
            add_edge(process_nodes[process_id], symbol_node, "contains")
        else:
            add_edge(query_node, symbol_node, "symbol")

        if file_path:
            file_node = _graph_node_id("file", file_path)
            add_node(file_node, Path(file_path).name, "file", size=0.95, path=file_path)
            add_edge(symbol_node, file_node, "defined_in")
        if len(nodes) >= 90:
            break

    for row in (query_result.get("definitions", []) if isinstance(query_result, dict) else []):
        if not isinstance(row, dict):
            continue
        file_path = normalize_file_path(_graph_value(row, ("filePath", "file_path")))
        symbol_label = _graph_value(row, ("name", "symbol", "id"))
        if symbol_label:
            symbol_node = _graph_node_id("symbol", symbol_label)
            add_node(symbol_node, symbol_label, "symbol", size=0.95)
            add_edge(query_node, symbol_node, "definition")
            if file_path:
                file_node = _graph_node_id("file", file_path)
                add_node(file_node, Path(file_path).name, "file", size=0.9, path=file_path)
                add_edge(symbol_node, file_node, "defined_in")
        elif file_path:
            file_node = _graph_node_id("file", file_path)
            add_node(file_node, Path(file_path).name, "file", size=0.9, path=file_path)
            add_edge(query_node, file_node, "definition_file")
        if len(nodes) >= 100:
            break

    context_symbol = context_result.get("symbol") if isinstance(context_result, dict) else None
    if isinstance(context_symbol, dict):
        ctx_name = _graph_value(context_symbol, ("name", "symbol", "id", "uid"))
        ctx_path = normalize_file_path(_graph_value(context_symbol, ("filePath", "file_path")))
        ctx_node = _graph_node_id("symbol", ctx_name or ctx_path or "context")
        add_node(ctx_node, ctx_name or "Context Symbol", "symbol", size=1.05)
        add_edge(target_node or query_node, ctx_node, "context")
        if ctx_path:
            file_node = _graph_node_id("file", ctx_path)
            add_node(file_node, Path(ctx_path).name, "file", size=0.9, path=ctx_path)
            add_edge(ctx_node, file_node, "in_file")

    for row in candidate_ranking[:32]:
        if not isinstance(row, dict):
            continue
        file_path = normalize_file_path(str(row.get("file_path") or ""))
        if not file_path:
            continue
        file_node = _graph_node_id("file", file_path)
        score_raw = row.get("score", 1.0)
        try:
            score = max(float(score_raw), 0.2)
        except (TypeError, ValueError):
            score = 1.0
        add_node(file_node, Path(file_path).name, "file", size=min(1.4, 0.8 + (score * 0.12)), path=file_path)
        add_edge(query_node, file_node, "candidate")

    impact_rows = _graph_impact_rows(impact_result if isinstance(impact_result, dict) else {})
    for depth, row in impact_rows[:40]:
        if not isinstance(row, dict):
            continue
        label = _graph_value(row, ("name", "symbol", "id", "uid"))
        file_path = normalize_file_path(_graph_value(row, ("filePath", "file_path")))
        impact_node = _graph_node_id("impact", label or file_path or f"depth{depth}")
        add_node(impact_node, label or f"Impact depth {depth}", "impact", size=0.9)
        add_edge(target_node or query_node, impact_node, f"impact_d{depth}")
        if file_path:
            file_node = _graph_node_id("file", file_path)
            add_node(file_node, Path(file_path).name, "file", size=0.88, path=file_path)
            add_edge(impact_node, file_node, "touches")

    return {"nodes": nodes[:120], "edges": edges[:220]}


def run_gitnexus_graph(
    question: str,
    repo_name: str | None = None,
    include_tests: bool = False,
) -> dict[str, Any]:
    normalized_question = " ".join(str(question or "").split())
    selected_repo = (repo_name or default_gitnexus_repo()).strip()
    graph: dict[str, Any] = {
        "repo": selected_repo,
        "repo_requested": selected_repo,
        "query": normalized_question,
        "processes": [],
        "entrypoints": [],
        "impact": {},
        "candidate_files": [],
        "candidate_file_ranking": [],
        "canvas": {"nodes": [], "edges": []},
        "target_symbol": None,
        "errors": [],
        "raw_counts": {"processes": 0, "nodes": 0, "edges": 0, "files": 0},
        "score": {"best": 0.0, "threshold": GRAPH_PROCESS_PRIORITY_THRESHOLD, "passed": False},
        "index": {
            "repo_id": selected_repo,
            "commit_hash": repo_commit_short(),
            "available_repos": [],
            "index_present": None,
            "build_timestamp": None,
            "node_count": None,
            "edge_count": None,
            "bootstrap": gitnexus_bootstrap_debug_state(),
        },
    }

    if not selected_repo or selected_repo == "unknown-repo":
        graph["errors"].append("gitnexus_repo_unresolved")
    if not settings.gitnexus_enabled:
        graph["errors"].append("gitnexus_disabled")
        graph["index"]["index_present"] = False
        return graph

    bootstrap_state = gitnexus_bootstrap_debug_state()
    if settings.gitnexus_bootstrap_enabled:
        bootstrap_state = ensure_gitnexus_bootstrap_index(force=False)
        graph["index"]["bootstrap"] = gitnexus_bootstrap_debug_state(bootstrap_state)

    try:
        client = get_gitnexus_client()
        repo_rows = client.list_repos()
        available_repos = _available_repo_names(repo_rows)
        resolved_repo = resolve_gitnexus_repo_name(selected_repo, available_repos)
        if resolved_repo != selected_repo:
            selected_repo = resolved_repo
            graph["repo"] = selected_repo
            graph["index"]["repo_id"] = selected_repo

        index_info = _repo_index_details(selected_repo, repo_rows)
        index_info["commit_hash"] = repo_commit_short()
        graph["index"] = index_info
        graph["index"]["bootstrap"] = gitnexus_bootstrap_debug_state(bootstrap_state)

        if index_info.get("index_present") is False and settings.gitnexus_bootstrap_enabled and not bootstrap_state.get("ok"):
            bootstrap_state = ensure_gitnexus_bootstrap_index(force=True)
            graph["index"]["bootstrap"] = gitnexus_bootstrap_debug_state(bootstrap_state)
            if bootstrap_state.get("ok"):
                close_gitnexus_client()
                client = get_gitnexus_client()
                repo_rows = client.list_repos()
                available_repos = _available_repo_names(repo_rows)
                resolved_repo = resolve_gitnexus_repo_name(selected_repo, available_repos)
                if resolved_repo != selected_repo:
                    selected_repo = resolved_repo
                    graph["repo"] = selected_repo
                index_info = _repo_index_details(selected_repo, repo_rows)
                index_info["commit_hash"] = repo_commit_short()
                index_info["bootstrap"] = gitnexus_bootstrap_debug_state(bootstrap_state)
                graph["index"] = index_info

        if graph["index"].get("index_present") is False:
            graph["errors"].append(f"gitnexus_repo_not_indexed:{selected_repo}")

        query_result = client.query(
            query=normalized_question,
            repo=selected_repo,
            limit=8,
            max_symbols=20,
            include_content=False,
        )
        graph["processes"] = list(query_result.get("processes", [])) if isinstance(query_result, dict) else []
        graph["entrypoints"] = graph_entrypoints(query_result if isinstance(query_result, dict) else {})

        target = infer_hybrid_target(normalized_question)
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
                include_tests=include_tests,
            )
            downstream_impact = client.impact(
                target=target,
                direction="downstream",
                repo=selected_repo,
                max_depth=3,
                min_confidence=0.75,
                include_tests=include_tests,
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
        graph["canvas"] = build_hybrid_graph_canvas(
            question=normalized_question,
            query_result=query_result if isinstance(query_result, dict) else {},
            context_result=context_result if isinstance(context_result, dict) else {},
            impact_result=merged_impact,
            candidate_ranking=candidate_ranking,
        )

        process_symbols = query_result.get("process_symbols", []) if isinstance(query_result, dict) else []
        definitions = query_result.get("definitions", []) if isinstance(query_result, dict) else []

        def _context_ref_count(payload: dict[str, Any], section: str) -> int:
            raw = payload.get(section, {})
            if not isinstance(raw, dict):
                return 0
            total = 0
            for values in raw.values():
                if isinstance(values, list):
                    total += sum(1 for row in values if isinstance(row, dict))
            return total

        def _impact_ref_count(payload: dict[str, Any]) -> int:
            by_depth = payload.get("byDepth", {})
            if not isinstance(by_depth, dict):
                return 0
            total = 0
            for values in by_depth.values():
                if isinstance(values, list):
                    total += sum(1 for row in values if isinstance(row, dict))
            return total

        incoming_count = _context_ref_count(context_result if isinstance(context_result, dict) else {}, "incoming")
        outgoing_count = _context_ref_count(context_result if isinstance(context_result, dict) else {}, "outgoing")
        impact_count = _impact_ref_count(merged_impact)
        node_count = (
            (len(process_symbols) if isinstance(process_symbols, list) else 0)
            + (len(definitions) if isinstance(definitions, list) else 0)
            + (1 if isinstance((context_result or {}).get("symbol"), dict) else 0)
            + incoming_count
            + outgoing_count
        )
        edge_count = incoming_count + outgoing_count + impact_count
        canvas_payload = graph.get("canvas", {})
        if isinstance(canvas_payload, dict):
            canvas_nodes = canvas_payload.get("nodes", [])
            canvas_edges = canvas_payload.get("edges", [])
            if isinstance(canvas_nodes, list):
                node_count = max(node_count, len(canvas_nodes))
            if isinstance(canvas_edges, list):
                edge_count = max(edge_count, len(canvas_edges))
        graph["raw_counts"] = {
            "processes": len(graph["processes"]) if isinstance(graph["processes"], list) else 0,
            "nodes": max(node_count, 0),
            "edges": max(edge_count, 0),
            "files": len(candidate_files),
        }

        best_priority = 0.0
        if isinstance(graph["processes"], list):
            for process in graph["processes"]:
                if not isinstance(process, dict):
                    continue
                try:
                    best_priority = max(best_priority, float(process.get("priority", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
        passed = bool(best_priority >= GRAPH_PROCESS_PRIORITY_THRESHOLD or candidate_files)
        graph["score"] = {
            "best": round(best_priority, 4),
            "threshold": GRAPH_PROCESS_PRIORITY_THRESHOLD,
            "passed": passed,
        }
    except (GitNexusClientError, HTTPException) as exc:
        graph["errors"].append(str(exc))
    except Exception as exc:
        graph["errors"].append(f"gitnexus_unavailable: {exc}")

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
    ui_mode: str | None = None,
    diagram_type: str | None = None,
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
            ui_mode=str(ui_mode or "").strip().lower() or None,
            diagram_type=str(diagram_type or "").strip() or None,
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
    telemetry: RagRequestTelemetry | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    client = get_openai_client()
    started = time.perf_counter()
    response = call_with_retries(
        "openai embeddings",
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        ),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if telemetry:
        usage = parse_embedding_usage(response)
        estimated_input_tokens = sum(token_count(text) for text in texts)
        telemetry.record_embedding(
            input_tokens=max(int(usage.get("input_tokens") or 0), estimated_input_tokens),
            latency_ms=elapsed_ms,
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
def _cached_question_embedding(question: str) -> tuple[tuple[float, ...], int]:
    client = get_openai_client()
    response = call_with_retries(
        "openai embeddings",
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=question,
        ),
    )
    usage = parse_embedding_usage(response)
    input_tokens = max(int(usage.get("input_tokens") or 0), token_count(question))
    return tuple(response.data[0].embedding), input_tokens


def embed_question(
    question: str,
    telemetry: RagRequestTelemetry | None = None,
) -> tuple[list[float], dict[str, Any]]:
    normalized = question.strip()
    if not normalized:
        return [], {"from_cache": False, "latency_ms": 0.0, "input_tokens": 0}
    cache_before = _cached_question_embedding.cache_info()
    started = time.perf_counter()
    embedding, input_tokens = _cached_question_embedding(normalized)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    cache_after = _cached_question_embedding.cache_info()
    from_cache = cache_after.hits > cache_before.hits
    if telemetry and not from_cache:
        telemetry.record_embedding(input_tokens=input_tokens, latency_ms=elapsed_ms)
    return list(embedding), {
        "from_cache": from_cache,
        "latency_ms": 0.0 if from_cache else elapsed_ms,
        "input_tokens": 0 if from_cache else input_tokens,
    }


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


def is_test_intent_query(question: str) -> bool:
    lowered = str(question or "").lower()
    return any(term in lowered for term in TEST_QUERY_HINTS)


def is_test_file_path(path: str | None) -> bool:
    normalized = normalize_file_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    return bool(TEST_PATH_SEGMENT_PATTERN.search(lowered) or TEST_FILE_NAME_PATTERN.search(lowered))


def filter_paths_by_test_policy(
    paths: list[str] | None,
    include_tests: bool,
) -> tuple[list[str], int, int]:
    normalized = dedupe_preserve_order(
        [path for path in (normalize_file_path(item) for item in (paths or [])) if path]
    )
    if include_tests:
        test_count = sum(1 for path in normalized if is_test_file_path(path))
        return normalized, 0, test_count

    filtered: list[str] = []
    excluded = 0
    for path in normalized:
        if is_test_file_path(path):
            excluded += 1
            continue
        filtered.append(path)
    final_test_count = sum(1 for path in filtered if is_test_file_path(path))
    return filtered, excluded, final_test_count


def filter_candidate_ranking_rows_by_test_policy(
    rows: list[dict[str, Any]] | None,
    include_tests: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    ranking = [row for row in (rows or []) if isinstance(row, dict)]
    if include_tests:
        test_count = sum(1 for row in ranking if is_test_file_path(row.get("file_path")))
        return ranking, 0, test_count

    filtered: list[dict[str, Any]] = []
    excluded = 0
    for row in ranking:
        if is_test_file_path(row.get("file_path")):
            excluded += 1
            continue
        filtered.append(row)
    final_test_count = sum(1 for row in filtered if is_test_file_path(row.get("file_path")))
    return filtered, excluded, final_test_count


def filter_citation_pairs_by_test_policy(
    citations: list[Citation],
    chunks: list[str],
    include_tests: bool,
    limit: int | None = None,
) -> tuple[list[Citation], list[str], int, int]:
    if include_tests:
        final_test_count = sum(1 for citation in citations if is_test_file_path(citation.file_path))
        bounded_citations = citations[:limit] if isinstance(limit, int) and limit > 0 else citations
        bounded_chunks = chunks[: len(bounded_citations)]
        return bounded_citations, bounded_chunks, 0, final_test_count

    filtered_citations: list[Citation] = []
    filtered_chunks: list[str] = []
    excluded = 0
    for citation, chunk in zip(citations, chunks, strict=False):
        if is_test_file_path(citation.file_path):
            excluded += 1
            continue
        filtered_citations.append(citation)
        filtered_chunks.append(chunk)
        if isinstance(limit, int) and limit > 0 and len(filtered_citations) >= limit:
            break
    final_test_count = sum(1 for citation in filtered_citations if is_test_file_path(citation.file_path))
    return filtered_citations, filtered_chunks, excluded, final_test_count


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
    if isinstance(runtime_repo_root_override, Path) and runtime_repo_root_override.is_dir():
        return runtime_repo_root_override

    configured_override = str(settings.repo_root_override or "").strip()
    if configured_override:
        candidate_override = Path(configured_override).expanduser()
        if not candidate_override.is_absolute():
            candidate_override = (Path.cwd() / candidate_override).resolve()
        if candidate_override.is_dir():
            return candidate_override

    bootstrap_path = gitnexus_bootstrap_repo_path()
    if settings.gitnexus_bootstrap_enabled and bootstrap_path.is_dir():
        return bootstrap_path

    resolved = Path(__file__).resolve()
    candidate = resolved.parents[2]
    # In container builds, app code may live at /app/app/main.py where parents[2] is "/".
    if str(candidate) == candidate.root or not candidate.name:
        fallback = resolved.parents[1]
        if fallback.name:
            return fallback
    return candidate


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


def normalized_ui_mode(value: QueryRequest | str | None) -> str:
    if isinstance(value, QueryRequest):
        raw = value.ui_mode or value.mode
    else:
        raw = value
    return str(raw or "").strip().lower()


def infer_diagram_type(text: str | None) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return "systemArchitecture"
    if "retrieval" in normalized or "rag" in normalized:
        return "retrievalFlow"
    if "audit diagram" in normalized or "risk map" in normalized or "audit view" in normalized:
        return "auditDiagram"
    if "execution" in normalized or "pipeline" in normalized or "runtime flow" in normalized:
        return "executionPipeline"
    if "data flow" in normalized or "lineage" in normalized:
        return "dataFlow"
    if "dependency" in normalized or "module graph" in normalized or "import" in normalized:
        return "dependencyGraph"
    if "build" in normalized or "runtime environment" in normalized or "compiler" in normalized:
        return "buildRuntime"
    return "systemArchitecture"


def normalize_diagram_type(value: str | None, question: str = "") -> str:
    raw = str(value or "").strip()
    if raw in DIRECT_DIAGRAM_TYPES:
        return raw
    return infer_diagram_type(question)


def clip_prompt_text(text: str, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max(limit, 1):
        return raw
    truncated = raw[: max(limit - 16, 1)].rstrip()
    last_newline = truncated.rfind("\n")
    if last_newline >= max(limit // 3, 80):
        truncated = truncated[:last_newline].rstrip()
    return f"{truncated}\n...[truncated]"


@lru_cache(maxsize=1)
def repo_context_inventory() -> dict[str, Any]:
    root = repo_root_path()
    skip_parts = {".git", ".gitnexus", ".venv", ".venv-audit", "__pycache__"}
    top_level_entries = [
        f"{path.name}/" if path.is_dir() else path.name
        for path in sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        if not path.name.startswith(".")
    ]
    extension_counts: Counter[str] = Counter()
    text_paths: set[str] = set()
    script_files: list[str] = []
    config_files: list[str] = []
    doc_files: list[str] = []
    total_file_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(root)).replace("\\", "/")
        rel_parts = set(Path(rel_path).parts)
        if rel_parts & skip_parts:
            continue
        total_file_count += 1
        suffix = path.suffix.lower()
        extension_counts[suffix or "[no_ext]"] += 1
        lower_rel = rel_path.lower()
        if suffix == ".sh":
            script_files.append(rel_path)
        if rel_path.startswith("conf/") or rel_path.startswith("etc/"):
            config_files.append(rel_path)
        if lower_rel.endswith(".md") or rel_path.startswith("docs/"):
            doc_files.append(rel_path)
        if suffix in TEXTUAL_CONTEXT_SUFFIXES or path.name.lower() in IMPORTANT_TEXT_FILE_NAMES:
            text_paths.add(rel_path)

    symbol_index = repo_symbol_index()
    source_rankings: list[dict[str, Any]] = []
    for file_path, definitions in (symbol_index.get("by_file", {}) or {}).items():
        symbols = [str(item.get("symbol", "")).strip() for item in definitions if str(item.get("symbol", "")).strip()]
        signatures = [str(item.get("signature", "")).strip() for item in definitions if str(item.get("signature", "")).strip()]
        source_rankings.append(
            {
                "file_path": file_path,
                "definition_count": len(definitions),
                "line_count": len(repo_file_lines(file_path)),
                "symbols": symbols[:8],
                "signatures": signatures[:8],
            }
        )
    source_rankings.sort(
        key=lambda row: (int(row.get("definition_count") or 0), int(row.get("line_count") or 0), str(row.get("file_path") or "")),
        reverse=True,
    )

    return {
        "repo_name": root.name,
        "repo_root": str(root),
        "commit_hash": repo_commit_short(),
        "top_level_entries": top_level_entries,
        "total_file_count": total_file_count,
        "extension_counts": dict(extension_counts),
        "text_paths": sorted(text_paths),
        "script_files": sorted(script_files),
        "config_files": sorted(config_files),
        "doc_files": sorted(doc_files),
        "source_rankings": source_rankings,
    }


def build_repo_overview_text() -> str:
    inventory = repo_context_inventory()
    extension_counts = inventory.get("extension_counts", {}) if isinstance(inventory, dict) else {}
    ranked_extensions = sorted(
        ((str(ext), int(count)) for ext, count in extension_counts.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[:8]
    source_rankings = inventory.get("source_rankings", []) if isinstance(inventory, dict) else []
    lines = [
        f"Repository: {inventory.get('repo_name', 'unknown')}",
        f"Repo root: {inventory.get('repo_root', '')}",
        f"Commit: {inventory.get('commit_hash') or 'unknown'}",
        f"Top-level entries: {', '.join(inventory.get('top_level_entries', [])[:16]) or 'n/a'}",
        f"Total files discovered: {inventory.get('total_file_count', 0)}",
        "File type counts: "
        + (
            ", ".join(f"{ext}={count}" for ext, count in ranked_extensions)
            if ranked_extensions
            else "n/a"
        ),
    ]
    script_files = inventory.get("script_files", []) if isinstance(inventory, dict) else []
    if script_files:
        lines.append(f"Representative scripts: {', '.join(script_files[:6])}")
    config_files = inventory.get("config_files", []) if isinstance(inventory, dict) else []
    if config_files:
        lines.append(f"Representative config/data files: {', '.join(config_files[:6])}")
    if source_rankings:
        lines.append("Representative source files by definition density:")
        for row in source_rankings[:6]:
            symbols = ", ".join(row.get("symbols", [])[:3]) or "n/a"
            lines.append(
                f"- {row.get('file_path', 'unknown')} ({row.get('definition_count', 0)} defs; symbols: {symbols})"
            )
    return "\n".join(lines)


def rank_context_paths_for_question(question: str) -> list[str]:
    inventory = repo_context_inventory()
    terms = {
        token.lower()
        for token in IDENTIFIER_PATTERN.findall(question)
        if len(token) >= 3
    }
    terms.update(
        token.lower()
        for token in WORD_PATTERN.findall(question)
        if len(token) >= 4
    )
    if not terms:
        return []

    ranked: list[tuple[float, str]] = []
    source_by_path = {
        str(row.get("file_path", "")): row
        for row in inventory.get("source_rankings", [])
        if str(row.get("file_path", ""))
    }
    for path in inventory.get("text_paths", []):
        lower_path = str(path).lower()
        row = source_by_path.get(str(path), {})
        symbol_blob = " ".join(str(item) for item in row.get("symbols", []))
        signature_blob = " ".join(str(item) for item in row.get("signatures", []))
        haystack = f"{lower_path} {symbol_blob.lower()} {signature_blob.lower()}".strip()
        score = 0.0
        for term in terms:
            if term in lower_path:
                score += 3.0
            if f" {term}" in f" {symbol_blob.lower()}":
                score += 4.0
            if term in signature_blob.lower():
                score += 2.0
        if score > 0:
            score += min(float(row.get("definition_count") or 0), 6.0) * 0.15
            ranked.append((score, str(path)))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for _, path in ranked]


def choose_direct_context_paths(question: str, ui_mode: str, diagram_type: str | None) -> list[str]:
    inventory = repo_context_inventory()
    available_paths = set(inventory.get("text_paths", []))
    selected: list[str] = []

    def add_path(path: str) -> None:
        normalized = str(path or "").strip().replace("\\", "/")
        if not normalized or normalized not in available_paths or normalized in selected:
            return
        selected.append(normalized)

    baseline_paths = [
        "README.md",
        "Makefile",
        "run_all_hazard.sh",
        "backend/Procfile",
        "backend/Dockerfile",
        "backend/requirements.txt",
    ]
    for path in baseline_paths:
        add_path(path)

    source_rankings = inventory.get("source_rankings", []) if isinstance(inventory, dict) else []
    source_paths = [str(row.get("file_path", "")) for row in source_rankings if str(row.get("file_path", ""))]
    script_files = list(inventory.get("script_files", [])) if isinstance(inventory, dict) else []
    config_files = [path for path in inventory.get("config_files", []) if path in available_paths] if isinstance(inventory, dict) else []

    if ui_mode == "audit":
        for path in script_files[:4]:
            add_path(path)
        for path in source_paths[:5]:
            add_path(path)
    else:
        resolved_type = normalize_diagram_type(diagram_type, question)
        if resolved_type == "executionPipeline":
            for path in ["run_all_hazard.sh", "Makefile"]:
                add_path(path)
            for path in script_files[:6]:
                add_path(path)
            for path in source_paths[:2]:
                add_path(path)
        elif resolved_type == "dataFlow":
            for path in ["run_all_hazard.sh", "README.md"]:
                add_path(path)
            for path in config_files[:4]:
                add_path(path)
            for path in script_files[:3]:
                add_path(path)
            for path in source_paths[:3]:
                add_path(path)
        elif resolved_type == "dependencyGraph":
            add_path("Makefile")
            for path in source_paths[:6]:
                add_path(path)
            for path in script_files[:2]:
                add_path(path)
        elif resolved_type == "buildRuntime":
            for path in [
                "Makefile",
                "run_all_hazard.sh",
                "backend/Dockerfile",
                "backend/Procfile",
                "backend/requirements.txt",
            ]:
                add_path(path)
            for path in script_files[:3]:
                add_path(path)
        else:
            for path in ["run_all_hazard.sh", "Makefile"]:
                add_path(path)
            for path in script_files[:3]:
                add_path(path)
            for path in source_paths[:4]:
                add_path(path)

    for path in rank_context_paths_for_question(question)[:5]:
        add_path(path)

    return selected[:DIRECT_CONTEXT_MAX_FILES]


def build_direct_repo_source(file_path: str) -> dict[str, Any] | None:
    normalized_path = str(file_path or "").strip().replace("\\", "/")
    if not normalized_path:
        return None
    lines = repo_file_lines(normalized_path)
    if not lines:
        return None

    definitions = (repo_symbol_index().get("by_file", {}) or {}).get(normalized_path, [])
    prompt_excerpt = ""
    line_start = 1
    line_end = min(len(lines), DIRECT_FILE_PREVIEW_LINES)
    if definitions and Path(normalized_path).suffix.lower() in FORTRAN_EXTENSIONS:
        segments: list[str] = []
        line_start = safe_int(definitions[0].get("line_start")) or 1
        line_end = line_start
        for definition in definitions[:DIRECT_SYMBOL_PREVIEW_LIMIT]:
            seg_start = safe_int(definition.get("line_start")) or 1
            seg_end = safe_int(definition.get("line_end")) or seg_start
            seg_end = min(seg_end, seg_start + 18)
            snippet = file_snippet(normalized_path, seg_start, seg_end)
            if not snippet:
                continue
            header = f"{definition.get('kind', 'symbol')} {definition.get('symbol', '')} ({normalized_path}:{seg_start}-{seg_end})"
            segments.append(f"{header}\n{snippet}")
            line_end = max(line_end, seg_end)
        prompt_excerpt = "\n\n".join(segments).strip()
    if not prompt_excerpt:
        prompt_excerpt = "\n".join(lines[:line_end]).strip()
    if not prompt_excerpt:
        return None

    prompt_excerpt = clip_prompt_text(prompt_excerpt, DIRECT_SOURCE_PROMPT_CHARS)
    snippet = clip_prompt_text(prompt_excerpt, DIRECT_SOURCE_SNIPPET_CHARS)
    return {
        "file_path": normalized_path,
        "line_start": line_start,
        "line_end": max(line_start, line_end),
        "prompt_excerpt": prompt_excerpt,
        "snippet": snippet,
        "source_type": "repo",
    }


def build_direct_upload_sources(uploaded_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not uploaded_files:
        return []
    metadata_chunks, _ = build_attachment_chunks(uploaded_files, source_type="temp-upload")
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for chunk in metadata_chunks:
        file_path = str(chunk.get("file_path", "")).strip()
        if not file_path or file_path in seen_paths:
            continue
        prompt_excerpt = clip_prompt_text(str(chunk.get("chunk_text", "")), DIRECT_UPLOAD_PROMPT_CHARS)
        if not prompt_excerpt:
            continue
        seen_paths.add(file_path)
        selected.append(
            {
                "file_path": file_path,
                "line_start": safe_int(chunk.get("line_start")) or 1,
                "line_end": safe_int(chunk.get("line_end")) or (safe_int(chunk.get("line_start")) or 1),
                "prompt_excerpt": prompt_excerpt,
                "snippet": clip_prompt_text(prompt_excerpt, DIRECT_SOURCE_SNIPPET_CHARS),
                "source_type": str(chunk.get("source_type", "temp-upload") or "temp-upload"),
            }
        )
        if len(selected) >= 4:
            break
    return selected


def build_direct_mode_context(
    question: str,
    ui_mode: str,
    diagram_type: str | None = None,
    uploaded_files: list[dict[str, Any]] | None = None,
) -> tuple[str, list[Citation], dict[str, Any]]:
    overview = build_repo_overview_text()
    selected_paths = choose_direct_context_paths(question, ui_mode, diagram_type)
    repo_sources = [item for item in (build_direct_repo_source(path) for path in selected_paths) if item]
    upload_sources = build_direct_upload_sources(uploaded_files or [])
    all_sources = repo_sources + upload_sources

    sections = [overview, "", "Selected repository excerpts:"]
    citations: list[Citation] = []
    included_sources: list[dict[str, Any]] = []
    context_chars = sum(len(part) for part in sections)
    for index, source in enumerate(all_sources, start=1):
        block = (
            f"[Source {index}] {source['file_path']}:{source['line_start']}-{source['line_end']}\n"
            f"{source['prompt_excerpt']}"
        )
        if included_sources and context_chars + len(block) + 2 > DIRECT_CONTEXT_MAX_CHARS:
            break
        sections.extend(["", block])
        context_chars += len(block) + 2
        included_sources.append(source)
        citations.append(
            Citation(
                file_path=str(source["file_path"]),
                line_start=int(source["line_start"]),
                line_end=int(source["line_end"]),
                score=round(max(0.58, 0.88 - ((len(included_sources) - 1) * 0.05)), 4),
                source_type=str(source["source_type"]),
                snippet=str(source["snippet"]),
            )
        )

    context = "\n".join(sections).strip()
    metadata = {
        "ui_mode": ui_mode,
        "diagram_type": normalize_diagram_type(diagram_type, question) if ui_mode == "diagrams" else "",
        "selected_source_count": len(included_sources),
        "selected_sources": [
            {
                "file_path": source["file_path"],
                "line_start": source["line_start"],
                "line_end": source["line_end"],
                "source_type": source["source_type"],
            }
            for source in included_sources
        ],
        "repo_overview": {
            "repo_name": repo_context_inventory().get("repo_name"),
            "commit_hash": repo_context_inventory().get("commit_hash"),
            "total_file_count": repo_context_inventory().get("total_file_count"),
            "top_level_entries": repo_context_inventory().get("top_level_entries", [])[:16],
        },
    }
    return context, citations, metadata


def build_direct_audit_system_prompt() -> str:
    return (
        "You are a principal software auditor reviewing a repository from a deterministic repo scan and curated file excerpts.\n"
        "Use only the repository context and uploaded file excerpts provided in the user message.\n"
        "Do not claim to have executed code, run tests, or inspected files that are not present in the provided context.\n"
        "Separate verified evidence from inference. Mark uncertain claims explicitly as Hypothesis and include a verification step.\n"
        "Prefer incremental fixes, operational clarity, and defensible engineering judgment. Do not recommend rewrites.\n\n"
        "Return markdown in this exact section order:\n"
        "Overview\n"
        "Key Findings\n"
        "Evidence (files/lines)\n"
        "Recommendations\n"
        "Next Actions\n\n"
        "Audit requirements:\n"
        "- Overview: concise system map, current repository posture, and major runtime/build boundaries.\n"
        "- Key Findings: 3-7 findings ordered by severity. For each finding include Priority (High|Medium|Low), Why it matters, Evidence, and a recommended fix.\n"
        "- Evidence (files/lines): list concrete file path and line anchors from the provided context.\n"
        "- Recommendations: safe, sequenced improvements with no big-bang rewrite advice.\n"
        "- Next Actions: immediate follow-ups and, if warranted, deeper-pass targets.\n"
        "- Keep the tone direct, concise, and engineering-focused."
    )


def direct_diagram_config(diagram_type: str) -> dict[str, Any]:
    configs = {
        "systemArchitecture": {
            "title": "system architecture diagram",
            "chart_type": "flowchart",
            "orientation": "TD",
            "goal": "Show high-level subsystems, major boundaries, and directional relationships.",
            "lane_templates": ["Orchestration", "Regional Scripts", "Core Binaries", "Inputs / Config", "Outputs / Logs"],
        },
        "executionPipeline": {
            "title": "execution pipeline diagram",
            "chart_type": "flowchart",
            "orientation": "LR",
            "goal": "Show runtime order, branching, and stage transitions from entrypoint to outputs.",
            "lane_templates": ["Entrypoint", "Build", "Execution", "Outputs"],
        },
        "retrievalFlow": {
            "title": "retrieval flow diagram",
            "chart_type": "flowchart",
            "orientation": "TD",
            "goal": "Show user input, frontend handling, backend routing, retrieval stages, answer generation, and response delivery.",
            "lane_templates": ["User", "Frontend", "Backend", "Retrieval", "LLM / Response"],
        },
        "dataFlow": {
            "title": "data flow diagram",
            "chart_type": "flowchart",
            "orientation": "TD",
            "goal": "Show inputs, transformations, and outputs with clear lineage.",
            "lane_templates": ["Inputs", "Transforms", "Storage", "Outputs"],
        },
        "dependencyGraph": {
            "title": "dependency graph",
            "chart_type": "graph",
            "orientation": "TD",
            "goal": "Show the most central modules/files and their directional dependencies.",
            "lane_templates": ["Scripts", "Sources", "Artifacts"],
        },
        "buildRuntime": {
            "title": "build and runtime environment diagram",
            "chart_type": "flowchart",
            "orientation": "TD",
            "goal": "Show compile-time stages, runtime dependencies, and produced artifacts.",
            "lane_templates": ["Build Inputs", "Build", "Runtime", "Artifacts"],
        },
        "auditDiagram": {
            "title": "audit diagram",
            "chart_type": "flowchart",
            "orientation": "TD",
            "goal": "Show major repository areas, risk concentrations, and the highest-signal audit paths from verified context.",
            "lane_templates": ["Entrypoints", "Core Components", "Risk Areas", "Evidence", "Actions"],
        },
    }
    resolved_type = normalize_diagram_type(diagram_type)
    return configs.get(resolved_type, configs["systemArchitecture"])


def build_direct_diagram_system_prompt(diagram_type: str) -> str:
    config = direct_diagram_config(diagram_type)
    return (
        "You are a principal software architect generating repository diagrams from a deterministic repo scan and curated file excerpts.\n"
        "Use only the provided repository context. Never invent components, files, edges, or execution stages.\n"
        "If evidence is insufficient, omit the element or label it '(hypothesis)' rather than fabricating detail.\n"
        "Return only JSON. No markdown fences. No prose before or after the JSON object.\n"
        f"The diagram requested is a {config['title']}.\n"
        f"Primary goal: {config['goal']}\n\n"
        "JSON schema:\n"
        '{\n'
        '  "title": "short title",\n'
        '  "orientation": "TD or LR",\n'
        '  "lanes": [{"id": "lane_id", "label": "Lane Label"}],\n'
        '  "nodes": [{"id": "node_id", "label": "Node Label", "lane": "lane_id"}],\n'
        '  "edges": [{"from": "node_id", "to": "node_id", "label": "optional edge label"}]\n'
        '}\n\n'
        "Diagram rules:\n"
        f"- Prefer 3-{DIRECT_DIAGRAM_MAX_LANES} lanes and 6-{DIRECT_DIAGRAM_MAX_NODES} nodes.\n"
        "- Lanes are swimlanes. They will be converted into Mermaid subgraph blocks.\n"
        f"- Recommended lane titles for this diagram: {', '.join(config['lane_templates'])}.\n"
        "- Use concise engineering labels that map to real repository paths, scripts, modules, directories, or runtime stages present in the context.\n"
        "- Avoid placeholder names such as Component A, Service B, or Module X.\n"
        "- Compress repeated or similar files into one representative node when possible.\n"
        "- Show only high-signal edges. Keep the diagram deterministic and connected.\n"
        "- If the user asks for a flow or user journey but the repo only exposes system/operator flows, model the nearest verified flow and keep labels concrete."
    )


def build_direct_diagram_fallback_prompt(diagram_type: str) -> str:
    config = direct_diagram_config(diagram_type)
    return (
        "You are a principal software architect generating repository diagrams from a deterministic repo scan and curated file excerpts.\n"
        "Use only the provided repository context. Never invent components, files, edges, or execution stages.\n"
        "If evidence is insufficient, omit the element or label it '(hypothesis)' rather than fabricating detail.\n"
        "Return only valid Mermaid. No markdown fences. No prose before or after the Mermaid output.\n"
        f"The Mermaid output must begin with: {config['chart_type']} {config['orientation']}\n"
        f"Primary goal: {config['goal']}\n"
        "- Use subgraph blocks as swimlanes.\n"
        f"- Prefer 3-{DIRECT_DIAGRAM_MAX_LANES} lanes and 6-{DIRECT_DIAGRAM_MAX_NODES} nodes.\n"
        "- Keep node labels concise and connected by only high-signal edges."
    )


MERMAID_VALID_PREFIXES = (
    "flowchart td",
    "flowchart lr",
    "graph td",
    "graph lr",
    "sequencediagram",
    "classdiagram",
)
DIRECT_DIAGRAM_FAILURE_MERMAID = "flowchart TD\nA[Diagram generation failed]"


def extract_json_object_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "{}"
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def mermaid_safe_id(value: str, *, prefix: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return f"{prefix}_{cleaned}"


def mermaid_safe_label(value: str) -> str:
    return " ".join(str(value or "").replace('"', "'").split()).strip() or "Unnamed"


def normalize_direct_diagram_spec(raw_spec: dict[str, Any], diagram_type: str) -> dict[str, Any]:
    config = direct_diagram_config(diagram_type)
    orientation = str(raw_spec.get("orientation") or config["orientation"]).strip().upper()
    if orientation not in {"TD", "LR"}:
        orientation = str(config["orientation"])

    raw_lanes = raw_spec.get("lanes") if isinstance(raw_spec.get("lanes"), list) else []
    lanes: list[dict[str, str]] = []
    lane_ids: set[str] = set()
    for idx, lane in enumerate(raw_lanes[:DIRECT_DIAGRAM_MAX_LANES], start=1):
        if not isinstance(lane, dict):
            continue
        label = mermaid_safe_label(lane.get("label") or lane.get("id") or f"Lane {idx}")
        lane_id = mermaid_safe_id(str(lane.get("id") or label), prefix="lane")
        if lane_id in lane_ids:
            continue
        lane_ids.add(lane_id)
        lanes.append({"id": lane_id, "label": label})
    if len(lanes) < 2:
        fallback_labels = list(config.get("lane_templates", []))[: max(2, len(lanes) or 2)]
        lanes = [{"id": mermaid_safe_id(label, prefix="lane"), "label": label} for label in fallback_labels]
        lane_ids = {lane["id"] for lane in lanes}

    raw_nodes = raw_spec.get("nodes") if isinstance(raw_spec.get("nodes"), list) else []
    lane_lookup = {lane["id"]: lane["id"] for lane in lanes}
    lane_lookup.update({mermaid_safe_id(lane["label"], prefix="lane"): lane["id"] for lane in lanes})
    nodes: list[dict[str, str]] = []
    node_ids: set[str] = set()
    default_lane_id = lanes[0]["id"]
    for idx, node in enumerate(raw_nodes[:DIRECT_DIAGRAM_MAX_NODES], start=1):
        if not isinstance(node, dict):
            continue
        label = mermaid_safe_label(node.get("label") or node.get("id") or f"Node {idx}")
        node_id = mermaid_safe_id(str(node.get("id") or label), prefix="node")
        if node_id in node_ids:
            continue
        requested_lane = mermaid_safe_id(str(node.get("lane") or default_lane_id), prefix="lane")
        lane_id = lane_lookup.get(requested_lane, default_lane_id)
        nodes.append({"id": node_id, "label": label, "lane": lane_id})
        node_ids.add(node_id)
    if len(nodes) < 3:
        raise ValueError("diagram spec did not produce enough nodes")

    raw_edges = raw_spec.get("edges") if isinstance(raw_spec.get("edges"), list) else []
    node_lookup = {node["id"]: node["id"] for node in nodes}
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in raw_edges[:DIRECT_DIAGRAM_MAX_EDGES]:
        if not isinstance(edge, dict):
            continue
        source = node_lookup.get(mermaid_safe_id(str(edge.get("from") or ""), prefix="node"))
        target = node_lookup.get(mermaid_safe_id(str(edge.get("to") or ""), prefix="node"))
        if not source or not target or source == target:
            continue
        label = mermaid_safe_label(edge.get("label") or "")
        edge_key = (source, target, label)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append({"from": source, "to": target, "label": label})
    if len(edges) < 2:
        raise ValueError("diagram spec did not produce enough valid edges")

    return {
        "title": mermaid_safe_label(raw_spec.get("title") or config["title"]),
        "chart_type": config["chart_type"],
        "orientation": orientation,
        "lanes": lanes,
        "nodes": nodes,
        "edges": edges,
    }


def build_mermaid_from_direct_diagram_spec(spec: dict[str, Any]) -> str:
    lanes = list(spec.get("lanes", []))
    nodes = list(spec.get("nodes", []))
    edges = list(spec.get("edges", []))
    lines = [f"{spec.get('chart_type', 'flowchart')} {spec.get('orientation', 'TD')}"]
    nodes_by_lane: dict[str, list[dict[str, str]]] = defaultdict(list)
    for node in nodes:
        nodes_by_lane[str(node.get("lane") or "")].append(node)
    for lane in lanes:
        lane_id = str(lane.get("id") or "")
        lane_label = mermaid_safe_label(lane.get("label") or lane_id)
        lines.append("")
        lines.append(f'    subgraph {lane_id}["{lane_label}"]')
        lines.append("        direction TB")
        for node in nodes_by_lane.get(lane_id, []):
            lines.append(f'        {node["id"]}["{mermaid_safe_label(node.get("label") or node["id"])}"]')
        lines.append("    end")
    lines.append("")
    for edge in edges:
        label = mermaid_safe_label(edge.get("label") or "")
        if label:
            lines.append(f'    {edge["from"]} -->|{label}| {edge["to"]}')
        else:
            lines.append(f'    {edge["from"]} --> {edge["to"]}')
    return "\n".join(lines).strip()


def parse_direct_diagram_spec(text: str, diagram_type: str) -> dict[str, Any]:
    raw = json.loads(extract_json_object_text(text))
    if not isinstance(raw, dict):
        raise ValueError("diagram response was not a JSON object")
    return normalize_direct_diagram_spec(raw, diagram_type)


def normalize_direct_diagram_output(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return DIRECT_DIAGRAM_FAILURE_MERMAID
    fenced = re.search(r"```(?:mermaid)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    cleaned = fenced.group(1).strip() if fenced else raw
    cleaned = cleaned.replace("```mermaid", "").replace("```", "").strip()
    lowered = cleaned.lower()
    if any(lowered.startswith(prefix) for prefix in MERMAID_VALID_PREFIXES):
        return cleaned
    return DIRECT_DIAGRAM_FAILURE_MERMAID


def record_direct_completion_usage(
    completion: Any,
    telemetry: RagRequestTelemetry | None,
    elapsed_ms: float,
) -> None:
    if not telemetry:
        return
    usage = parse_openai_usage(getattr(completion, "usage", None))
    telemetry.record_llm(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_tokens") or 0),
        latency_ms=elapsed_ms,
        model_name=str(getattr(completion, "model", settings.openai_chat_model)),
    )


def generate_direct_diagram_answer(
    *,
    question: str,
    context: str,
    telemetry: RagRequestTelemetry | None = None,
    diagram_type: str | None = None,
) -> str:
    resolved_type = normalize_diagram_type(diagram_type, question)
    client = get_openai_client()
    started = time.perf_counter()
    completion = call_with_retries(
        "openai direct diagram completion",
        lambda: client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": build_direct_diagram_system_prompt(resolved_type)},
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{question}\n\n"
                        f"Repository context:\n{context}"
                    ),
                },
            ],
        ),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    record_direct_completion_usage(completion, telemetry, elapsed_ms)
    content = completion.choices[0].message.content or ""
    try:
        return normalize_direct_diagram_output(
            build_mermaid_from_direct_diagram_spec(parse_direct_diagram_spec(content, resolved_type))
        )
    except Exception as exc:
        logger.warning("diagram json generation failed, falling back to raw Mermaid: %s", exc)

    fallback_started = time.perf_counter()
    fallback_completion = call_with_retries(
        "openai direct diagram fallback",
        lambda: client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": build_direct_diagram_fallback_prompt(resolved_type)},
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{question}\n\n"
                        f"Repository context:\n{context}"
                    ),
                },
            ],
        ),
    )
    fallback_elapsed_ms = (time.perf_counter() - fallback_started) * 1000.0
    record_direct_completion_usage(fallback_completion, telemetry, fallback_elapsed_ms)
    fallback_content = fallback_completion.choices[0].message.content
    return normalize_direct_diagram_output(fallback_content or "")


def generate_direct_mode_answer(
    *,
    ui_mode: str,
    question: str,
    context: str,
    telemetry: RagRequestTelemetry | None = None,
    diagram_type: str | None = None,
) -> str:
    resolved_ui_mode = normalized_ui_mode(ui_mode)
    if resolved_ui_mode not in DIRECT_UI_MODES:
        raise ValueError(f"Unsupported direct mode: {ui_mode}")

    if resolved_ui_mode == "diagrams":
        return generate_direct_diagram_answer(
            question=question,
            context=context,
            telemetry=telemetry,
            diagram_type=diagram_type,
        )

    client = get_openai_client()
    started = time.perf_counter()
    completion = call_with_retries(
        "openai direct mode completion",
        lambda: client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.15,
            messages=[
                {"role": "system", "content": build_direct_audit_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{question}\n\n"
                        f"Repository context:\n{context}"
                    ),
                },
            ],
        ),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    record_direct_completion_usage(completion, telemetry, elapsed_ms)
    content = completion.choices[0].message.content
    return content.strip() if content else "No answer generated."


def build_direct_mode_debug_payload(
    *,
    context: str,
    metadata: dict[str, Any],
    latency_ms: float | None = None,
) -> dict[str, Any]:
    payload = {
        "route_debug": {
            "route": "direct_agent",
            "steps": ["repo_scan", "source_selection", "llm"],
        },
        "hybrid_debug": {},
        "direct_context": metadata,
        "final_context_preview": context[:3000],
        "context_token_estimate": token_count(context),
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    return payload


def execute_direct_ui_mode_request(
    payload: QueryRequest,
    telemetry: RagRequestTelemetry,
    uploaded_files: list[dict[str, Any]] | None = None,
) -> QueryResponse:
    ui_mode = normalized_ui_mode(payload)
    diagram_type = normalize_diagram_type(payload.diagram_type, payload.question) if ui_mode == "diagrams" else ""
    context_started = time.perf_counter()
    context, citations, metadata = build_direct_mode_context(
        question=payload.question,
        ui_mode=ui_mode,
        diagram_type=diagram_type,
        uploaded_files=uploaded_files or [],
    )
    context_elapsed_ms = (time.perf_counter() - context_started) * 1000.0
    metadata["diagram_type"] = diagram_type
    selected_files = {
        citation.file_path
        for citation in citations
        if citation.file_path and not str(citation.file_path).startswith("uploaded/")
    }
    telemetry.record_counts(
        retrieved_file_count=len(selected_files),
        retrieved_chunk_count=len(citations),
        selected_chunk_count=len(citations),
    )
    telemetry.mark_retrieval_complete()
    answer = generate_direct_mode_answer(
        ui_mode=ui_mode,
        question=payload.question,
        context=context,
        telemetry=telemetry,
        diagram_type=diagram_type,
    )
    telemetry.record_postprocess(context_elapsed_ms)
    telemetry.mark_success(answer)
    response_telemetry = finalize_and_persist_telemetry(telemetry)
    debug_payload = {}
    if payload.debug:
        debug_payload = build_direct_mode_debug_payload(
            context=context,
            metadata=metadata,
            latency_ms=(time.perf_counter() - telemetry._started_perf) * 1000.0,
        )
        debug_payload["telemetry"] = response_telemetry
    response_type = "diagram" if ui_mode == "diagrams" else "text"
    response_format = "mermaid" if ui_mode == "diagrams" else "markdown"
    return QueryResponse(
        answer=answer,
        type=response_type,
        format=response_format,
        content=answer,
        citations=citations,
        graph={},
        evidence=[],
        evidence_strength={
            "label": "Unknown",
            "score": None,
            "reason": f"{ui_mode.capitalize()} mode uses direct repository scan context and bypasses retrieval scoring.",
            "metrics": {
                "mode": ui_mode,
                "diagram_type": diagram_type,
                "selected_sources": len(citations),
            },
        },
        debug=debug_payload,
        telemetry=response_telemetry,
    )


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
) -> tuple[list[dict[str, Any]], bool, dict[str, float]]:
    key = pinecone_query_cache_key(query_text, namespace, candidate_top_k, pinecone_filter)
    cached = query_cache_get(key)
    if cached is not None:
        return cached, True, {"read_units": 0.0, "write_units": 0.0, "query_count": 0.0, "latency_ms": 0.0}

    query_kwargs: dict[str, Any] = {
        "vector": question_vector,
        "top_k": candidate_top_k,
        "include_metadata": True,
        "namespace": namespace,
    }
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter
    started = time.perf_counter()
    results = call_with_retries("pinecone query", lambda: index.query(**query_kwargs))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    usage = parse_pinecone_usage(results)
    query_matches = normalize_cached_matches(normalize_matches(results))
    query_cache_put(key, query_matches)
    return query_matches, False, {
        "read_units": float(usage.get("read_units") or 0.0),
        "write_units": float(usage.get("write_units") or 0.0),
        "query_count": 1.0,
        "latency_ms": elapsed_ms,
    }


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
    include_tests: bool = True,
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
    if not include_tests and is_test_file_path(file_path):
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
    include_tests: bool = True,
    telemetry: RagRequestTelemetry | None = None,
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
    normalized_candidate_files, excluded_candidate_tests, _ = filter_paths_by_test_policy(
        candidate_files,
        include_tests=include_tests,
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
    excluded_test_candidates = max(int(excluded_candidate_tests), 0)
    for query_text in queries:
        question_vector, embedding_debug = embed_question(query_text, telemetry=telemetry)
        embed_ms += float(embedding_debug.get("latency_ms") or 0.0)
        if not question_vector:
            subquery_counts.append({"query": query_text, "matches": 0})
            continue
        ensure_embedding_dimension_matches_index(len(question_vector), index_dim=index_dim)

        query_matches: list = []
        for namespace in target_namespaces:
            query_matches, from_cache, pinecone_usage = query_index_with_cache(
                index=index,
                query_text=query_text,
                question_vector=question_vector,
                candidate_top_k=candidate_top_k,
                namespace=namespace,
                pinecone_filter=pinecone_filter,
            )
            pinecone_ms += float(pinecone_usage.get("latency_ms") or 0.0)
            if from_cache:
                cache_hits += 1
            else:
                cache_misses += 1
                if telemetry:
                    telemetry.record_pinecone(
                        read_units=float(pinecone_usage.get("read_units") or 0.0),
                        write_units=float(pinecone_usage.get("write_units") or 0.0),
                        query_count=int(pinecone_usage.get("query_count") or 0),
                        latency_ms=float(pinecone_usage.get("latency_ms") or 0.0),
                    )
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
            if file_path and not include_tests and is_test_file_path(file_path):
                excluded_test_candidates += 1
                continue
            if not metadata_matches_filters(
                metadata,
                path_prefix=path_prefix,
                language=language,
                source_type=source_type_filter,
                candidate_files=normalized_candidate_files,
                repo_name=repo_name,
                include_tests=include_tests,
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
                "test_filter": {
                    "include_tests": bool(include_tests),
                    "excluded_test_candidates": excluded_test_candidates,
                    "final_test_candidates": 0,
                },
            },
        )

    rerank_started = time.perf_counter()
    reranked, debug_candidates = rerank_matches(question=question, matches=matches, top_k=top_k)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
    if telemetry:
        telemetry.record_rerank(input_tokens=token_count(question), latency_ms=rerank_ms)
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    final_test_candidates = sum(1 for citation in citations if is_test_file_path(citation.file_path))
    debug = {
        "candidates": debug_candidates,
        "subqueries": subquery_counts,
        "timings_ms": {
            "embed": round(embed_ms, 2),
            "pinecone_query": round(pinecone_ms, 2),
            "rerank": round(rerank_ms, 2),
        },
        "cache": {"hits": cache_hits, "misses": cache_misses},
        "test_filter": {
            "include_tests": bool(include_tests),
            "excluded_test_candidates": excluded_test_candidates,
            "final_test_candidates": final_test_candidates,
        },
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
    include_tests: bool = True,
    telemetry: RagRequestTelemetry | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    if not uploaded_files:
        return [], [], {"candidates": [], "subqueries": []}

    metadata_chunks, _ = build_attachment_chunks(uploaded_files, source_type="temp-upload")
    if not metadata_chunks:
        return [], [], {"candidates": [], "subqueries": []}

    normalized_candidate_files, excluded_candidate_tests, _ = filter_paths_by_test_policy(
        candidate_files,
        include_tests=include_tests,
    )
    matches: list[dict] = []
    subquery_counts: list[dict[str, Any]] = []
    excluded_test_candidates = max(int(excluded_candidate_tests), 0)
    queries = retrieval_queries or [question]
    embed_ms = 0.0
    for query_text in queries:
        question_vector, embedding_debug = embed_question(query_text, telemetry=telemetry)
        embed_ms += float(embedding_debug.get("latency_ms") or 0.0)
        if not question_vector:
            subquery_counts.append({"query": query_text, "matches": 0})
            continue

        count = 0
        for metadata_batch in batched(metadata_chunks, UPLOAD_EMBED_BATCH_SIZE):
            batch_started = time.perf_counter()
            embeddings = embed_texts(
                [metadata_chunk_text(metadata) for metadata in metadata_batch],
                telemetry=telemetry,
            )
            embed_ms += (time.perf_counter() - batch_started) * 1000.0
            for metadata, embedding in zip(metadata_batch, embeddings, strict=True):
                scoped = metadata.copy()
                scoped["query_used"] = query_text
                if not include_tests and is_test_file_path(scoped.get("file_path")):
                    excluded_test_candidates += 1
                    continue
                if not metadata_matches_filters(
                    scoped,
                    path_prefix=path_prefix,
                    language=language,
                    source_type=source_type_filter,
                    candidate_files=normalized_candidate_files,
                    repo_name=repo_name,
                    include_tests=include_tests,
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
        return (
            [],
            [],
            {
                "candidates": [],
                "subqueries": subquery_counts,
                "timings_ms": {"embed": round(embed_ms, 2), "rerank": 0.0},
                "test_filter": {
                    "include_tests": bool(include_tests),
                    "excluded_test_candidates": excluded_test_candidates,
                    "final_test_candidates": 0,
                },
            },
        )

    rerank_started = time.perf_counter()
    reranked, debug_candidates = rerank_matches(question=question, matches=matches, top_k=top_k)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
    if telemetry:
        telemetry.record_rerank(input_tokens=token_count(question), latency_ms=rerank_ms)
    citations = [Citation(**extract_citation(metadata, score)) for metadata, score in reranked]
    chunks = [metadata_chunk_text(metadata) for metadata, _ in reranked]
    final_test_candidates = sum(1 for citation in citations if is_test_file_path(citation.file_path))
    debug = {
        "candidates": debug_candidates,
        "subqueries": subquery_counts,
        "timings_ms": {"embed": round(embed_ms, 2), "rerank": round(rerank_ms, 2)},
        "test_filter": {
            "include_tests": bool(include_tests),
            "excluded_test_candidates": excluded_test_candidates,
            "final_test_candidates": final_test_candidates,
        },
    }
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
    include_tests: bool | None = None,
    test_intent_detected: bool | None = None,
    fallback_with_tests: bool = False,
    telemetry: RagRequestTelemetry | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any]]:
    started = time.perf_counter()
    normalized_mode = normalize_mode(mode)
    effective_top_k = max(top_k, 1)
    if normalized_mode == "search" and is_exhaustive_file_query(question):
        effective_top_k = max(effective_top_k * 4, 40)

    include_repo = scope in {"repo", "both"}
    include_uploads = scope in {"uploads", "both"}
    resolved_test_intent = is_test_intent_query(question) if test_intent_detected is None else bool(test_intent_detected)
    include_tests_effective = bool(include_tests) if include_tests is not None else resolved_test_intent
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
    lexical_files_raw = [path for path in (normalize_file_path(item) for item in lexical_debug.get("candidate_files", []) or []) if path]
    lexical_files, _, lexical_final_tests = filter_paths_by_test_policy(
        lexical_files_raw,
        include_tests=include_tests_effective,
    )
    lexical_scores = lexical_debug.get("file_scores", {}) if isinstance(lexical_debug, dict) else {}
    provided_candidates_raw = [path for path in (normalize_file_path(item) for item in (candidate_files or [])) if path]
    provided_candidates, _, provided_final_tests = filter_paths_by_test_policy(
        provided_candidates_raw,
        include_tests=include_tests_effective,
    )
    primary_candidate_files = dedupe_preserve_order(
        [*provided_candidates, *(lexical_files if provided_candidates else [])]
    )
    excluded_test_candidates = 0

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
                include_tests=include_tests_effective,
                telemetry=telemetry,
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
                    include_tests=include_tests_effective,
                    telemetry=telemetry,
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
                include_tests=include_tests_effective,
                telemetry=telemetry,
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
                include_tests=include_tests_effective,
                telemetry=telemetry,
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
    citations, chunks, excluded_citation_tests, final_test_candidates = filter_citation_pairs_by_test_policy(
        citations,
        chunks,
        include_tests=include_tests_effective,
        limit=effective_top_k,
    )
    excluded_test_candidates += excluded_citation_tests
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
        citations, chunks, post_context_excluded, final_test_candidates = filter_citation_pairs_by_test_policy(
            citations,
            chunks,
            include_tests=include_tests_effective,
            limit=effective_top_k,
        )
        excluded_test_candidates += post_context_excluded
        citations, chunks = dedupe_citation_pairs(citations, chunks, limit=effective_top_k)
    context_ms = (time.perf_counter() - context_started) * 1000.0
    final_test_candidates = sum(1 for citation in citations if is_test_file_path(citation.file_path))
    excluded_test_candidates += max(
        int((index_debug.get("test_filter", {}) or {}).get("excluded_test_candidates", 0)),
        0,
    )
    excluded_test_candidates += max(
        int((lexical_index_debug.get("test_filter", {}) or {}).get("excluded_test_candidates", 0)),
        0,
    )
    excluded_test_candidates += max(
        int((persistent_upload_debug.get("test_filter", {}) or {}).get("excluded_test_candidates", 0)),
        0,
    )
    excluded_test_candidates += max(
        int((temp_upload_debug.get("test_filter", {}) or {}).get("excluded_test_candidates", 0)),
        0,
    )

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
            "candidate_files_raw_count": len(provided_candidates_raw),
            "lexical_candidate_files_count": len(lexical_files),
            "lexical_candidate_files_raw_count": len(lexical_files_raw),
            "repo": repo_name,
        },
        "test_filter": {
            "include_tests": bool(include_tests_effective),
            "test_intent_detected": bool(resolved_test_intent),
            "fallback_with_tests": bool(fallback_with_tests),
            "excluded_test_candidates": max(int(excluded_test_candidates), 0),
            "final_test_candidates": max(int(final_test_candidates), 0),
            "candidate_test_files": {
                "provided": int(provided_final_tests),
                "lexical": int(lexical_final_tests),
            },
        },
    }
    return citations, chunks, debug


def _normalized_candidate_paths(paths: list[str] | None) -> list[str]:
    return dedupe_preserve_order([path for path in (normalize_file_path(item) for item in (paths or [])) if path])


def classify_graph_fallback_reason(
    graph_payload: dict[str, Any],
    signals: dict[str, Any],
    low_conf_reason: str | None,
) -> str:
    errors = [str(item).lower().strip() for item in (graph_payload.get("errors", []) or []) if str(item).strip()]
    if not settings.gitnexus_enabled:
        return "graph_disabled"
    if any("repo_unresolved" in item for item in errors):
        return "graph_repo_unknown"
    if any("repo_not_indexed" in item for item in errors):
        return "graph_not_indexed"
    if any("no indexed repositories" in item for item in errors):
        return "graph_not_indexed"
    if any("gitnexus_unavailable" in item for item in errors) or any("no such file or directory: 'npx'" in item for item in errors):
        return "graph_runtime_unavailable"
    if low_conf_reason == "graph_no_candidates":
        return "graph_no_candidate_files"
    if low_conf_reason:
        return f"retrieval_{low_conf_reason}"
    raw_counts = graph_payload.get("raw_counts", {}) if isinstance(graph_payload, dict) else {}
    score = graph_payload.get("score", {}) if isinstance(graph_payload, dict) else {}
    try:
        best = float((score or {}).get("best") or 0.0)
    except (TypeError, ValueError):
        best = 0.0
    try:
        threshold = float((score or {}).get("threshold") or GRAPH_PROCESS_PRIORITY_THRESHOLD)
    except (TypeError, ValueError):
        threshold = GRAPH_PROCESS_PRIORITY_THRESHOLD
    if best > 0 and best < threshold:
        return "graph_score_below_threshold"
    processes = safe_int(raw_counts.get("processes")) or 0
    files = safe_int(raw_counts.get("files")) or 0
    if bool(signals.get("structure_intent")) and processes == 0 and files == 0:
        return "graph_no_matches"
    return ""


def build_hybrid_debug_payload(
    question: str,
    graph_payload: dict[str, Any],
    fallback_reason: str,
    candidate_files_count: int,
    citations: list[Citation],
    include_tests: bool = False,
    test_intent_detected: bool = False,
    fallback_with_tests: bool = False,
    excluded_test_candidates: int = 0,
    final_test_candidates: int | None = None,
) -> dict[str, Any]:
    graph_index = graph_payload.get("index", {}) if isinstance(graph_payload, dict) else {}
    raw_counts = graph_payload.get("raw_counts", {}) if isinstance(graph_payload, dict) else {}
    graph_score = graph_payload.get("score", {}) if isinstance(graph_payload, dict) else {}

    index_present_value = graph_index.get("index_present")
    if isinstance(index_present_value, bool):
        graph_index_present = index_present_value
    else:
        graph_index_present = bool((safe_int(raw_counts.get("nodes")) or 0) > 0 or (safe_int(raw_counts.get("edges")) or 0) > 0)

    best_score = float(graph_score.get("best") or 0.0)
    threshold = float(graph_score.get("threshold") or GRAPH_PROCESS_PRIORITY_THRESHOLD)
    passed = bool(graph_score.get("passed")) if isinstance(graph_score, dict) else False

    final_test_count = (
        int(final_test_candidates)
        if isinstance(final_test_candidates, int)
        else len({citation.file_path for citation in citations if is_test_file_path(citation.file_path)})
    )

    return {
        "graph_enabled": bool(settings.gitnexus_enabled),
        "graph_index_present": graph_index_present,
        "graph_query": str(graph_payload.get("query") or question),
        "graph_query_type": "natural_language",
        "include_tests": bool(include_tests),
        "test_intent_detected": bool(test_intent_detected),
        "fallback_with_tests": bool(fallback_with_tests),
        "graph_hits": {
            "processes": safe_int(raw_counts.get("processes")) or 0,
            "nodes": safe_int(raw_counts.get("nodes")) or 0,
            "edges": safe_int(raw_counts.get("edges")) or 0,
            "files": safe_int(raw_counts.get("files")) or 0,
        },
        "graph_score": {
            "best": round(best_score, 4),
            "threshold": round(threshold, 4),
            "passed": bool(passed),
        },
        "fallback_reason": fallback_reason,
        "excluded_test_candidates": max(int(excluded_test_candidates), 0),
        "final_test_candidates": max(final_test_count, 0),
        "candidate_files_count": max(int(candidate_files_count), 0),
        "evidence_files_count": len({citation.file_path for citation in citations}),
        "graph_repo": str(graph_payload.get("repo") or ""),
        "graph_metadata": {
            "repo_id": graph_index.get("repo_id"),
            "commit_hash": graph_index.get("commit_hash"),
            "build_timestamp": graph_index.get("build_timestamp"),
            "node_count": graph_index.get("node_count"),
            "edge_count": graph_index.get("edge_count"),
        },
    }


def run_routed_retrieval_plan(
    question: str,
    mode: str,
    top_k: int,
    scope: str,
    project_id: str,
    path_prefix: str | None = None,
    language: str | None = None,
    source_type_filter: str | None = None,
    telemetry: RagRequestTelemetry | None = None,
) -> tuple[list[Citation], list[str], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    normalized_question = " ".join(str(question or "").split())
    base_plan = select_retrieval_plan(question=normalized_question, mode=mode)
    selected_plan = base_plan
    normalized_mode = normalize_mode(mode)
    if normalized_mode == "hybrid":
        if selected_plan == PLAN_VECTOR_ONLY:
            selected_plan = PLAN_GRAPH_PLUS_VECTOR
        elif selected_plan == PLAN_KEYWORD_PLUS_VECTOR:
            selected_plan = PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR
    signals = detect_route_signals(question)
    test_intent_detected = is_test_intent_query(normalized_question)
    signals["test_intent_detected"] = bool(test_intent_detected)
    route_debug = route_debug_template(route=selected_plan, signals=signals)
    route_debug["initial_route"] = base_plan
    if selected_plan != base_plan:
        route_debug["mode_override"] = "hybrid_prefers_graph"
    include_tests = bool(test_intent_detected)
    fallback_with_tests = False
    excluded_test_candidates = 0
    final_test_candidates = 0

    final_top_k = min(max(int(top_k), 1), 5)
    target_repo = default_gitnexus_repo()
    graph_payload: dict[str, Any] = {
        "repo": target_repo,
        "query": normalized_question,
        "processes": [],
        "entrypoints": [],
        "impact": {},
        "candidate_files": [],
        "candidate_file_ranking": [],
        "target_symbol": None,
        "errors": [],
        "raw_counts": {"processes": 0, "nodes": 0, "edges": 0, "files": 0},
        "score": {"best": 0.0, "threshold": GRAPH_PROCESS_PRIORITY_THRESHOLD, "passed": False},
        "index": {
            "repo_id": target_repo,
            "commit_hash": repo_commit_short(),
            "available_repos": [],
            "index_present": None,
            "build_timestamp": None,
            "node_count": None,
            "edge_count": None,
        },
    }
    graph_files_all: list[str] = []
    graph_files: list[str] = []
    keyword_files_all: list[str] = []
    keyword_files: list[str] = []
    retrieval_debug: dict[str, Any] = {"subqueries": [], "candidates": []}
    citations: list[Citation] = []
    chunks: list[str] = []
    selected_candidate_files: list[str] = []
    low_conf_reason: str | None = None

    graph_done = False
    keyword_done = False

    def step_graph() -> None:
        nonlocal graph_payload, graph_files_all, graph_files, graph_done, excluded_test_candidates
        if graph_done:
            return
        started = time.perf_counter()
        graph_payload = run_gitnexus_graph(
            normalized_question,
            repo_name=target_repo,
            include_tests=include_tests,
        )
        graph_files_all = _normalized_candidate_paths(
            graph_payload.get("candidate_files", []) if isinstance(graph_payload, dict) else []
        )
        graph_files, graph_excluded_tests, _ = filter_paths_by_test_policy(graph_files_all, include_tests=include_tests)
        excluded_test_candidates += graph_excluded_tests
        ranking_rows, ranking_excluded_tests, _ = filter_candidate_ranking_rows_by_test_policy(
            graph_payload.get("candidate_file_ranking", []) if isinstance(graph_payload, dict) else [],
            include_tests=include_tests,
        )
        excluded_test_candidates += ranking_excluded_tests
        if isinstance(graph_payload, dict):
            graph_payload["candidate_files"] = list(graph_files)
            graph_payload["candidate_file_ranking"] = ranking_rows
            raw_counts = graph_payload.get("raw_counts")
            if isinstance(raw_counts, dict):
                raw_counts["files"] = len(graph_files)
        processes = graph_payload.get("processes", []) if isinstance(graph_payload, dict) else []
        entrypoints = graph_payload.get("entrypoints", []) if isinstance(graph_payload, dict) else []
        graph_errors = graph_payload.get("errors", []) if isinstance(graph_payload, dict) else []
        lowered_errors = [str(item).lower() for item in graph_errors] if isinstance(graph_errors, list) else []
        symbol_not_found = any("symbol" in item and "not found" in item for item in lowered_errors)
        if symbol_not_found and str(graph_payload.get("target_symbol") or "").strip():
            graph_files_all = []
            graph_files = []
            if isinstance(graph_payload, dict):
                graph_payload["candidate_files"] = []
                graph_payload["candidate_file_ranking"] = []
                raw_counts = graph_payload.get("raw_counts")
                if isinstance(raw_counts, dict):
                    raw_counts["files"] = 0
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
        nonlocal keyword_files_all, keyword_files, keyword_done, excluded_test_candidates
        if keyword_done:
            return
        started = time.perf_counter()
        lexical_payload = lexical_candidate_files(normalized_question)
        keyword_files_all = _normalized_candidate_paths(
            lexical_payload.get("candidate_files", []) if isinstance(lexical_payload, dict) else []
        )
        keyword_files, keyword_excluded_tests, _ = filter_paths_by_test_policy(
            keyword_files_all,
            include_tests=include_tests,
        )
        excluded_test_candidates += keyword_excluded_tests
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

    def step_vector(
        candidate_files: list[str] | None,
        include_tests_for_step: bool,
        fallback_with_tests_flag: bool = False,
    ) -> tuple[list[Citation], list[str], dict[str, Any]]:
        nonlocal excluded_test_candidates
        started = time.perf_counter()
        local_citations, local_chunks, local_debug = retrieve_with_optional_uploads(
            question=normalized_question,
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
            include_tests=include_tests_for_step,
            test_intent_detected=test_intent_detected,
            fallback_with_tests=fallback_with_tests_flag,
            telemetry=telemetry,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        index_debug = (local_debug.get("index", {}) if isinstance(local_debug, dict) else {}) or {}
        index_timing = (index_debug.get("timings_ms", {}) if isinstance(index_debug, dict) else {}) or {}
        pinecone_ms = float(index_timing.get("pinecone_query") or 0.0)
        rerank_ms = float(index_timing.get("rerank") or 0.0)
        local_test_debug = (local_debug.get("test_filter", {}) if isinstance(local_debug, dict) else {}) or {}
        excluded_test_candidates += max(int(local_test_debug.get("excluded_test_candidates", 0) or 0), 0)

        route_debug["steps"].append(
            {
                "name": "pinecone",
                "ms": round(pinecone_ms if pinecone_ms > 0 else elapsed_ms, 2),
                "top_k": ROUTER_PINECONE_TOP_K,
                "filtered": bool(candidate_files),
                "include_tests": bool(include_tests_for_step),
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

    def candidate_files_for_plan(plan: str, include_tests_for_candidates: bool) -> list[str]:
        graph_candidates = graph_files_all if include_tests_for_candidates else graph_files
        keyword_candidates = keyword_files_all if include_tests_for_candidates else keyword_files
        if plan == PLAN_GRAPH_PLUS_VECTOR:
            return list(graph_candidates)
        if plan == PLAN_KEYWORD_PLUS_VECTOR:
            return list(keyword_candidates)
        if plan == PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR:
            return _normalized_candidate_paths([*graph_candidates, *keyword_candidates])
        return []

    if selected_plan == PLAN_GRAPH_ONLY:
        step_graph()
        fallback_reason = classify_graph_fallback_reason(graph_payload, signals, low_conf_reason=None)
        hybrid_debug = build_hybrid_debug_payload(
            question=normalized_question,
            graph_payload=graph_payload,
            fallback_reason=fallback_reason,
            candidate_files_count=len(graph_files),
            citations=[],
            include_tests=include_tests,
            test_intent_detected=test_intent_detected,
            fallback_with_tests=False,
            excluded_test_candidates=excluded_test_candidates,
            final_test_candidates=0,
        )
        route_debug["test_filter"] = {
            "include_tests": bool(include_tests),
            "test_intent_detected": bool(test_intent_detected),
            "fallback_with_tests": False,
            "excluded_test_candidates": max(int(excluded_test_candidates), 0),
            "final_test_candidates": 0,
        }
        route_debug["hybrid_debug"] = hybrid_debug
        graph_payload["hybrid_debug"] = hybrid_debug
        logger.info("hybrid_debug=%s", json.dumps(hybrid_debug, sort_keys=True))
        return citations, chunks, retrieval_debug, graph_payload, route_debug, selected_plan

    if selected_plan in {PLAN_GRAPH_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
        step_graph()
    if selected_plan in {PLAN_KEYWORD_ONLY, PLAN_KEYWORD_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
        step_keyword()

    candidate_files = candidate_files_for_plan(selected_plan, include_tests_for_candidates=include_tests)
    selected_candidate_files = list(candidate_files)
    citations, chunks, retrieval_debug = step_vector(
        candidate_files or None,
        include_tests_for_step=include_tests,
        fallback_with_tests_flag=False,
    )

    low_conf_reason = low_confidence_reason(citations)
    if low_conf_reason and candidate_files:
        unconstrained_citations, unconstrained_chunks, unconstrained_debug = step_vector(
            None,
            include_tests_for_step=include_tests,
            fallback_with_tests_flag=False,
        )
        if unconstrained_citations:
            citations = unconstrained_citations
            chunks = unconstrained_chunks
            retrieval_debug = unconstrained_debug
            selected_candidate_files = []
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

        if selected_plan in {PLAN_KEYWORD_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
            step_keyword()
        if selected_plan in {PLAN_GRAPH_PLUS_VECTOR, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR}:
            step_graph()

        escalation_candidates = candidate_files_for_plan(selected_plan, include_tests_for_candidates=include_tests)
        selected_candidate_files = list(escalation_candidates)
        citations, chunks, retrieval_debug = step_vector(
            escalation_candidates or None,
            include_tests_for_step=include_tests,
            fallback_with_tests_flag=False,
        )
        low_conf_reason = low_confidence_reason(citations)
    else:
        route_debug["escalation"] = {"did_escalate": False, "reason": None}

    if (not include_tests) and low_conf_reason:
        fallback_candidates = candidate_files_for_plan(selected_plan, include_tests_for_candidates=True)
        fallback_citations, fallback_chunks, fallback_debug = step_vector(
            fallback_candidates or None,
            include_tests_for_step=True,
            fallback_with_tests_flag=True,
        )
        fallback_final_test_candidates = sum(
            1 for citation in fallback_citations if is_test_file_path(citation.file_path)
        )
        should_adopt_test_fallback = (
            (not citations and bool(fallback_citations))
            or fallback_final_test_candidates > 0
            or len(fallback_citations) > len(citations)
        )
        if should_adopt_test_fallback:
            fallback_with_tests = True
            include_tests = True
            if isinstance(graph_payload, dict):
                graph_payload["candidate_files"] = list(fallback_candidates)
                raw_counts = graph_payload.get("raw_counts")
                if isinstance(raw_counts, dict):
                    raw_counts["files"] = len(fallback_candidates)
            route_debug["escalation"] = {"did_escalate": True, "reason": f"tests_fallback:{low_conf_reason}"}
            if fallback_citations:
                citations = fallback_citations
                chunks = fallback_chunks
                retrieval_debug = fallback_debug
                selected_candidate_files = list(fallback_candidates)
            low_conf_reason = low_confidence_reason(citations)

    final_test_candidates = sum(1 for citation in citations if is_test_file_path(citation.file_path))

    fallback_reason = classify_graph_fallback_reason(graph_payload, signals, low_conf_reason)
    hybrid_debug = build_hybrid_debug_payload(
        question=normalized_question,
        graph_payload=graph_payload,
        fallback_reason=fallback_reason,
        candidate_files_count=len(selected_candidate_files),
        citations=citations,
        include_tests=include_tests,
        test_intent_detected=test_intent_detected,
        fallback_with_tests=fallback_with_tests,
        excluded_test_candidates=excluded_test_candidates,
        final_test_candidates=final_test_candidates,
    )
    route_debug["test_filter"] = {
        "include_tests": bool(include_tests),
        "test_intent_detected": bool(test_intent_detected),
        "fallback_with_tests": bool(fallback_with_tests),
        "excluded_test_candidates": max(int(excluded_test_candidates), 0),
        "final_test_candidates": max(int(final_test_candidates), 0),
    }
    route_debug["hybrid_debug"] = hybrid_debug
    if isinstance(graph_payload, dict):
        graph_payload["hybrid_debug"] = hybrid_debug
    logger.info("hybrid_debug=%s", json.dumps(hybrid_debug, sort_keys=True))

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
    hybrid_debug = graph.get("hybrid_debug", {}) if isinstance(graph, dict) else {}
    fallback_reason = str(hybrid_debug.get("fallback_reason") or "")
    graph_meta = hybrid_debug.get("graph_metadata", {}) if isinstance(hybrid_debug, dict) else {}
    repo_id = str(graph_meta.get("repo_id") or graph.get("repo") or "unknown-repo")
    commit_hash = str(graph_meta.get("commit_hash") or "unknown")
    processes = graph.get("processes", []) if isinstance(graph, dict) else []
    candidate_files = graph.get("candidate_files", []) if isinstance(graph, dict) else []
    graph_errors = graph.get("errors", []) if isinstance(graph, dict) else []
    process_count = len(processes) if isinstance(processes, list) else 0
    candidate_count = len(candidate_files) if isinstance(candidate_files, list) else 0
    target_symbol = str(graph.get("target_symbol") or "").strip()

    fallback_reason_notes = {
        "graph_not_indexed": (
            f"Graph not indexed for repo {repo_id} at commit {commit_hash}. "
            "Run `npx -y gitnexus@latest analyze` in the repo and restart the service."
        ),
        "graph_runtime_unavailable": "Graph runtime unavailable in this deployment (missing `npx` and/or GitNexus MCP runtime).",
        "graph_repo_unknown": "Graph repo id is unresolved. Set `GITNEXUS_DEFAULT_REPO` to a valid indexed repo.",
        "graph_no_candidate_files": "Graph query returned no candidate files, so constrained retrieval could not run.",
        "retrieval_no_matches": "Constrained Pinecone retrieval returned no chunks for this query.",
        "retrieval_graph_unavailable": "Graph service reported errors, so retrieval relied on fallback behavior.",
        "retrieval_low_confidence": "Retrieved chunks were low-confidence after reranking.",
    }
    lines.append("Summary")
    if evidence_rows:
        lines.append(
            f"- Evidence-backed result: {len(evidence_rows)} line-level citation(s) returned for this question."
        )
    else:
        lines.append("- Architecture-only result: no line-level Pinecone citations were returned.")
    lines.append(f"- Graph coverage: {process_count} process flow(s) in repo `{repo_id}`.")
    if fallback_reason:
        lines.append(f"- Retrieval diagnostic: {fallback_reason_notes.get(fallback_reason, fallback_reason)}")
    if used_fallback:
        lines.append("- Retrieval fallback path was used.")

    if isinstance(processes, list) and processes:
        top_processes: list[str] = []
        for process in processes[:3]:
            if not isinstance(process, dict):
                continue
            label = str(process.get("summary") or process.get("id") or "Unnamed process").strip()
            if label:
                top_processes.append(label)
        if top_processes:
            lines.append(f"- Top graph signals: {', '.join(top_processes)}.")

    lines.append("")
    if evidence_rows:
        lines.append("Evidence Highlights")
        for row in evidence_rows[:3]:
            lines.append(
                f"- [{row.get('citation_index')}] {row.get('file_path')}:{row.get('line_start')}-{row.get('line_end')}"
            )
    else:
        lines.append("Why Evidence Is Missing")
        lines.append(f"- Candidate file constraints: {candidate_count} file(s).")
        if target_symbol:
            lines.append(f"- Graph target symbol: `{target_symbol}`.")
        if isinstance(graph_errors, list):
            condensed_errors = [str(item).strip() for item in graph_errors if str(item).strip()]
            if condensed_errors:
                lines.append(f"- Graph errors: {'; '.join(condensed_errors[:2])}")

    lines.append("")
    lines.append("Next Best Query")
    if target_symbol:
        lines.append(f"- Show callers of {target_symbol} with file:line citations.")
    elif evidence_rows:
        lines.append("- Explain the control/data flow using only citations [1]-[3].")
    else:
        lines.append("- Use Search mode on one candidate file to verify Pinecone coverage, then rerun Hybrid.")
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


def telemetry_mode_value(payload: QueryRequest) -> str:
    raw = str(payload.ui_mode or payload.mode or "chat").strip().lower()
    return raw or "chat"


def extract_retrieval_counts(retrieval_debug: dict[str, Any], citations: list[Citation]) -> tuple[int, int, int]:
    candidate_rows: list[dict[str, Any]] = []
    for key in ("index", "index_lexical", "uploads"):
        section = retrieval_debug.get(key, {}) if isinstance(retrieval_debug, dict) else {}
        if not isinstance(section, dict):
            continue
        rows = section.get("candidates", [])
        if isinstance(rows, list):
            candidate_rows.extend(item for item in rows if isinstance(item, dict))
    retrieved_file_count = len(
        {
            str(item.get("file_path", "")).strip()
            for item in candidate_rows
            if str(item.get("file_path", "")).strip()
        }
    )
    retrieved_chunk_count = len(candidate_rows)
    selected_chunk_count = len(citations)
    if retrieved_file_count <= 0 and citations:
        retrieved_file_count = len({citation.file_path for citation in citations if citation.file_path})
    if retrieved_chunk_count <= 0 and citations:
        retrieved_chunk_count = len(citations)
    return retrieved_file_count, retrieved_chunk_count, selected_chunk_count


def attach_retrieval_telemetry(
    telemetry: RagRequestTelemetry,
    retrieval_debug: dict[str, Any],
    citations: list[Citation],
) -> None:
    if not telemetry:
        return
    retrieved_file_count, retrieved_chunk_count, selected_chunk_count = extract_retrieval_counts(
        retrieval_debug,
        citations,
    )
    telemetry.record_counts(
        retrieved_file_count=retrieved_file_count,
        retrieved_chunk_count=retrieved_chunk_count,
        selected_chunk_count=selected_chunk_count,
    )
    telemetry.mark_retrieval_complete()


def finalize_and_persist_telemetry(telemetry: RagRequestTelemetry) -> dict[str, Any]:
    telemetry.finalize()
    if settings.telemetry_enabled:
        store = get_telemetry_store()
        store.persist(telemetry)
    logger.info("%s", emit_telemetry_log(telemetry))
    return telemetry.to_response_dict()


def is_weak_evidence(evidence: dict[str, Any]) -> bool:
    label = str(evidence.get("label", "")).lower()
    score = float(evidence.get("score", 0.0) or 0.0)
    return label == "low" or score < 0.42


def generate_answer(
    question: str,
    context: str,
    telemetry: RagRequestTelemetry | None = None,
) -> str:
    client = get_openai_client()
    started = time.perf_counter()
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
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if telemetry:
        usage = parse_openai_usage(getattr(completion, "usage", None))
        telemetry.record_llm(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_input_tokens=int(usage.get("cached_tokens") or 0),
            latency_ms=elapsed_ms,
            model_name=str(getattr(completion, "model", settings.openai_chat_model)),
        )
    content = completion.choices[0].message.content
    return content.strip() if content else "No answer generated."


def build_request_telemetry(payload: QueryRequest) -> RagRequestTelemetry:
    return create_request_telemetry(
        user_query=payload.question,
        repo_name=normalize_project_id(payload.project_id),
        mode=telemetry_mode_value(payload),
        top_k=payload.top_k,
        model_name=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        rerank_enabled=bool(settings.retrieval_deterministic_rerank_enabled),
    )


def execute_search_request(
    payload: QueryRequest,
    uploaded_files: list[dict[str, Any]] | None = None,
) -> SearchResponse:
    telemetry = build_request_telemetry(payload)
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
            uploaded_files=uploaded_files or [],
            mode=normalize_mode(payload.mode),
            scope=normalize_scope(payload.scope),
            project_id=normalize_project_id(payload.project_id),
            path_prefix=path_prefix,
            language=language,
            source_type_filter=source_type_filter,
            telemetry=telemetry,
        )
    except HTTPException as exc:
        telemetry.mark_failure("retrieval", exc.detail)
        finalize_and_persist_telemetry(telemetry)
        raise
    except Exception as exc:
        telemetry.mark_failure("retrieval", exc)
        finalize_and_persist_telemetry(telemetry)
        raise HTTPException(status_code=502, detail=f"Vector search failed: {exc}") from exc

    try:
        postprocess_started = time.perf_counter()
        analysis = apply_mode_analysis(
            mode=payload.mode,
            question=payload.question,
            citations=citations,
            chunks=chunks,
        )
        citations = analysis["citations"]
        chunks = analysis["chunks"]
        attach_retrieval_telemetry(telemetry, retrieval_debug, citations)
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
        telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
        telemetry.mark_success(str(analysis.get("summary") or ""))
        response_telemetry = finalize_and_persist_telemetry(telemetry)
        if payload.debug:
            debug_payload["telemetry"] = response_telemetry
        return SearchResponse(
            matches=citations,
            evidence_strength=evidence,
            debug=debug_payload,
            telemetry=response_telemetry,
            result_type=str(analysis.get("result_type", "Ranked Chunks")),
            summary=analysis.get("summary"),
            follow_ups=list(analysis.get("follow_ups", [])),
            file_results=list(analysis.get("file_results", [])),
            graph_edges=list(analysis.get("graph_edges", [])),
            pattern_examples=list(analysis.get("pattern_examples", [])),
        )
    except HTTPException as exc:
        telemetry.mark_failure("postprocess", exc.detail)
        finalize_and_persist_telemetry(telemetry)
        raise
    except Exception as exc:
        telemetry.mark_failure("postprocess", exc)
        finalize_and_persist_telemetry(telemetry)
        raise HTTPException(status_code=500, detail=f"Search response postprocess failed: {exc}") from exc


def execute_query_request(
    payload: QueryRequest,
    uploaded_files: list[dict[str, Any]] | None = None,
) -> QueryResponse:
    telemetry = build_request_telemetry(payload)
    started = time.perf_counter()
    normalized_mode = normalize_mode(payload.mode)
    normalized_ui = normalized_ui_mode(payload)
    normalized_scope = normalize_scope(payload.scope)
    normalized_project_id = normalize_project_id(payload.project_id)
    path_prefix, language, source_type_filter = normalized_request_filters(
        payload.path_prefix,
        payload.language,
        payload.source_type,
    )
    uploaded_files = uploaded_files or []

    if normalized_ui in DIRECT_UI_MODES:
        try:
            return execute_direct_ui_mode_request(payload, telemetry=telemetry, uploaded_files=uploaded_files)
        except HTTPException as exc:
            telemetry.mark_failure("postprocess", exc.detail)
            finalize_and_persist_telemetry(telemetry)
            raise
        except Exception as exc:
            stage = "llm" if telemetry.llm_latency_ms <= 0 else "postprocess"
            telemetry.mark_failure(stage, exc)
            finalize_and_persist_telemetry(telemetry)
            if stage == "llm":
                raise HTTPException(status_code=502, detail=f"Direct {normalized_ui} generation failed: {exc}") from exc
            raise HTTPException(status_code=500, detail=f"Direct {normalized_ui} response failed: {exc}") from exc

    try:
        if uploaded_files and normalized_mode not in {"hybrid", "graph"}:
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
                telemetry=telemetry,
            )
            graph_payload: dict[str, Any] = {}
            route_debug: dict[str, Any] = {"route": "direct", "steps": [], "hybrid_debug": {}}
            routed_plan = PLAN_VECTOR_ONLY
        else:
            citations, chunks, retrieval_debug, graph_payload, route_debug, routed_plan = run_routed_retrieval_plan(
                question=payload.question,
                mode=normalized_mode,
                top_k=payload.top_k,
                scope=normalized_scope,
                project_id=normalized_project_id,
                path_prefix=path_prefix,
                language=language,
                source_type_filter=source_type_filter,
                telemetry=telemetry,
            )
    except HTTPException as exc:
        telemetry.mark_failure("retrieval", exc.detail)
        finalize_and_persist_telemetry(telemetry)
        raise
    except Exception as exc:
        telemetry.mark_failure("retrieval", exc)
        finalize_and_persist_telemetry(telemetry)
        raise HTTPException(status_code=502, detail=f"Routed retrieval failed: {exc}") from exc

    if graph_payload.get("repo"):
        telemetry.repo_name = str(graph_payload.get("repo"))

    debug_payload: dict[str, Any] = {
        "route_debug": route_debug,
        "hybrid_debug": route_debug.get("hybrid_debug", {}),
    }

    try:
        postprocess_started = time.perf_counter()
        evidence = compute_evidence_strength(
            payload.question,
            citations,
            retrieval_debug,
            mode=payload.mode,
            mode_metrics={},
        )
        evidence_rows = build_hybrid_evidence_rows(citations, chunks, limit=8)
        attach_retrieval_telemetry(telemetry, retrieval_debug, citations)

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
            telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
            telemetry.mark_success(answer)
            response_telemetry = finalize_and_persist_telemetry(telemetry)
            if payload.debug:
                debug_payload["telemetry"] = response_telemetry
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
                telemetry=response_telemetry,
            )

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
                verbose_debug["hybrid_debug"] = route_debug.get("hybrid_debug", {})
                debug_payload = verbose_debug
            telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
            telemetry.mark_success(answer)
            response_telemetry = finalize_and_persist_telemetry(telemetry)
            if payload.debug:
                debug_payload["telemetry"] = response_telemetry
            return QueryResponse(
                answer=answer,
                citations=citations,
                graph=graph_payload,
                evidence=evidence_rows,
                evidence_strength=evidence,
                debug=debug_payload,
                telemetry=response_telemetry,
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
                verbose_debug["hybrid_debug"] = route_debug.get("hybrid_debug", {})
                debug_payload = verbose_debug
            answer = insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms)
            telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
            telemetry.mark_success(answer)
            response_telemetry = finalize_and_persist_telemetry(telemetry)
            if payload.debug:
                debug_payload["telemetry"] = response_telemetry
            return QueryResponse(
                answer=answer,
                citations=[],
                graph=graph_payload,
                evidence=[],
                evidence_strength=evidence,
                debug=debug_payload,
                telemetry=response_telemetry,
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
                verbose_debug["hybrid_debug"] = route_debug.get("hybrid_debug", {})
                debug_payload = verbose_debug
            answer = insufficient_evidence_answer(payload.question, suggestions, missing_terms=missing_terms)
            telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
            telemetry.mark_success(answer)
            response_telemetry = finalize_and_persist_telemetry(telemetry)
            if payload.debug:
                debug_payload["telemetry"] = response_telemetry
            return QueryResponse(
                answer=answer,
                citations=response_citations,
                graph=graph_payload,
                evidence=build_hybrid_evidence_rows(response_citations, chunks, limit=8),
                evidence_strength=evidence,
                debug=debug_payload,
                telemetry=response_telemetry,
            )

        answer = generate_answer(payload.question, context, telemetry=telemetry)
        if payload.debug:
            verbose_debug = build_debug_payload(
                retrieval_debug=retrieval_debug,
                context=context,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            verbose_debug["route_debug"] = route_debug
            verbose_debug["hybrid_debug"] = route_debug.get("hybrid_debug", {})
            debug_payload = verbose_debug
        telemetry.record_postprocess((time.perf_counter() - postprocess_started) * 1000.0)
        telemetry.mark_success(answer)
        response_telemetry = finalize_and_persist_telemetry(telemetry)
        if payload.debug:
            debug_payload["telemetry"] = response_telemetry
        return QueryResponse(
            answer=answer,
            citations=citations,
            graph=graph_payload,
            evidence=evidence_rows,
            evidence_strength=evidence,
            debug=debug_payload,
            telemetry=response_telemetry,
        )
    except HTTPException as exc:
        telemetry.mark_failure("postprocess", exc.detail)
        finalize_and_persist_telemetry(telemetry)
        raise
    except ValidationError as exc:
        telemetry.mark_failure("postprocess", exc)
        finalize_and_persist_telemetry(telemetry)
        raise HTTPException(status_code=500, detail=f"Response validation error: {exc}") from exc
    except Exception as exc:
        stage = "llm" if telemetry.llm_latency_ms <= 0 and citations else "postprocess"
        telemetry.mark_failure(stage, exc)
        finalize_and_persist_telemetry(telemetry)
        if stage == "llm":
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"Query response postprocess failed: {exc}") from exc


@app.post("/api/search", response_model=SearchResponse)
def search(payload: QueryRequest) -> SearchResponse:
    return execute_search_request(payload)


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return execute_query_request(payload)


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
    ui_mode: str | None = Form(default=None),
    diagram_type: str | None = Form(default=None),
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
        ui_mode=ui_mode or mode,
        diagram_type=diagram_type,
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
    return execute_search_request(payload, uploaded_files=uploaded_files)


@app.post("/api/query/upload", response_model=QueryResponse)
async def query_with_uploads(
    question: str = Form(...),
    top_k: int = Form(5),
    files: list[UploadFile] | None = File(default=None),
    debug: str | None = Form(default=None),
    mode: str = Form("chat"),
    ui_mode: str | None = Form(default=None),
    diagram_type: str | None = Form(default=None),
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
        ui_mode=ui_mode or mode,
        diagram_type=diagram_type,
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
    return execute_query_request(payload, uploaded_files=uploaded_files)
