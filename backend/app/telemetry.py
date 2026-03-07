from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import settings
from app.pricing import RAG_PRICING

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised in deployment with psycopg installed
    psycopg = None
    dict_row = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(item) for item in values)
    rank = max(0.0, min(1.0, pct)) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return (ordered[low] * (1.0 - weight)) + (ordered[high] * weight)


def _extract_attr(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def parse_openai_usage(usage: Any) -> dict[str, int]:
    prompt_tokens = _safe_int(_extract_attr(usage, "prompt_tokens"))
    completion_tokens = _safe_int(_extract_attr(usage, "completion_tokens"))
    cached_tokens = _safe_int(_extract_attr(usage, "prompt_tokens_details", "cached_tokens"))
    if cached_tokens <= 0:
        cached_tokens = _safe_int(_extract_attr(usage, "input_tokens_details", "cached_tokens"))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def parse_embedding_usage(response: Any) -> dict[str, int]:
    usage = _extract_attr(response, "usage")
    return {
        "input_tokens": _safe_int(_extract_attr(usage, "prompt_tokens") or _extract_attr(usage, "total_tokens")),
    }


def parse_pinecone_usage(response: Any) -> dict[str, float]:
    usage = _extract_attr(response, "usage")
    return {
        "read_units": _safe_float(_extract_attr(usage, "read_units")),
        "write_units": _safe_float(_extract_attr(usage, "write_units")),
    }


@dataclass
class RagRequestTelemetry:
    request_id: str
    created_at: str
    user_query: str
    repo_name: str
    mode: str
    model_name: str
    embedding_model: str
    top_k: int
    rerank_enabled: bool
    timestamp: str = field(default_factory=_utc_now_iso)
    final_answer_length: int = 0
    status: str = "success"
    error_message: str = ""
    stage_failed: str = ""
    total_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    pinecone_query_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    postprocess_latency_ms: float = 0.0
    time_to_first_retrieval_ms: float = 0.0
    time_to_final_answer_ms: float = 0.0
    pinecone_read_units: float = 0.0
    pinecone_write_units: float = 0.0
    pinecone_query_count: int = 0
    pinecone_cost_usd_est: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cached_input_tokens: int = 0
    llm_cost_usd_est: float = 0.0
    embedding_input_tokens: int = 0
    embedding_cost_usd_est: float = 0.0
    rerank_input_tokens: int = 0
    rerank_cost_usd_est: float = 0.0
    total_cost_usd_est: float = 0.0
    retrieved_file_count: int = 0
    retrieved_chunk_count: int = 0
    selected_chunk_count: int = 0
    _started_perf: float = field(default_factory=time.perf_counter, repr=False)

    def add_latency(self, stage: str, elapsed_ms: float) -> None:
        value = max(float(elapsed_ms or 0.0), 0.0)
        if stage == "embedding":
            self.embedding_latency_ms += value
        elif stage == "pinecone_query":
            self.pinecone_query_latency_ms += value
        elif stage == "rerank":
            self.rerank_latency_ms += value
        elif stage == "llm":
            self.llm_latency_ms += value
        elif stage == "postprocess":
            self.postprocess_latency_ms += value

    def mark_retrieval_complete(self) -> None:
        if self.time_to_first_retrieval_ms <= 0:
            self.time_to_first_retrieval_ms = max((time.perf_counter() - self._started_perf) * 1000.0, 0.0)

    def record_embedding(self, *, input_tokens: int = 0, latency_ms: float = 0.0) -> None:
        self.embedding_input_tokens += max(int(input_tokens or 0), 0)
        self.add_latency("embedding", latency_ms)

    def record_pinecone(self, *, read_units: float = 0.0, write_units: float = 0.0, query_count: int = 0, latency_ms: float = 0.0) -> None:
        self.pinecone_read_units += max(float(read_units or 0.0), 0.0)
        self.pinecone_write_units += max(float(write_units or 0.0), 0.0)
        self.pinecone_query_count += max(int(query_count or 0), 0)
        self.add_latency("pinecone_query", latency_ms)

    def record_rerank(self, *, input_tokens: int = 0, latency_ms: float = 0.0) -> None:
        self.rerank_input_tokens += max(int(input_tokens or 0), 0)
        self.add_latency("rerank", latency_ms)

    def record_llm(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        latency_ms: float = 0.0,
        model_name: str | None = None,
    ) -> None:
        self.llm_input_tokens += max(int(input_tokens or 0), 0)
        self.llm_output_tokens += max(int(output_tokens or 0), 0)
        self.llm_cached_input_tokens += max(int(cached_input_tokens or 0), 0)
        if model_name:
            self.model_name = str(model_name)
        self.add_latency("llm", latency_ms)

    def record_postprocess(self, latency_ms: float) -> None:
        self.add_latency("postprocess", latency_ms)

    def record_counts(
        self,
        *,
        retrieved_file_count: int | None = None,
        retrieved_chunk_count: int | None = None,
        selected_chunk_count: int | None = None,
    ) -> None:
        if retrieved_file_count is not None:
            self.retrieved_file_count = max(int(retrieved_file_count), 0)
        if retrieved_chunk_count is not None:
            self.retrieved_chunk_count = max(int(retrieved_chunk_count), 0)
        if selected_chunk_count is not None:
            self.selected_chunk_count = max(int(selected_chunk_count), 0)

    def mark_success(self, final_answer: str) -> None:
        self.status = "success"
        self.error_message = ""
        self.stage_failed = ""
        self.final_answer_length = len(str(final_answer or ""))

    def mark_failure(self, stage: str, error: Exception | str) -> None:
        self.status = "failure"
        self.stage_failed = str(stage or "").strip()
        self.error_message = str(error or "").strip()

    def finalize(self) -> None:
        self.total_latency_ms = max((time.perf_counter() - self._started_perf) * 1000.0, 0.0)
        self.time_to_final_answer_ms = self.total_latency_ms
        self.pinecone_cost_usd_est = (
            (self.pinecone_read_units / 1_000_000.0) * RAG_PRICING.pinecone.read_unit_per_million_usd
            + (self.pinecone_write_units / 1_000_000.0) * RAG_PRICING.pinecone.write_unit_per_million_usd
        )
        self.llm_cost_usd_est = (
            (self.llm_input_tokens / 1_000_000.0) * RAG_PRICING.llm.input_per_million_usd
            + (self.llm_cached_input_tokens / 1_000_000.0) * RAG_PRICING.llm.cached_input_per_million_usd
            + (self.llm_output_tokens / 1_000_000.0) * RAG_PRICING.llm.output_per_million_usd
        )
        self.embedding_cost_usd_est = (
            (self.embedding_input_tokens / 1_000_000.0) * RAG_PRICING.embedding.input_per_million_usd
        )
        self.rerank_cost_usd_est = (
            (self.rerank_input_tokens / 1_000_000.0) * RAG_PRICING.rerank.input_per_million_usd
        )
        self.total_cost_usd_est = (
            self.pinecone_cost_usd_est
            + self.llm_cost_usd_est
            + self.embedding_cost_usd_est
            + self.rerank_cost_usd_est
        )

    def to_row(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_started_perf", None)
        data["created_at"] = self.created_at or self.timestamp
        return data

    def to_response_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "repo_name": self.repo_name,
            "mode": self.mode,
            "status": self.status,
            "stage_failed": self.stage_failed,
            "final_answer_length": self.final_answer_length,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "retrieval_latency_ms": round(
                self.embedding_latency_ms + self.pinecone_query_latency_ms + self.rerank_latency_ms,
                2,
            ),
            "generation_latency_ms": round(self.llm_latency_ms, 2),
            "postprocess_latency_ms": round(self.postprocess_latency_ms, 2),
            "time_to_first_retrieval_ms": round(self.time_to_first_retrieval_ms, 2),
            "time_to_final_answer_ms": round(self.time_to_final_answer_ms, 2),
            "pinecone_read_units": round(self.pinecone_read_units, 4),
            "pinecone_write_units": round(self.pinecone_write_units, 4),
            "pinecone_query_count": self.pinecone_query_count,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cached_input_tokens": self.llm_cached_input_tokens,
            "embedding_input_tokens": self.embedding_input_tokens,
            "rerank_input_tokens": self.rerank_input_tokens,
            "retrieved_file_count": self.retrieved_file_count,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "selected_chunk_count": self.selected_chunk_count,
            "estimated_cost_usd": round(self.total_cost_usd_est, 8),
            "cost_breakdown_usd": {
                "pinecone": round(self.pinecone_cost_usd_est, 8),
                "llm": round(self.llm_cost_usd_est, 8),
                "embedding": round(self.embedding_cost_usd_est, 8),
                "rerank": round(self.rerank_cost_usd_est, 8),
            },
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "rerank_enabled": self.rerank_enabled,
            "error_message": self.error_message,
        }

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "event": "rag_request_complete",
            "request_id": self.request_id,
            "repo_name": self.repo_name,
            "mode": self.mode,
            "status": self.status,
            "stage_failed": self.stage_failed or None,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "time_to_first_retrieval_ms": round(self.time_to_first_retrieval_ms, 2),
            "pinecone_read_units": round(self.pinecone_read_units, 4),
            "pinecone_write_units": round(self.pinecone_write_units, 4),
            "pinecone_query_count": self.pinecone_query_count,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embedding_input_tokens": self.embedding_input_tokens,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "selected_chunk_count": self.selected_chunk_count,
            "total_cost_usd_est": round(self.total_cost_usd_est, 8),
            "error_message": self.error_message or None,
        }


def create_request_telemetry(
    *,
    user_query: str,
    repo_name: str,
    mode: str,
    top_k: int,
    model_name: str,
    embedding_model: str,
    rerank_enabled: bool,
) -> RagRequestTelemetry:
    return RagRequestTelemetry(
        request_id=f"rag_{uuid.uuid4().hex[:12]}",
        created_at=_utc_now_iso(),
        user_query=str(user_query),
        repo_name=str(repo_name),
        mode=str(mode),
        model_name=str(model_name),
        embedding_model=str(embedding_model),
        top_k=max(int(top_k or 0), 0),
        rerank_enabled=bool(rerank_enabled),
    )


class TelemetryStore:
    TABLE_NAME = "rag_request_telemetry"
    UPSERT_COLUMNS = [
        "request_id",
        "created_at",
        "timestamp",
        "repo_name",
        "mode",
        "user_query",
        "model_name",
        "embedding_model",
        "top_k",
        "rerank_enabled",
        "final_answer_length",
        "status",
        "stage_failed",
        "error_message",
        "total_latency_ms",
        "embedding_latency_ms",
        "pinecone_query_latency_ms",
        "rerank_latency_ms",
        "llm_latency_ms",
        "postprocess_latency_ms",
        "time_to_first_retrieval_ms",
        "time_to_final_answer_ms",
        "pinecone_read_units",
        "pinecone_write_units",
        "pinecone_query_count",
        "pinecone_cost_usd_est",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_cached_input_tokens",
        "llm_cost_usd_est",
        "embedding_input_tokens",
        "embedding_cost_usd_est",
        "rerank_input_tokens",
        "rerank_cost_usd_est",
        "total_cost_usd_est",
        "retrieved_file_count",
        "retrieved_chunk_count",
        "selected_chunk_count",
    ]

    def __init__(self, db_path: str | None = None, database_url: str | None = None) -> None:
        configured_url = str(
            database_url or settings.telemetry_database_url or os.getenv("DATABASE_URL") or ""
        ).strip()
        self.database_url = ""
        self.db_path: Path | None = None
        self._lock = threading.Lock()
        if configured_url:
            if configured_url.startswith("postgres://"):
                configured_url = configured_url.replace("postgres://", "postgresql://", 1)
            if psycopg is None:
                raise RuntimeError(
                    "Telemetry database URL is configured, but psycopg is not installed. "
                    "Add psycopg[binary] to backend dependencies."
                )
            self.backend = "postgres"
            self.database_url = configured_url
        else:
            self.backend = "sqlite"
            configured_path = str(db_path or settings.telemetry_db_path or "").strip()
            if configured_path:
                self.db_path = Path(configured_path).expanduser().resolve()
            else:
                self.db_path = (Path(__file__).resolve().parent / "data" / "rag_telemetry.sqlite3").resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> Any:
        if self.backend == "postgres":
            return psycopg.connect(self.database_url, row_factory=dict_row)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _placeholder(self) -> str:
        return "%s" if self.backend == "postgres" else "?"

    def _normalize_rows(self, rows: list[Any]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as conn:
            if self.backend == "postgres":
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                      id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                      request_id TEXT UNIQUE,
                      created_at TIMESTAMPTZ,
                      timestamp TIMESTAMPTZ,
                      repo_name TEXT,
                      mode TEXT,
                      user_query TEXT,
                      model_name TEXT,
                      embedding_model TEXT,
                      top_k INTEGER,
                      rerank_enabled BOOLEAN,
                      final_answer_length INTEGER,
                      status TEXT,
                      stage_failed TEXT,
                      error_message TEXT,
                      total_latency_ms DOUBLE PRECISION,
                      embedding_latency_ms DOUBLE PRECISION,
                      pinecone_query_latency_ms DOUBLE PRECISION,
                      rerank_latency_ms DOUBLE PRECISION,
                      llm_latency_ms DOUBLE PRECISION,
                      postprocess_latency_ms DOUBLE PRECISION,
                      time_to_first_retrieval_ms DOUBLE PRECISION,
                      time_to_final_answer_ms DOUBLE PRECISION,
                      pinecone_read_units DOUBLE PRECISION,
                      pinecone_write_units DOUBLE PRECISION,
                      pinecone_query_count INTEGER,
                      pinecone_cost_usd_est DOUBLE PRECISION,
                      llm_input_tokens INTEGER,
                      llm_output_tokens INTEGER,
                      llm_cached_input_tokens INTEGER,
                      llm_cost_usd_est DOUBLE PRECISION,
                      embedding_input_tokens INTEGER,
                      embedding_cost_usd_est DOUBLE PRECISION,
                      rerank_input_tokens INTEGER,
                      rerank_cost_usd_est DOUBLE PRECISION,
                      total_cost_usd_est DOUBLE PRECISION,
                      retrieved_file_count INTEGER,
                      retrieved_chunk_count INTEGER,
                      selected_chunk_count INTEGER
                    )
                    """
                )
            else:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      request_id TEXT UNIQUE,
                      created_at TEXT,
                      timestamp TEXT,
                      repo_name TEXT,
                      mode TEXT,
                      user_query TEXT,
                      model_name TEXT,
                      embedding_model TEXT,
                      top_k INTEGER,
                      rerank_enabled INTEGER,
                      final_answer_length INTEGER,
                      status TEXT,
                      stage_failed TEXT,
                      error_message TEXT,
                      total_latency_ms REAL,
                      embedding_latency_ms REAL,
                      pinecone_query_latency_ms REAL,
                      rerank_latency_ms REAL,
                      llm_latency_ms REAL,
                      postprocess_latency_ms REAL,
                      time_to_first_retrieval_ms REAL,
                      time_to_final_answer_ms REAL,
                      pinecone_read_units REAL,
                      pinecone_write_units REAL,
                      pinecone_query_count INTEGER,
                      pinecone_cost_usd_est REAL,
                      llm_input_tokens INTEGER,
                      llm_output_tokens INTEGER,
                      llm_cached_input_tokens INTEGER,
                      llm_cost_usd_est REAL,
                      embedding_input_tokens INTEGER,
                      embedding_cost_usd_est REAL,
                      rerank_input_tokens INTEGER,
                      rerank_cost_usd_est REAL,
                      total_cost_usd_est REAL,
                      retrieved_file_count INTEGER,
                      retrieved_chunk_count INTEGER,
                      selected_chunk_count INTEGER
                    )
                    """
                )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_rag_request_telemetry_created_at ON {self.TABLE_NAME}(created_at DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_rag_request_telemetry_mode ON {self.TABLE_NAME}(mode)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_rag_request_telemetry_repo ON {self.TABLE_NAME}(repo_name)"
            )
            conn.commit()

    def persist(self, telemetry: RagRequestTelemetry) -> None:
        row = telemetry.to_row()
        placeholders = ", ".join([self._placeholder()] * len(self.UPSERT_COLUMNS))
        values = [row.get(column) for column in self.UPSERT_COLUMNS]
        updates = ", ".join(
            f"{column} = EXCLUDED.{column}" for column in self.UPSERT_COLUMNS if column != "request_id"
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                  {", ".join(self.UPSERT_COLUMNS)}
                ) VALUES ({placeholders})
                ON CONFLICT (request_id) DO UPDATE SET
                  {updates}
                """,
                values,
            )
            conn.commit()

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        safe_limit = max(int(limit or settings.telemetry_recent_limit), 1)
        placeholder = self._placeholder()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  request_id,
                  created_at,
                  repo_name,
                  mode,
                  status,
                  stage_failed,
                  total_latency_ms,
                  pinecone_read_units,
                  pinecone_write_units,
                  pinecone_query_count,
                  llm_input_tokens,
                  llm_output_tokens,
                  embedding_input_tokens,
                  total_cost_usd_est,
                  retrieved_chunk_count,
                  selected_chunk_count,
                  error_message
                FROM {self.TABLE_NAME}
                ORDER BY created_at DESC
                LIMIT {placeholder}
                """,
                (safe_limit,),
            ).fetchall()
        return self._normalize_rows(rows)

    def summary(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  request_id,
                  created_at,
                  repo_name,
                  mode,
                  status,
                  model_name,
                  total_latency_ms,
                  total_cost_usd_est,
                  retrieved_chunk_count,
                  selected_chunk_count
                FROM {self.TABLE_NAME}
                ORDER BY created_at DESC
                """
            ).fetchall()
        records = self._normalize_rows(rows)
        latencies = [float(row.get("total_latency_ms") or 0.0) for row in records if row.get("total_latency_ms") is not None]
        costs = [float(row.get("total_cost_usd_est") or 0.0) for row in records]
        failures = [row for row in records if str(row.get("status")) == "failure"]
        retrieved_chunks = [int(row.get("retrieved_chunk_count") or 0) for row in records]
        selected_chunks = [int(row.get("selected_chunk_count") or 0) for row in records]

        def aggregate_by(key: str) -> list[dict[str, Any]]:
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in records:
                group_key = str(row.get(key) or "unknown")
                groups.setdefault(group_key, []).append(row)
            output: list[dict[str, Any]] = []
            for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
                output.append(
                    {
                        key: group_key,
                        "request_count": len(group_rows),
                        "total_cost_usd_est": round(sum(float(item.get("total_cost_usd_est") or 0.0) for item in group_rows), 8),
                        "avg_cost_usd_est": round(mean(float(item.get("total_cost_usd_est") or 0.0) for item in group_rows), 8),
                        "avg_latency_ms": round(mean(float(item.get("total_latency_ms") or 0.0) for item in group_rows), 2),
                    }
                )
            return output

        total_requests = len(records)
        return {
            "request_count": total_requests,
            "avg_cost_usd_est": round(mean(costs), 8) if costs else 0.0,
            "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
            "p50_latency_ms": round(_percentile(latencies, 0.50), 2) if latencies else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
            "p99_latency_ms": round(_percentile(latencies, 0.99), 2) if latencies else 0.0,
            "failure_rate": round((len(failures) / total_requests), 4) if total_requests else 0.0,
            "average_retrieved_chunks": round(mean(retrieved_chunks), 2) if retrieved_chunks else 0.0,
            "average_selected_chunks": round(mean(selected_chunks), 2) if selected_chunks else 0.0,
            "cost_by_mode": aggregate_by("mode"),
            "cost_by_repo": aggregate_by("repo_name"),
            "cost_by_model": aggregate_by("model_name"),
        }


_store: TelemetryStore | None = None


def get_telemetry_store() -> TelemetryStore:
    global _store
    if _store is None:
        _store = TelemetryStore()
    return _store


def emit_telemetry_log(telemetry: RagRequestTelemetry) -> str:
    return json.dumps(telemetry.to_log_payload(), sort_keys=True)
