import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main
from app.telemetry import TelemetryStore, create_request_telemetry


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: object = None):
        self.executed.append((" ".join(query.split()), params))
        return self

    def fetchall(self):
        return []

    def commit(self) -> None:
        return None


class _FakePsycopgModule:
    def __init__(self, connection: _FakePostgresConnection) -> None:
        self.connection = connection
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def connect(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.connection


class RagTelemetryTests(unittest.TestCase):
    def test_finalize_computes_stage_costs_and_response_fields(self) -> None:
        telemetry = create_request_telemetry(
            user_query="What calls GailTable?",
            repo_name="nshmp-main",
            mode="hybrid",
            top_k=12,
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            rerank_enabled=True,
        )

        telemetry.record_embedding(input_tokens=120, latency_ms=18.5)
        telemetry.record_pinecone(read_units=2.0, write_units=0.0, query_count=1, latency_ms=24.0)
        telemetry.record_rerank(input_tokens=80, latency_ms=4.0)
        telemetry.record_llm(
            input_tokens=900,
            output_tokens=240,
            cached_input_tokens=100,
            latency_ms=620.0,
            model_name="gpt-4o-mini-2026-01-01",
        )
        telemetry.record_postprocess(8.5)
        telemetry.record_counts(retrieved_file_count=4, retrieved_chunk_count=12, selected_chunk_count=5)
        telemetry.mark_retrieval_complete()
        telemetry.mark_success("Answer with citations")
        telemetry.finalize()

        payload = telemetry.to_response_dict()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["retrieved_chunk_count"], 12)
        self.assertEqual(payload["selected_chunk_count"], 5)
        self.assertEqual(payload["llm_input_tokens"], 900)
        self.assertGreater(payload["generation_latency_ms"], 0.0)
        self.assertGreater(payload["estimated_cost_usd"], 0.0)
        self.assertEqual(payload["model_name"], "gpt-4o-mini-2026-01-01")

    def test_store_persist_recent_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag-telemetry.sqlite3"
            store = TelemetryStore(str(db_path))

            first = create_request_telemetry(
                user_query="Explain hazard flow",
                repo_name="nshmp-main",
                mode="chat",
                top_k=5,
                model_name="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                rerank_enabled=True,
            )
            first.record_embedding(input_tokens=50, latency_ms=10.0)
            first.record_pinecone(read_units=1.0, query_count=1, latency_ms=22.0)
            first.record_llm(input_tokens=400, output_tokens=120, latency_ms=320.0)
            first.record_counts(retrieved_file_count=3, retrieved_chunk_count=5, selected_chunk_count=5)
            first.mark_success("ok")
            first.finalize()
            store.persist(first)

            second = create_request_telemetry(
                user_query="Show call chain for GailTable",
                repo_name="nshmp-main",
                mode="hybrid",
                top_k=12,
                model_name="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                rerank_enabled=True,
            )
            second.record_embedding(input_tokens=65, latency_ms=11.0)
            second.record_pinecone(read_units=2.5, query_count=2, latency_ms=34.0)
            second.record_counts(retrieved_file_count=5, retrieved_chunk_count=9, selected_chunk_count=4)
            second.mark_failure("llm", "timeout")
            second.finalize()
            store.persist(second)

            recent = store.recent(limit=5)
            summary = store.summary()

            self.assertEqual(len(recent), 2)
            self.assertEqual(summary["request_count"], 2)
            self.assertGreaterEqual(summary["avg_latency_ms"], 0.0)
            self.assertGreaterEqual(summary["avg_cost_usd_est"], 0.0)
            self.assertGreater(summary["failure_rate"], 0.0)
            self.assertTrue(any(item["mode"] == "hybrid" for item in summary["cost_by_mode"]))
            self.assertTrue(any(item["repo_name"] == "nshmp-main" for item in summary["cost_by_repo"]))
            self.assertTrue(any(item["model_name"] == "gpt-4o-mini" for item in summary["cost_by_model"]))

    def test_main_telemetry_endpoints_return_store_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag-telemetry.sqlite3"
            store = TelemetryStore(str(db_path))
            telemetry = create_request_telemetry(
                user_query="Find files referencing ground motion models",
                repo_name="nshmp-main",
                mode="hybrid",
                top_k=12,
                model_name="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                rerank_enabled=True,
            )
            telemetry.record_embedding(input_tokens=75, latency_ms=12.0)
            telemetry.record_pinecone(read_units=1.5, query_count=1, latency_ms=25.0)
            telemetry.record_counts(retrieved_file_count=4, retrieved_chunk_count=8, selected_chunk_count=5)
            telemetry.mark_success("done")
            telemetry.finalize()
            store.persist(telemetry)

            original_enabled = main.settings.telemetry_enabled
            original_get_store = main.get_telemetry_store
            try:
                main.settings.telemetry_enabled = True
                main.get_telemetry_store = lambda: store  # type: ignore[assignment]

                summary_response = main.telemetry_summary()
                requests_response = main.telemetry_requests(limit=10)

                self.assertEqual(summary_response.request_count, 1)
                self.assertEqual(len(requests_response.requests), 1)
                self.assertEqual(requests_response.requests[0]["request_id"], telemetry.request_id)
            finally:
                main.settings.telemetry_enabled = original_enabled
                main.get_telemetry_store = original_get_store  # type: ignore[assignment]

    def test_store_uses_postgres_backend_when_database_url_is_configured(self) -> None:
        fake_conn = _FakePostgresConnection()
        fake_psycopg = _FakePsycopgModule(fake_conn)

        with patch("app.telemetry.psycopg", fake_psycopg), patch("app.telemetry.dict_row", object()):
            store = TelemetryStore(database_url="postgresql://telemetry:test@localhost:5432/telemetry")

            telemetry = create_request_telemetry(
                user_query="Show call chain for GailTable",
                repo_name="nshmp-main",
                mode="hybrid",
                top_k=12,
                model_name="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                rerank_enabled=True,
            )
            telemetry.mark_success("ok")
            telemetry.finalize()
            store.persist(telemetry)
            store.recent(limit=10)
            store.summary()

        self.assertEqual(store.backend, "postgres")
        self.assertTrue(fake_psycopg.calls)
        self.assertTrue(any("ON CONFLICT (request_id) DO UPDATE SET" in query for query, _ in fake_conn.executed))
        self.assertTrue(any("LIMIT %s" in query for query, _ in fake_conn.executed))


if __name__ == "__main__":
    unittest.main()
