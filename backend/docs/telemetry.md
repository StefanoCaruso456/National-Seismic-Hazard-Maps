# RAG Telemetry

LegacyLens records one telemetry object per RAG request and persists it after completion or failure.

## What is captured

- request identity: `request_id`, `created_at`, `user_query`, `repo_name`, `mode`
- model config: `model_name`, `embedding_model`, `top_k`, `rerank_enabled`
- stage latencies:
  - `embedding_latency_ms`
  - `pinecone_query_latency_ms`
  - `rerank_latency_ms`
  - `llm_latency_ms`
  - `postprocess_latency_ms`
  - `time_to_first_retrieval_ms`
  - `time_to_final_answer_ms`
- usage:
  - Pinecone read/write units and query count
  - embedding input tokens
  - LLM input/output/cached tokens
  - rerank input tokens
- retrieval counts:
  - `retrieved_file_count`
  - `retrieved_chunk_count`
  - `selected_chunk_count`
- status:
  - `success` or `failure`
  - `stage_failed`
  - `error_message`

## Persistence

Telemetry is stored in the table `rag_request_telemetry`.

Backend selection:

- Railway / production: PostgreSQL via `TELEMETRY_DATABASE_URL` or `DATABASE_URL`
- local fallback: SQLite at `backend/app/data/rag_telemetry.sqlite3`

Overrides:

- `TELEMETRY_DATABASE_URL=postgresql://...`
- `TELEMETRY_DB_PATH=/path/to/rag_telemetry.sqlite3`

Schema reference:

- `backend/docs/rag_request_telemetry.sql`

## Pricing

Cost estimation is config-driven in:

- `backend/app/pricing.py`
- `backend/app/config.py`

Environment overrides:

- `PRICING_PINECONE_READ_UNIT_PER_MILLION_USD`
- `PRICING_PINECONE_WRITE_UNIT_PER_MILLION_USD`
- `PRICING_LLM_INPUT_PER_MILLION_USD`
- `PRICING_LLM_CACHED_INPUT_PER_MILLION_USD`
- `PRICING_LLM_OUTPUT_PER_MILLION_USD`
- `PRICING_EMBEDDING_INPUT_PER_MILLION_USD`
- `PRICING_RERANK_INPUT_PER_MILLION_USD`

## Pinecone RU source

Pinecone usage is taken from the raw query response:

1. `query_index_with_cache(...)` executes `index.query(...)`
2. `parse_pinecone_usage(...)` extracts `response.usage.read_units` and `response.usage.write_units`
3. the request telemetry object records those values via `telemetry.record_pinecone(...)`

The code path lives in:

- `backend/app/main.py`
- `backend/app/telemetry.py`

## API

- `GET /api/telemetry/summary`
- `GET /api/telemetry/requests?limit=25`

## UI

Telemetry appears in two places:

- response-level request telemetry chips under assistant answers
- the debug panel summary and recent request list
