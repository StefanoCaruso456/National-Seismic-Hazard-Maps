# LegacyLens Backend

FastAPI RAG backend for legacy Fortran code understanding.

## Local run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Landing page:

```bash
open http://localhost:8000/
```

## Ingest Fortran codebase into Pinecone

From the `backend/` folder:

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main:v1
```

If chunking logic or metadata schema changes, reindex cleanly:

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main:v1 --delete-existing
```

Dry run to validate syntax-aware chunking (200-500 tokens):

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main:v1 --dry-run
```

## Audit / Debug

Local audit (validates file discovery + chunk sizing; optionally checks Pinecone vector counts if `PINECONE_API_KEY` is configured):

```bash
python -m app.audit --repo-root .. --namespace nshmp-main:v1
```

End-to-end retrieval smoke test (adds semantic query probe + PASS/FAIL gate summary):

```bash
python -m app.audit --repo-root .. --namespace nshmp-main:v1 --smoke-query "Where is subroutine hazallXL defined?" --top-k 5
```

Retrieval eval harness (Recall@5/10 + nDCG@10):

```bash
python -m app.eval.run_eval --dataset app/eval/dataset.jsonl --top-k 10 --out /tmp/legacylens_eval.json
```

Runtime debug endpoints are disabled by default. To enable them, set `APP_DEBUG=true` (local `.env` or Railway env var):

- `GET /api/debug/pinecone` shows index visibility, raw stats, and parsed `namespace_vector_count`
- `GET /api/debug/repo-scan` confirms the deployed container can see Fortran files
- `GET /api/debug/sample-chunks?file_path=src/...` shows a sample chunk preview for one file

## API endpoints

- `POST /api/search` mode-aware retrieval (`search|patterns|dependencies`) with snippets + file/line citations
- `POST /api/query` answer generation (`chat|graph|hybrid`) with retrieved Pinecone context
- `POST /api/search/upload` multipart search with temporary retrieval scope from attached files
- `POST /api/query/upload` multipart RAG answer with attached-file chunking + retrieval
- `POST /api/uploads/ingest` persist uploads into Pinecone attachment namespace (`attachments:<project_id>:v1`) with SHA-256 dedupe
- `GET /api/uploads` list persisted uploads for project
- `POST /api/uploads/{file_sha}/pin` pin/unpin upload metadata entry
- `DELETE /api/uploads/{file_sha}` delete attachment vectors by `file_sha`
- `GET /api/retrieval-info` retrieval scoring/limits metadata for the UI (`repo_url`, upload limits, score weights)
- `GET /health` deployment health + startup smoke gate status (`ok`/`degraded`)

`/api/search` and `/api/query` accept `debug=true` in JSON to return retrieval traces.
`/api/search/upload` and `/api/query/upload` accept multipart field `debug=true` for the same trace payload.
`mode` can be `chat|search|patterns|dependencies|graph|hybrid` (mode-aware evidence + response formatting).
`scope` can be `repo|uploads|both`; `project_id` selects attachment namespace scope.
Optional retrieval filters: `path_prefix`, `language` (`fortran|text|pdf`), `source_type` (`repo|upload|temp-upload`).
Upload persistence is opt-in via multipart field `persist_uploads=true` (default: not persisted).

## Retrieval quality and reliability defaults

The backend uses:

- Retry with exponential backoff for OpenAI + Pinecone calls
- In-memory query embedding cache (LRU)
- Candidate expansion (`top_k * multiplier`) then hybrid rerank (semantic + lexical identifier overlap)
- Exact-term guardrail for quoted/hyphenated identifiers (e.g. `CUSTOMER-RECORD`): weak/irrelevant evidence is downranked or refused
- Dedup by `file_path + line range`
- Minimum hybrid score filter (fallback to best available results if no match passes threshold)
- Optional namespace fallback (`PINECONE_FALLBACK_NAMESPACE`)
- Attachment-aware retrieval across repo namespace + persistent upload namespace
- Upload ingestion pipeline with SHA-256 dedupe and PDF/text extraction
- Query rewrite/decomposition for implementation/workflow/config style questions
- Hybrid mode: GitNexus MCP (`query/context/impact`) -> candidate files -> Pinecone constrained retrieval -> architecture-first answer with citations
- Intent router for config-style queries (prioritizes `conf/`, `etc/`, `scripts/`, run scripts, Makefile)
- Dependency graph extraction (CALL/USE/INCLUDE/COMMON) with definition resolution
- Pattern extraction for Fortran loop/IO blocks with concrete code snippets
- Evidence strength scoring (`High|Medium|Low`) with mode-aware gates

## Railway settings

- Source repo: `StefanoCaruso456/National-Seismic-Hazard-Maps`
- Root Directory: `backend`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Healthcheck Path: `/health`

Required env vars:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default: `legacylens-openai-index`)
- `PINECONE_NAMESPACE` (default: `nshmp-main:v1`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `ENFORCE_EMBEDDING_DIMENSION` (default: `true`)
- `EXTERNAL_CALL_RETRIES` (default: `3`)
- `EXTERNAL_CALL_BACKOFF_SECONDS` (default: `0.5`)
- `EMBEDDING_CACHE_SIZE` (default: `512`)
- `RETRIEVAL_CANDIDATE_MULTIPLIER` (default: `4`)
- `RETRIEVAL_MAX_CANDIDATES` (default: `40`)
- `RETRIEVAL_LEXICAL_WEIGHT` (default: `0.25`)
- `RETRIEVAL_MIN_HYBRID_SCORE` (default: `0.35`)
- `RETRIEVAL_FOCUS_TERM_GUARDRAIL_ENABLED` (default: `true`)
- `RETRIEVAL_FOCUS_TERM_ABSENT_CAP` (default: `0.20`)
- `RETRIEVAL_FOCUS_TERM_PARTIAL_COVERAGE_CAP` (default: `0.45`)
- `HYBRID_TOP_K_DEFAULT` (default: `12`)
- `HYBRID_MAX_CANDIDATE_FILES` (default: `50`)
- `GITNEXUS_ENABLED` (default: `true`)
- `GITNEXUS_BASE_URL` (default: `http://127.0.0.1:4000`; reserved for HTTP sidecar compatibility)
- `GITNEXUS_DEFAULT_REPO` (optional; defaults to current repo folder name)
- `GITNEXUS_MCP_COMMAND` (default: `npx -y gitnexus@latest mcp`)
- `GITNEXUS_CALL_TIMEOUT_SECONDS` (default: `30`)
- `GITNEXUS_STARTUP_TIMEOUT_SECONDS` (default: `45`)
- `GITNEXUS_BOOTSTRAP_ENABLED` (default: `false`; when true, clones/indexes a configured source repo on startup)
- `GITNEXUS_BOOTSTRAP_REPO_URL` (optional source git URL used by bootstrap)
- `GITNEXUS_BOOTSTRAP_REPO_PATH` (default: `/tmp/nshmp-main`)
- `GITNEXUS_BOOTSTRAP_REPO_REF` (optional branch/tag for bootstrap clone)
- `GITNEXUS_ANALYZE_COMMAND` (default: `gitnexus analyze`)
- `GITNEXUS_ANALYZE_TIMEOUT_SECONDS` (default: `180`)
- `REPO_ROOT_OVERRIDE` (optional absolute path used for repo file reads / lexical search)
- `RAG_MAX_CONTEXT_CHUNKS` (default: `6`)
- `PINECONE_FALLBACK_NAMESPACE` (optional; empty by default)
- `STARTUP_SMOKE_MODE` (`off|warn|strict`, default: `off`)
- `STARTUP_SMOKE_QUERY` (default: `startup health probe`)
- `STARTUP_SMOKE_TOP_K` (default: `1`)

## GitNexus (Hybrid mode) setup

From repository root:

```bash
npx -y gitnexus@latest analyze .
```

MCP server command used by backend (configurable):

```bash
npx -y gitnexus@latest mcp
```

Railway production recommendation (backend-only deploy root):

```bash
GITNEXUS_BOOTSTRAP_ENABLED=true
GITNEXUS_BOOTSTRAP_REPO_URL=https://github.com/StefanoCaruso456/National-Seismic-Hazard-Maps
GITNEXUS_BOOTSTRAP_REPO_PATH=/tmp/nshmp-main
GITNEXUS_DEFAULT_REPO=nshmp-main
REPO_ROOT_OVERRIDE=/tmp/nshmp-main
```

## MVP hard-gate mapping

- Ingest legacy Fortran: `python -m app.ingest`
- Syntax-aware chunking: section-aware split by `program|subroutine|function|module|interface|block data`, then 200-500 token chunks
- Embeddings: OpenAI `text-embedding-3-small`
- Vector DB storage: Pinecone `upsert` with metadata
- Semantic search: `POST /api/search`
- Natural-language web query: landing page + `POST /api/query`
- Relevant snippets + line refs: citation metadata (`file_path`, `line_start`, `line_end`, `snippet`)
- Basic answer generation: OpenAI chat model with retrieved context
- Deployed publicly: Railway custom domain
