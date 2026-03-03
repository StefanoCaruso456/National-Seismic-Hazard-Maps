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
python -m app.ingest --repo-root .. --namespace nshmp-main
```

If chunking logic or metadata schema changes, reindex cleanly:

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main --delete-existing
```

Dry run to validate syntax-aware chunking (200-500 tokens):

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main --dry-run
```

## Audit / Debug

Local audit (validates file discovery + chunk sizing; optionally checks Pinecone vector counts if `PINECONE_API_KEY` is configured):

```bash
python -m app.audit --repo-root .. --namespace nshmp-main
```

End-to-end retrieval smoke test (adds semantic query probe + PASS/FAIL gate summary):

```bash
python -m app.audit --repo-root .. --namespace nshmp-main --smoke-query "Where is subroutine hazallXL defined?" --top-k 5
```

Runtime debug endpoints are disabled by default. To enable them, set `APP_DEBUG=true` (local `.env` or Railway env var):

- `GET /api/debug/pinecone` shows index visibility, raw stats, and parsed `namespace_vector_count`
- `GET /api/debug/repo-scan` confirms the deployed container can see Fortran files
- `GET /api/debug/sample-chunks?file_path=src/...` shows a sample chunk preview for one file

## API endpoints

- `POST /api/search` semantic vector search (returns snippets + file/line citations)
- `POST /api/query` RAG answer generation using retrieved Pinecone context
- `GET /health` deployment health

## Retrieval quality and reliability defaults

The backend uses:

- Retry with exponential backoff for OpenAI + Pinecone calls
- In-memory query embedding cache (LRU)
- Candidate expansion (`top_k * multiplier`) then hybrid rerank (semantic + lexical identifier overlap)
- Dedup by `file_path + line range`
- Minimum hybrid score filter (fallback to best available results if no match passes threshold)
- Optional namespace fallback (`PINECONE_FALLBACK_NAMESPACE`)

## Railway settings

- Source repo: `StefanoCaruso456/National-Seismic-Hazard-Maps`
- Root Directory: `backend`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Healthcheck Path: `/health`

Required env vars:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default: `legacylens-openai-index`)
- `PINECONE_NAMESPACE` (default: `nshmp-main`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `EXTERNAL_CALL_RETRIES` (default: `3`)
- `EXTERNAL_CALL_BACKOFF_SECONDS` (default: `0.5`)
- `EMBEDDING_CACHE_SIZE` (default: `512`)
- `RETRIEVAL_CANDIDATE_MULTIPLIER` (default: `4`)
- `RETRIEVAL_MAX_CANDIDATES` (default: `40`)
- `RETRIEVAL_LEXICAL_WEIGHT` (default: `0.25`)
- `RETRIEVAL_MIN_HYBRID_SCORE` (default: `0.35`)
- `RAG_MAX_CONTEXT_CHUNKS` (default: `6`)
- `PINECONE_FALLBACK_NAMESPACE` (optional; empty by default)

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
