# LegacyLens + NSHMP Fortran

This repository contains two things:

1. The original National Seismic Hazard Mapping Project (NSHMP) Fortran codebase and data/config assets.
2. LegacyLens, a retrieval-augmented code intelligence app built to analyze this legacy Fortran system with line-level evidence and architecture context.

## Problem / Solution / What / Why / How

### Problem

Legacy scientific Fortran systems are hard to work with:

- critical logic is spread across many files and decades of style drift
- call chains and dependencies are expensive to trace manually
- onboarding often takes months before engineers can make safe changes

For teams, this means codebase insight that traditionally takes years of accumulated context now blocks delivery speed and change safety.

### Solution

LegacyLens combines vector retrieval, lexical symbol search, and graph retrieval into one deterministic workflow so engineers can move from question to grounded evidence quickly.

### What

LegacyLens is a code intelligence tool for NSHMP-style legacy repositories that returns:

- architecture context (entrypoints, flows, impact)
- line-level evidence and citations
- retrieval diagnostics and confidence signals

### Why

The goal is to compress understanding time:

- from slow manual archaeology to fast, evidence-backed exploration
- from uncertain edits to traceable, citation-grounded decisions
- from siloed expert knowledge to team-accessible understanding

In practice: what used to take weeks or months to trace can often be scoped in hours.

### How

1. Route each query to the best retrieval plan (vector, keyword+vector, graph+vector, or escalation).
2. Use graph signals to identify high-probability candidate files.
3. Retrieve and rerank Pinecone chunks with deterministic scoring and dedupe.
4. Expand context and enforce identifier guardrails.
5. Return architecture + evidence with debug telemetry.

## What LegacyLens does

LegacyLens helps engineers query a large legacy Fortran repository with:

- semantic retrieval from Pinecone
- exact-identifier lexical retrieval
- graph-assisted architecture retrieval via GitNexus MCP
- deterministic reranking and grounded citations
- a hybrid UI showing architecture graph + evidence panel

## Feature Inventory

### Core retrieval

- Multi-plan retrieval router:
  - `VECTOR_ONLY`
  - `KEYWORD_ONLY`
  - `GRAPH_ONLY`
  - `KEYWORD_PLUS_VECTOR`
  - `GRAPH_PLUS_VECTOR`
  - `GRAPH_PLUS_KEYWORD_PLUS_VECTOR` (single escalation path)
- Deterministic low-confidence gate and one-step escalation.
- Pinecone candidate retrieval at larger top-k followed by rerank and dedupe.
- Metadata filters (`file_path`, `language`, `source_type`, `repo`) for constrained retrieval.

### Hybrid graph + evidence

- GitNexus MCP integration (stdio) with tool calls for:
  - graph query
  - symbol context
  - impact analysis (upstream/downstream)
- Candidate file extraction/ranking from graph outputs.
- Graph-constrained Pinecone retrieval with fallback to unconstrained retrieval when needed.
- Structured `route_debug` and `hybrid_debug` telemetry, including graph availability, score thresholds, fallback reason, and step timings.

### Fortran-aware ingestion

- Recursive discovery for Fortran extensions (`.f`, `.for`, `.f90`, `.f95`, `.f03`, `.f08`, `.inc`).
- Exclusion of non-source directories (`.git`, `.venv`, `build`, `dist`, `vendor`, `node_modules`, etc.).
- Encoding fallback (`utf-8`, then `latin-1`).
- Syntax-aware splitting by Fortran units (`program`, `module`, `subroutine`, `function`, `interface`, `block data`) including `contains` and `use` extraction.
- Line-accurate chunk metadata and embedding metadata.
- Namespace delete/reindex support and dry-run mode.

### Retrieval quality controls

- Identifier-aware lexical candidate generation (`rg`-based).
- Deterministic rerank features:
  - semantic score
  - token overlap
  - exact identifier hits
  - symbol/module boosts
  - Fortran action statement boosts
  - comment-density penalty
- Context expansion beyond chunk:
  - parent routine/module
  - neighbor lines
  - optional header context
- Focus-term guardrails for identifier-sensitive questions.
- Test-file retrieval policy:
  - tests stay indexed
  - excluded by default in retrieval
  - auto-included for test intent queries
  - one fallback pass with tests if low confidence

### Product/UI capabilities

- Modes: `Chat`, `Hybrid`, `Search`, `Code Patterns`, `Dependencies`, `Diagrams`, `Run Audit`.
- Hybrid architecture panel:
  - interactive 3D graph canvas
  - node legend with counts by kind
  - top graph flows
  - candidate scope and diagnostics
- Hybrid evidence panel:
  - citation cards with file + line range
  - fallback and filter diagnostics
- Debug trace panel (`debug=true`) for route, retrieval timings, rerank data, and graph diagnostics.
- Upload workflows:
  - temporary upload retrieval
  - optional persistent upload ingestion into Pinecone attachment namespaces
  - pin/list/delete uploaded sources

### Evaluation and audit tooling

- Local audit CLI:
  - indexing prerequisites
  - chunk quality window checks
  - optional semantic smoke query
  - gate summary
- Retrieval evaluation harness:
  - Precision@5
  - MRR@5
  - Recall@50
  - latency p50/p95
  - citation accuracy sampling
  - index/metadata coverage
- Test suite for router, hybrid pipeline, retrieval enhancements, focus guardrails, and GitNexus client behavior.

## Stack and Frameworks

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic + pydantic-settings
- python-dotenv
- python-multipart
- pypdf

### Retrieval and AI

- OpenAI API:
  - embeddings: `text-embedding-3-small`
  - chat generation: `gpt-4o-mini`
- Pinecone vector database
- GitNexus MCP (stdio) for graph retrieval

### Frontend

- Server-rendered static web app (`backend/app/static`)
- Vanilla JavaScript + CSS
- 3D graph rendering:
  - `three.js`
  - `3d-force-graph`

### Deployment

- Railway (service root: `backend`)

## RAG System (How it works)

1. User asks a question in one of the app modes.
2. Router detects query signals (identifier-heavy, structure intent, etc.) and picks a retrieval plan.
3. Depending on plan:
   - vector retrieval (Pinecone),
   - keyword retrieval + vector,
   - graph retrieval + vector.
4. If graph is used, GitNexus returns process/context/impact signals and candidate files.
5. Pinecone retrieval runs with metadata filters when candidate files exist.
6. Results are deterministically reranked, deduped, optionally expanded for context, and guarded for focus terms.
7. Response is returned with citations, evidence strength, and debug telemetry.
8. Hybrid mode additionally renders architecture context and evidence side by side.

## Quick Start (Local)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Ingestion and Indexing

```bash
cd backend
python -m app.ingest --repo-root .. --namespace nshmp-main:v1
```

Reindex:

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main:v1 --delete-existing
```

Dry-run:

```bash
python -m app.ingest --repo-root .. --namespace nshmp-main:v1 --dry-run
```

## GitNexus setup (Hybrid mode)

From repo root:

```bash
npx -y gitnexus@latest analyze .
```

MCP command expected by backend:

```bash
npx -y gitnexus@latest mcp
```

## Historical NSHMP Context

This repository houses the legacy codes used to generate the [2008](http://pubs.usgs.gov/of/2008/1128/) and [2014](http://pubs.usgs.gov/of/2014/1091/) updates to the NSHM for the conterminous US. The 2008 codes are tagged as [nshm2008r3](https://github.com/usgs/nshmp-haz-fortran/tree/nshm2008r3). The 2014 codes are tagged as [nshm2014r1](https://github.com/usgs/nshmp-haz-fortran/tree/nshm2014r1). These codes include configuration and data files used with [GFortran](http://gcc.gnu.org/fortran/) for published hazard maps and data products.

Note: the legacy Fortran code is provided as-is, with limited support, and has been superseded by [nshmp-haz](https://github.com/usgs/nshmp-haz).

For 2014 update Q&A and errata, see the project [wiki](https://github.com/usgs/nshmp-haz-fortran/wiki).
