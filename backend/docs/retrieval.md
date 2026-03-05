# Retrieval Pipeline (Fortran + Pinecone)

## Goals
- Improve retrieval quality for legacy Fortran identifiers and control-flow questions.
- Preserve file/line-grounded citations.
- Keep query latency stable with deterministic, low-cost scoring.

## End-to-End Flow
1. Query normalization and decomposition (`rewrite_and_decompose_query` in `app/main.py`):
   - Canonicalizes the user question.
   - Generates focused subqueries for symbol/location/dependency patterns.

2. Optional lexical candidate generation (`lexical_candidate_files`):
   - Triggered for identifier-heavy queries when `RETRIEVAL_IDENTIFIER_LEXICAL_ENABLED=true`.
   - Uses `rg` to find exact token hits, call sites, and definition-like lines.
   - Produces ranked `candidate_files` and normalized file scores.

3. Pinecone retrieval (`retrieve_citations_and_chunks`):
   - Embeds each subquery using the configured embedding model.
   - Queries Pinecone with optional metadata filter (`file_path`, `language`, `source_type`, `repo`).
   - Uses query-result cache (`RETRIEVAL_QUERY_CACHE_TTL_SECONDS`) and embedding cache (`EMBEDDING_CACHE_SIZE`).

4. Deterministic rerank (`rerank_matches`):
   - Candidate pool is at least top-20 before rerank.
   - Score features:
     - normalized semantic score
     - token overlap
     - exact identifier hits
     - symbol/module match boosts
     - Fortran action statement boosts (`call/read/write/open/allocate/...`)
     - lexical file boost
     - comment-density penalty
   - Returns final ranked chunks and debug ranks (`semantic_rank`, `rerank_rank`).

5. Context assembly (`expand_context_for_citations`):
   - Expands to parent routine/module when available.
   - Adds neighbor lines and optional file header context.
   - Preserves citation line ranges for answer grounding.

## Fortran-Aware Ingestion
Implemented in `app/ingest.py`:
- Recursive file discovery with extension allowlist and build/vendor exclusion.
- Encoding fallback (`utf-8` then `latin-1`).
- Fortran boundary parsing (`module/subroutine/function/...` with `end ...`).
- Metadata on each chunk:
  - `file_path`, `start_line`, `end_line`
  - `language=fortran`
  - `symbol_type`, `symbol_name`, `module_name`
  - `contains_block`, `imports`
  - embedding metadata (`embedding_model`, `embedding_dimension`, schema version)

## Feature Flags
`app/config.py`:
- `RETRIEVAL_IDENTIFIER_LEXICAL_ENABLED` (default `true`)
- `RETRIEVAL_DETERMINISTIC_RERANK_ENABLED` (default `true`)
- `RETRIEVAL_CONTEXT_EXPANSION_ENABLED` (default `true`)

These flags are useful for controlled evaluation and rollback.

## Telemetry
`retrieve_with_optional_uploads` and `retrieve_citations_and_chunks` emit debug timings:
- rewrite
- lexical candidate generation
- embedding
- Pinecone query
- rerank
- context assembly
- total latency

Debug candidate rows include semantic rank and rerank rank for before/after comparison.

## Failure Modes
- Identifier not present in codebase: retrieval may still return semantically related chunks; focus-term guardrail can suppress unsupported answers.
- Very broad queries: high recall but weaker precision unless constrained by `path_prefix` or lexical candidates.
- Missing local repo files at runtime: context expansion cannot validate/expand line ranges and falls back to chunk text.
- Cold cache or Pinecone latency spikes: p95 can exceed target in peak load.

## Known Tradeoffs
- Deterministic rerank is cheap and predictable but not as semantically expressive as cross-encoders.
- Lexical `rg` stage improves exact-token recall but can add small overhead for identifier-heavy queries.
- Context expansion improves grounding but increases prompt payload size when parent blocks are large.
