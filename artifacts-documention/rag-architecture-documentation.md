# RAG Architecture Documentation

This document describes the current Retrieval-Augmented Generation architecture used in `National-Seismic-Hazard-Maps` for code understanding. It covers the RAG path only. The `Diagrams` and `Run Audit` buttons now use a separate direct LLM repo-scan path and are not part of this retrieval pipeline.

## Vector DB Selection

The system uses **Pinecone** as the vector database for repository chunk retrieval. It was chosen for four practical reasons:

- Managed vector search reduced infrastructure work compared with self-hosting FAISS, pgvector, or Elasticsearch.
- Metadata filtering is built into the query path and is heavily used here for `repo`, `file_path`, `language`, `source_type`, and namespace scoping.
- Namespace support cleanly separates the main repository index from upload-specific indexes.
- Pinecone exposes read/write usage units, which made end-to-end request telemetry and cost estimation feasible.

Tradeoffs considered:

- Pinecone adds an external dependency and usage cost.
- Pure vector search is not strong enough by itself for exact legacy-code identifier lookup, so the backend adds lexical candidate generation and deterministic reranking on top.
- Pinecone is still the evidence store, but it does not provide full architectural reasoning; Hybrid mode adds GitNexus graph signals to narrow candidate files before Pinecone retrieves line-level evidence.

## Embedding Strategy

The embedding model is **OpenAI `text-embedding-3-small`**.

Why it fits this system:

- It is inexpensive enough to support repeated subquery embedding and attachment indexing.
- It works for mixed natural-language and source-code retrieval, which matters because users ask conceptual questions and exact identifier questions in the same UI.
- The backend enforces embedding dimension consistency against the Pinecone index to avoid silent ingestion/query mismatches.

Known tradeoff:

- The model is not code-specific. That gap is compensated for with exact-token lexical search, identifier-aware boosts, module/symbol boosts, and focus-term guardrails during reranking and scoring.

## Chunking Approach

The ingestion path is **Fortran-aware and structure-first**.

Current behavior:

- Files are discovered recursively with an extension allowlist and exclusion of non-source/vendor/build directories.
- Files are read with encoding fallback (`utf-8`, then `latin-1`).
- The parser detects Fortran boundaries for `program`, `subroutine`, `function`, `module`, `interface`, and `block data`.
- Each section carries metadata such as `file_path`, `line_start`, `line_end`, `symbol_type`, `symbol_name`, `module_name`, `contains_block`, and `imports`.
- Large sections are split into bounded chunks and very small chunks are merged back into neighbors when possible.

Why this approach was chosen:

- Legacy scientific code is easier to reason about when chunk boundaries align to routines/modules instead of arbitrary token windows.
- Deterministic boundaries improve reproducibility and make line-grounded citations more trustworthy.
- This keeps ingestion cheaper and simpler than LLM-based semantic chunking.

Important limitations:

- Main repository chunking does **not** use overlap.
- It is **not** using LLM semantic chunking.
- Hierarchy is only partial: section metadata preserves routine/module context, but the storage layer is still flat chunk retrieval.
- The current ingestion and retrieval implementation is **Fortran-focused**, not COBOL-focused.

## Retrieval Pipeline

The end-to-end query flow is:

1. **Query rewrite and decomposition**  
   The backend normalizes the question and generates focused subqueries for symbol, location, and dependency-style retrieval.

2. **Optional lexical candidate generation**  
   For identifier-heavy questions, the backend runs `rg` locally to find exact token hits, definitions, and call-like lines. This produces ranked candidate files before vector search.

3. **Pinecone retrieval**  
   Each rewritten subquery is embedded and queried against Pinecone with optional metadata filters. Embedding results and query results are cached to reduce repeated latency.

4. **Deterministic rerank**  
   Candidates are reranked using:
   - semantic score
   - token overlap
   - exact identifier hits
   - symbol/module boosts
   - Fortran action-statement boosts
   - lexical file boosts
   - comment-density penalty

5. **Context assembly**  
   Retrieved chunks are expanded to neighboring lines and, when possible, to the enclosing routine/module. Optional file-header context is added for structure-oriented questions.

6. **Answer generation**  
   The LLM receives assembled context and returns a cited answer. Response telemetry is persisted with stage timings and cost estimates.

Hybrid-specific note:

- In Hybrid mode, GitNexus graph/context/impact signals can narrow candidate files first, then Pinecone still provides the final line-level evidence.

## Failure Modes

Known edge cases and weak spots:

- **Missing exact identifier in evidence**: semantic retrieval may find related code, but the focus-term guardrail can cap confidence heavily or drop unsupported answers.
- **Broad queries**: recall can be high while precision drops unless lexical candidates, file filters, or Hybrid graph constraints help narrow scope.
- **Cold cache / Pinecone latency spikes**: retrieval remains functional, but p95 latency can drift upward.
- **Graph unavailable or not indexed**: Hybrid can fall back, but structure-heavy questions lose precision.
- **Missing local repo files at runtime**: context expansion cannot re-open files and falls back to stored chunk text only.
- **Chunk-boundary misses**: because repository chunks do not overlap, some cross-boundary context can be missed until context expansion recovers it.
- **Confidence UX is intentionally conservative**: backend evidence scoring caps some modes when coverage is weak, and the UI uses stricter snippet labels than backend overall labels.
- **Language coverage is narrow**: current parsing/chunking behavior is built around Fortran source structure.

## Performance Results

What is verified today:

- The backend now has request-scoped telemetry for `embedding`, `Pinecone`, `rerank`, `LLM`, `postprocess`, and total request time.
- Current backend validation passes **46** local tests covering retrieval, routing, focus guardrails, hybrid behavior, and telemetry.
- Representative validated telemetry examples in tests show:
  - embedding: about **10-18.5 ms**
  - Pinecone query: about **22-34 ms**
  - rerank: about **4 ms**
  - LLM generation: about **320-620 ms**
  - postprocess: about **8.5 ms**

Precision/quality status:

- The repo does **not** currently include a formal offline evaluation report with metrics like Precision@K, Recall@K, or MRR.
- Quality control today is mainly architectural:
  - exact-token lexical candidate generation
  - deterministic reranking
  - focus-term guardrails
  - context expansion to enclosing routines/modules
  - test-file filtering with controlled fallback

Practical result:

- The system is strongest on file-grounded code explanation, dependency tracing, and pattern extraction in a Fortran codebase.
- It is weaker on very broad conceptual questions, missing identifiers, and queries that require structural reasoning without graph support.
