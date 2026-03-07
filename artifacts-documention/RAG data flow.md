# RAG Data Flow

This document shows the end-to-end request path for the RAG-backed modes in LegacyLens.

## What Happens

1. The user submits a prompt from the UI with a mode, filters, and optional uploads.
2. The backend validates the request and starts request-scoped telemetry.
3. For Graph/Hybrid, the graph layer first narrows the search to likely files, symbols, and dependencies.
4. The query is embedded with OpenAI and searched against Pinecone for relevant code chunks.
5. Retrieved chunks are reranked, expanded to line context, and assembled into the final prompt context.
6. The LLM generates the answer, and the API returns the response with citations, confidence, and telemetry.

## Mermaid Chart

```mermaid
flowchart TD
    A[User enters prompt in UI] --> B[Frontend collects mode, question, filters, uploads]
    B --> C[POST request to API]
    C --> D[Backend validates request and starts telemetry]
    D --> E{UI mode}

    E -->|Audit / Diagrams| F[Build repo overview + select file excerpts]
    F --> G[Direct LLM prompt]
    G --> H[Generate audit report or diagram]
    H --> Z[Finalize telemetry + send response]

    E -->|Chat / Search / Patterns / Dependencies / Graph-Hybrid| I[Normalize mode, scope, filters]
    I --> J{Graph-Hybrid?}

    J -->|Yes| K[GitNexus graph/context/impact lookup]
    K --> L[Candidate files from graph]
    J -->|No| M[Standard retrieval routing]

    L --> N[Optional lexical identifier search]
    M --> N
    N --> O[Embed query or subqueries]
    O --> P[Pinecone retrieves chunks]
    P --> Q[Deterministic rerank + guardrails]
    Q --> R[Context assembly + line expansion]

    R --> S{Mode-specific output}
    S -->|Chat / Graph-Hybrid| T[LLM generates cited answer]
    S -->|Search| U[Return ranked chunks]
    S -->|Patterns| V[Extract pattern examples]
    S -->|Dependencies| W[Build dependency summary / edges]

    T --> Z
    U --> Z
    V --> Z
    W --> Z

    Z --> AA[Frontend renders answer, graph, citations, confidence, telemetry]
```

## Key Distinction

- Graph/Hybrid is graph-guided RAG: graph narrows the search, and RAG supplies the line-level evidence.
- Audit and Diagrams are direct LLM modes and do not run through the Pinecone retrieval pipeline.
