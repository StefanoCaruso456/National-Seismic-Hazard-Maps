# Retrieval Router

## Purpose
The router selects a deterministic retrieval plan per query before running retrieval primitives.

## Plans
- `VECTOR_ONLY`
- `KEYWORD_ONLY`
- `GRAPH_ONLY`
- `KEYWORD_PLUS_VECTOR`
- `GRAPH_PLUS_VECTOR`
- `GRAPH_PLUS_KEYWORD_PLUS_VECTOR` (escalation only)

## Signals
Implemented in `app/router.py` via `detect_route_signals`:
- `identifier_terms`: detected exact/symbol-like terms (CALL targets, ALLCAPS, snake_case, digit-heavy tokens)
- `structure_intent`: query contains structure intent phrases
- `has_extensions`: query references Fortran-like file extensions (`.f`, `.f90`, `.inc`, `.for`)

## Routing Rules
Applied by `select_retrieval_plan`:
1. Identifier-heavy query -> `KEYWORD_PLUS_VECTOR`
2. Structure-intent query -> `GRAPH_PLUS_VECTOR`
3. Otherwise -> `VECTOR_ONLY`
4. `mode=graph` override -> `GRAPH_ONLY`

## Low-Confidence Gate
Deterministic gate implemented by `low_confidence_reason`:
- `no_matches`: zero results
- `duplicate_ranges`: duplicate `(file_path, line_start, line_end)` ranges
- `insufficient_unique_results`: fewer than 3 unique ranges
- `top_score_below_threshold`: max score `< 0.45`

## Escalation Rules
Single-step escalation, max once (`escalated_plan`):
- `VECTOR_ONLY` -> `KEYWORD_PLUS_VECTOR`
- `GRAPH_PLUS_VECTOR` -> `GRAPH_PLUS_KEYWORD_PLUS_VECTOR`
- Any other plan -> no escalation

## Pinecone Requirements
Router execution in `/api/query` enforces:
- candidate retrieval `top_k = 20`
- `include_metadata = true`
- rerank and keep top `5`
- metadata file filter when candidate files exist
- dedupe by `(file_path, line_start, line_end)`

## route_debug
Every `/api/query` response includes `debug.route_debug` with:
- `route`
- `signals`
- `budgets`
- `steps` (timings + candidates)
- `escalation`

## Extension Points
- Add new intent phrases in `STRUCTURE_HINTS`
- Extend identifier regexes for new languages/patterns
- Tune low-confidence thresholds (`LOW_CONFIDENCE_TOP_SCORE_THRESHOLD`, `LOW_CONFIDENCE_MIN_UNIQUE_RESULTS`)
- Add additional escalation paths with strict max-one-escalation constraint
