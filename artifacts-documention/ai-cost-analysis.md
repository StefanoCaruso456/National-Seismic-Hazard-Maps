# AI Cost Analysis

Date: March 7, 2026

This document summarizes the current AI cost profile for `National-Seismic-Hazard-Maps` using observed app telemetry and the pricing configuration in the backend. All dollar values below are estimated usage costs, not vendor invoice totals.

## Scope and Data Quality

- Telemetry records embeddings, Pinecone usage, rerank usage, LLM usage, and stage latency.
- Current Railway deployment is logging telemetry to container-local SQLite, not Postgres.
- Result: the numbers below are useful and real, but the history is not durable across container restarts.

## Development and Testing Costs

Observed telemetry window before the latest Railway container reset:

- Total tracked requests: `11`
- Total estimated spend: `$0.01533674`
- Average cost per request: `$0.00139425`
- Average latency: `11.74 s`
- p50 latency: `10.52 s`
- p95 latency: `19.92 s`
- Failure rate: `0%`

### Cost by Query Mode

| Mode | Requests | Avg Cost / Request | Avg Latency |
| --- | ---: | ---: | ---: |
| Chat | 3 | $0.00209877 | 12.95 s |
| Diagrams | 2 | $0.00195421 | 14.53 s |
| Hybrid | 2 | $0.00153943 | 18.44 s |
| Dependencies | 2 | $0.00057771 | 6.60 s |
| Search | 1 | $0.00060928 | 6.12 s |
| Patterns | 1 | $0.00028846 | 5.06 s |

Insight: query mix matters. In the observed window, `chat` and `diagrams` were about 3-7x more expensive than `patterns` and `dependencies`.

## Example Per-Request Breakdown

Latest observed request before reset:

- Mode: `diagrams`
- User focus: `System Architecture: National Hazard Run`
- Total latency: `10.52 s`
- Total estimated cost: `$0.0012865`

### Cost Breakdown

| Category | Logged Usage | Calculation | Estimated Cost |
| --- | --- | --- | ---: |
| Embedding API | 6,725 input tokens | `6725 / 1,000,000 * $0.02` | $0.0001345 |
| Vector DB usage | 72 Pinecone read units | `72 / 1,000,000 * $16` | $0.0011520 |
| Rerank | 514 input tokens | `514 / 1,000,000 * $0.00` | $0.0000000 |
| LLM generation | 0 logged input/output tokens | `0 * model rates` | $0.0000000 |
| Total | - | Sum of above | $0.0012865 |

### Latency Breakdown

| Stage | Latency |
| --- | ---: |
| Embedding | 2.57 s |
| Pinecone query | 7.25 s |
| Rerank | 0.58 s |
| LLM | 0.00 s |
| Postprocess | 0.00 s |
| Total | 10.52 s |

Insight: in this sample, Pinecone dominated both latency and variable cost. Embedding cost was small, and rerank cost was effectively zero under the current pricing config.

## Production Cost Projections

The table below estimates monthly variable serving cost at different user scales.

### Assumptions

- `5` queries per user per day
- `30` days per month
- Average variable cost per request = `$0.00139425`
- Average query mix stays similar to the observed 11-request window
- Embedding cost for new code additions assumes `1,000,000` new tokens per month = `$0.02/month`
- Pinecone usage cost is included as variable read-unit cost
- Fixed Pinecone plan/storage charges and Railway hosting are not included because the app telemetry does not track vendor subscription fees

### Monthly Variable Serving Cost

| Users | Queries / Month | Estimated Cost / Month |
| --- | ---: | ---: |
| 100 | 15,000 | $20.91 |
| 1,000 | 150,000 | $209.14 |
| 10,000 | 1,500,000 | $2,091.38 |
| 100,000 | 15,000,000 | $20,913.75 |

### New Code Addition Cost

| Monthly New Code Tokens Embedded | Embedding Cost |
| --- | ---: |
| 1,000,000 | $0.02 |
| 10,000,000 | $0.20 |
| 100,000,000 | $2.00 |

Insight: based on the current pricing config, code-ingestion embedding cost is negligible compared with live query serving cost.

## What This Means

- The current cost driver is retrieval, especially Pinecone read usage.
- Small changes in per-query cost become large at scale. Every extra `$0.001` per request adds about `$15,000/month` at `100,000` users with the assumptions above.
- If `Diagrams` and `Run Audit` are moved to the direct non-RAG path in production, the cost mix should shift away from Pinecone and should be re-measured.
- Before using these numbers as a final production report, telemetry should be moved from SQLite fallback to Postgres for durable tracking.
