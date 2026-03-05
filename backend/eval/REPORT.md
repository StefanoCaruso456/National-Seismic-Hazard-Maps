# Retrieval Evaluation Report

Generated: 2026-03-05T06:34:28.157794+00:00
Endpoint: https://www.nationalseismichazardmaps.com/api/search
Queries: 35

## Metrics (Baseline Proxy vs Improved)

| Metric | Baseline Proxy | Improved | Delta |
|---|---:|---:|---:|
| Precision@5 | 0.3371 | 0.3829 | +0.0458 |
| MRR@5 | 0.4724 | 0.4595 | -0.0129 |
| Recall@50 | 0.7429 | 0.8000 | +0.0571 |

## Latency

| Profile | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| Baseline Proxy | 1279.42 | 1797.27 | 1292.57 |
| Improved | 1348.21 | 1860.63 | 1345.71 |

## Citation Accuracy

- Baseline Proxy: 0.6000
- Improved: 0.5000

## Coverage

- Fortran files discovered: 28
- Files with chunks: 28 (100.00%)
- Chunks with required metadata: 2050/2050 (100.00%)

## Targets

- Precision@5 > 0.70: FAIL
- Latency p95 < 3000ms: PASS
- File coverage = 100%: PASS
- Chunk metadata coverage = 100%: PASS

## Notes

- Baseline Proxy is computed from semantic rank ordering of the same retrieved candidate set (before deterministic rerank fusion).
- Improved uses final hybrid rerank ordering.
