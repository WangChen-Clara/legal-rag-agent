# Title 12 Context Retrieval Evaluation

## Setup

- Schema: `title12-context-retrieval-eval-v1`
- Split: development
- Questions: 20
- Index: `data\indexes\title12_bge_large_2025-09-01\vector_db.index`
- Model: `bge-large-en-v1.5`
- Device: `cuda`
- Holdout retrieval inspected: no

## Metrics

| Variant | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | Elapsed ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.400 | 0.800 | 0.900 | 0.925 | 0.566 | 2362.7 |
| explicit_only | 0.800 | 0.950 | 0.950 | 0.975 | 0.846 | 1731.4 |
| semantic_cross_reference | 0.050 | 0.700 | 0.850 | 0.850 | 0.253 | 1650.1 |
| full_context | 0.800 | 0.950 | 1.000 | 1.000 | 0.863 | 1903.8 |

## Focus Questions

| Variant | Question | First complete rank | Recall@10 | Top sources |
|---|---|---:|---:|---|
| baseline | title12-dev-q001 | - | 0.00 | 211.10:semantic, 211.8:semantic, 211.10:semantic, 211.10:semantic, 211.33:semantic, 211.10:semantic, 211.9:semantic, 211.32:semantic, 211.9:semantic, 211.10:semantic |
| baseline | title12-dev-q018 | - | 0.50 | 217.135:semantic, 217.135:semantic, 217.135:semantic, 324.135:semantic, 3.135:semantic, 324.135:semantic, 324.135:semantic, 3.135:semantic, 3.135:semantic, 1240.38:semantic |
| explicit_only | title12-dev-q001 | 1 | 1.00 | 211.31:explicit_citation, 211.10:semantic, 211.8:semantic, 211.10:semantic, 211.10:semantic, 211.33:semantic, 211.10:semantic, 211.9:semantic, 211.32:semantic, 211.9:semantic |
| explicit_only | title12-dev-q018 | - | 0.50 | 217.135:explicit_citation, 217.135:semantic, 217.135:semantic, 324.135:semantic, 3.135:semantic, 324.135:semantic, 324.135:semantic, 3.135:semantic, 3.135:semantic, 1240.38:semantic |
| semantic_cross_reference | title12-dev-q001 | - | 0.00 | 211.1:cross_reference, 211.21:cross_reference, 211.10:semantic, 211.8:semantic, 211.10:semantic, 211.10:semantic, 211.33:semantic, 211.10:semantic, 211.9:semantic, 211.32:semantic |
| semantic_cross_reference | title12-dev-q018 | 4 | 1.00 | 217.134:cross_reference, 217.142:cross_reference, 3.134:cross_reference, 217.135:semantic, 217.135:semantic, 217.135:semantic, 324.135:semantic, 3.135:semantic, 324.135:semantic, 324.135:semantic |
| full_context | title12-dev-q001 | 1 | 1.00 | 211.31:explicit_citation, 211.10:semantic, 211.8:semantic, 211.10:semantic, 211.10:semantic, 211.33:semantic, 211.10:semantic, 211.9:semantic, 211.32:semantic, 211.9:semantic |
| full_context | title12-dev-q018 | 2 | 1.00 | 217.135:explicit_citation, 217.134:cross_reference, 217.142:cross_reference, 217.135:semantic, 217.135:semantic, 324.135:semantic, 3.135:semantic, 324.135:semantic, 324.135:semantic, 3.135:semantic |

## Failures At 10

| Variant | Question |
|---|---|
| baseline | title12-dev-q001 |
| baseline | title12-dev-q018 |
| explicit_only | title12-dev-q018 |
| semantic_cross_reference | title12-dev-q001 |
| semantic_cross_reference | title12-dev-q003 |
| semantic_cross_reference | title12-dev-q009 |

## Interpretation

`full_context` is a candidate strategy named `citation_aware_context_retrieval`,
not a final default. It should be enabled when the user query contains an explicit
CFR section reference. On this Development split it combines explicit citation
priority with one-hop cross-reference expansion from the explicit section.

Compared with baseline, `full_context` removes the q001 and q018 Top-10 failures
and improves Hit@10, Recall@10, and MRR@10. `semantic_cross_reference` performs
worse than baseline, so cross-reference expansion from ordinary semantic hits
should not be enabled by default.

Remaining risks before Holdout: an existing but incorrect user citation can still
pollute evidence, cross-reference expansion is only one hop and reference-order
based, and long sections may require more than the first chunk or a parent-section
fetch path.
