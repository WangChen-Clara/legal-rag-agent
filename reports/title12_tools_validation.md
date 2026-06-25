# Title 12 Tools Validation

- Schema: `title12-tools-validation-v1`
- Status: `passed`
- Search questions: 2
- Fetch section: `12 CFR 217.134(a)(1)`
- Index: `data\indexes\title12_bge_large_2025-09-01\vector_db.index`
- Sections: `data\canonical\title12_2025-09-01\sections.jsonl`
- Device: `cuda`
- Holdout retrieval inspected: no
- LLM called: no

## Search Validations

| Question | Status | Top evidence |
|---|---|---|
| title12-dev-q001 | passed | 211.31:explicit_citation, 211.10:semantic, 211.8:semantic, 211.10:semantic, 211.10:semantic |
| title12-dev-q018 | passed | 217.135:explicit_citation, 217.134:cross_reference, 217.142:cross_reference, 217.135:semantic, 217.135:semantic |

## Fetch Section

- Status: `passed`
- Section: `217.134`
- Version date: `2025-09-01`
- Safe for citation: `True`
- Source URL: `https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134`

## Interpretation

The Phase 4 tool prototype is read-only and validates against the fixed Title 12
`2025-09-01` canonical corpus. `search_regulations` returns structured evidence
only; it does not generate an answer. `fetch_section` returns the full official
parent section for citation display or later verification.

Remaining work: define a formal `Tool` interface, add `ToolRegistry` and
`ToolResult`, add timeout/error typing, implement `verify_citation`, and implement
`compare_versions`.
