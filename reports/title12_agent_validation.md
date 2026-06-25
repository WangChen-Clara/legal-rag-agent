# Title 12 Agent Validation

- Schema: `title12-agent-validation-v1`
- Status: `passed`
- Questions: 2
- Max steps: 4
- Max fetch sections: 2
- Index: `data\indexes\title12_bge_large_2025-09-01\vector_db.index`
- Sections: `data\canonical\title12_2025-09-01\sections.jsonl`
- Device: `cuda`
- Holdout retrieval inspected: no
- LLM called: no

## Validations

| Question | Status | Termination | Citations | Steps |
|---|---|---|---|---|
| title12-dev-q001 | passed | completed | 12 CFR 211.31 (2025-09-01) | search_regulations, fetch_section, final_answer |
| title12-dev-q018 | passed | completed | 12 CFR 217.135 (2025-09-01), 12 CFR 217.134 (2025-09-01) | search_regulations, fetch_section, fetch_section, final_answer |

## Interpretation

This validates the Phase 5 minimal deterministic Agent Harness. The loop is:
`search_regulations` -> optional `fetch_section` -> `final_answer`, bounded by
`max_steps`. The generated answer is template-based and is intended to validate
control flow and evidence/citation plumbing, not final legal answer quality.

Remaining work: add a structured LLM decision step, improve answer generation,
add durable JSON trace output, and later fold in `ToolRegistry` / `ToolResult`
once the harness shape is stable.
