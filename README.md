# Legal RAG Agent

A CLI-first / Python API prototype for a citation-aware Legal RAG Agent over a
fixed eCFR Title 12 snapshot.

The project demonstrates a bounded, read-only Agent Harness:

```text
question
-> search_regulations
-> optional fetch_section
-> final cited answer
```

The current demo is deterministic and template-based. It is designed to show
retrieval, tool use, evidence handling, and citation plumbing before adding remote
LLM decision-making.

## Snapshot

- Source: eCFR Title 12
- Version date: `2025-09-01`
- Corpus granularity: official section parent documents plus structured chunks
- Citation mode: fixed-snapshot URLs such as
  `https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134`

This is not a real-time legal research system and is not legal advice.

## What Works

- Official Title 12 canonical corpus builder
- Structured chunk builder
- BGE Large + FAISS retrieval index
- Citation-aware retrieval:
  - explicit CFR section priority, for example `12 CFR 211.31`
  - one-hop cross-reference expansion from an explicitly cited section
- Read-only Python tools:
  - `search_regulations`
  - `fetch_section`
- Minimal deterministic Agent Harness with max-step limits
- Development-only retrieval and Agent validation reports
- Markdown demo showing agent steps, evidence, fetched sections, and citations

## Demo

Run the deterministic Agent demo:

```powershell
cd path\to\rag_law_clean
$env:PYTHONPATH = "$PWD\src"
python scripts\demo_title12_agent.py --device cuda
```

The rendered report is:

```text
reports/title12_agent_demo.md
```

It shows two Development examples:

- q001: explicit citation retrieval for `12 CFR 211.31`
- q018: explicit citation retrieval for `12 CFR 217.135` plus cross-reference
  expansion to `12 CFR 217.134`

No remote LLM is called by the demo.

## Validation

Agent loop validation:

```powershell
python scripts\validate_title12_agent.py --device cuda
```

Tool validation:

```powershell
python scripts\validate_title12_tools.py --device cuda
```

Offline tests:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q -p no:cacheprovider
```

Latest local result: `112 passed`.

## Local Assets

Large generated assets are intentionally not part of the public repository:

- local Python environments: `.venv/`, `.conda-gpu/`
- FAISS indexes: `data/indexes/`
- raw eCFR XML: `data/raw/`
- generated canonical corpus and chunks: `data/canonical/`
- alignment and recovered historical data: `data/alignment/`, `data/recovered/`
- evaluation construction JSON files: `data/eval/*.json`
- bulky JSON reports: `reports/*.json`
- process-heavy Markdown notes are ignored by default; curated demo reports are kept

The embedding model, FAISS index, and canonical corpus are local runtime assets.
Set their paths in scripts or config for your machine. A local model path might look
like:

```text
path\to\bge-large-en-v1.5
```

The public repository is intended to contain source code, tests, lightweight
documentation, and curated Markdown demo reports, not generated corpora or indexes.

## Repository Contents

- `src/rag_law/retriever.py`: FAISS retrieval and citation-aware context retrieval
- `src/rag_law/tools.py`: read-only tool API
- `src/rag_law/agent.py`: deterministic Agent Harness
- `scripts/demo_title12_agent.py`: application-level markdown demo
- `scripts/validate_title12_agent.py`: real Agent validation on Development examples
- `scripts/validate_title12_tools.py`: real tool validation
- `reports/title12_agent_demo.md`: generated demo report

## Security

Remote LLM credentials must come from the process environment:

```powershell
$env:RAG_LAW_API_KEY = "your-key"
```

No API key should be committed to source, YAML, Markdown, Dockerfiles, notebooks, or
logs. The deterministic Agent demo does not require an API key.

## Current Limits

- The demo answer is template-based, not a final natural-language legal answer.
- The Agent loop is deterministic; it does not yet use an LLM to choose tools.
- Holdout retrieval has intentionally not been inspected during strategy design.
- Citation verification and version comparison are not implemented yet.
- The project is tied to the fixed `2025-09-01` snapshot and must not be presented
  as current law.
