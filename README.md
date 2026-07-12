# Legal RAG Agent

[English](README.md) | [简体中文](README.zh-CN.md)

## Project Summary

Legal RAG Agent is a citation-aware question answering system for a fixed eCFR Title 12 financial regulation snapshot.

The project demonstrates a trustworthy RAG/Agent workflow for legal research scenarios: retrieval is tied to regulation sections, citations are verified before answer generation, the LLM only composes answers from verified evidence, and every run can be inspected through a structured trace.

It is designed as a portfolio-grade applied LLM project rather than a general legal chatbot.

## Highlights

- Fixed eCFR Title 12 snapshot: `2025-09-01`
- BGE Large + FAISS dense retrieval
- Citation-aware retrieval for explicit CFR references
- Experimental lexical and hybrid retrieval
- Read-only legal tools:
  - `search_regulations`
  - `fetch_section`
  - `verify_citation`
- Verified Agent loop with bounded steps
- Optional OpenAI-compatible LLM answer generation
- System-controlled citations from verified sections
- LLM-as-Judge answer-quality evaluation
- FastAPI service with trace lookup
- Full local test suite: `154 passed`

## Architecture

```text
eCFR Title 12 fixed snapshot
        |
        v
canonical sections + chunks
        |
        v
BGE embeddings + FAISS index
        |
        v
citation-aware retrieval
        |
        v
LegalRAGAgent
        |
        +--> search_regulations
        +--> fetch_section
        +--> verify_citation
        +--> LLM answer generation / deterministic fallback
        |
        v
answer + citations + trace
        |
        +--> retrieval evaluation
        +--> process evaluation
        +--> LLM-as-Judge evaluation
        +--> FastAPI service
```

More details:

- [Architecture](docs/architecture.md)
- [中文架构说明](docs/architecture_zh.md)

## Agent Workflow

The Agent follows a deterministic, inspectable workflow:

```text
question
-> search_regulations
-> fetch_section
-> verify_citation
-> final_answer
```

The LLM is only used after citation verification. It receives verified evidence and a system-generated citation list. It does not decide which legal authority to cite.

If the LLM call fails, the Agent falls back to deterministic answer generation.

## Evaluation

The project separates evaluation into three layers:

| Layer | Purpose |
|---|---|
| Retrieval evaluation | Check whether the right sections are retrieved |
| Process evaluation | Check whether the Agent follows the expected tool workflow |
| LLM-as-Judge evaluation | Check answer relevance, faithfulness, citation support, and legal caution |

Current validated local test result:

```text
154 passed
```

Generated evaluation reports include:

```text
reports/title12_hybrid_retrieval_eval.md
reports/title12_agent_process_eval.md
reports/title12_llm_judge_eval.md
```

LLM-as-Judge is treated as an auxiliary signal, not ground truth. For credible answer-quality evaluation, the recommended setup is:

```text
answer model = qwen2.5:7b-instruct
judge model  = deepseek-v4-pro
```

## Quickstart

Install dependencies:

```powershell
cd D:\pythonProject\rag_law_clean
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe -m pip install -r requirements.environment.txt
```

Set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
```

Local generated assets are not committed to the repository. The verified local setup uses:

```text
D:\pythonProject\rag_law\bge-large-en-v1.5
```

as the embedding model path.

## CLI Demo

Run without LLM generation:

```powershell
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\ask_agent.py `
  "What does 12 CFR 211.31 apply to?" `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5
```

Run with local Ollama LLM generation:

```powershell
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\ask_agent.py `
  "What does 12 CFR 211.31 apply to?" `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5 `
  --use-llm `
  --llm-base-url http://localhost:11434/v1 `
  --llm-model qwen2.5:7b-instruct `
  --llm-api-key ollama
```

The CLI prints the answer, citations, fetched sections, citation verification status, top evidence, termination reason, and trace path.

## API Demo

Start the FastAPI service:

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
$env:RAG_LAW_EMBEDDING_MODEL = "D:\pythonProject\rag_law\bge-large-en-v1.5"
$env:RAG_LAW_USE_LLM = "true"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  -m uvicorn rag_law.api:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Ask:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question":"What does 12 CFR 211.31 apply to?"}'
```

Trace lookup:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/trace/<trace_id>
```

## Security

Do not commit real API keys.

Use `.env` locally and `.env.example` as the public template:

```env
JUDGE_API_KEY=
JUDGE_BASE_URL=https://api.deepseek.com
JUDGE_MODEL=deepseek-v4-pro

ANSWER_BASE_URL=http://localhost:11434/v1
ANSWER_MODEL=qwen2.5:7b-instruct
ANSWER_API_KEY=ollama
```

`.env` is ignored by Git.

## Limitations

- The repository does not include the local eCFR corpus, FAISS index, or embedding model.
- The system is tied to the fixed `2025-09-01` Title 12 snapshot.
- The evaluation set is small and should not be presented as broad legal QA generalization.
- LLM-as-Judge is an auxiliary evaluation method, not ground truth.
- The Agent workflow is deterministic and does not yet use an LLM planner for tool selection.
- This project is not legal advice.
