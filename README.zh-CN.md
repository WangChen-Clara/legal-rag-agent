# Legal RAG Agent

[English](README.md) | [简体中文](README.zh-CN.md)

## Project Summary

Legal RAG Agent 是一个面向 eCFR Title 12 金融监管法规固定快照的 citation-aware 问答系统。

项目展示的是法律场景下更可信的 RAG/Agent 工作流：检索绑定到法规 section，答案生成前先验证 citation，LLM 只基于 verified evidence 组织答案，每次运行都可以通过结构化 trace 复查。

这个仓库定位为求职展示型大模型应用项目，不是通用法律聊天机器人。

## Highlights

- 固定 eCFR Title 12 快照：`2025-09-01`
- BGE Large + FAISS 稠密检索
- 面向显式 CFR 引用的 citation-aware retrieval
- 实验性 lexical 和 hybrid retrieval
- 只读法律工具：
  - `search_regulations`
  - `fetch_section`
  - `verify_citation`
- 有步数边界的 verified Agent loop
- 可选 OpenAI-compatible LLM 答案生成
- citation 由系统从 verified sections 中生成
- LLM-as-Judge 答案质量评测
- FastAPI 服务和 trace 查询
- 本地全量测试：`154 passed`

## Architecture

```text
eCFR Title 12 固定快照
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

更多说明：

- [Architecture](docs/architecture.md)
- [中文架构说明](docs/architecture_zh.md)

## Agent Workflow

Agent 使用确定性、可检查的流程：

```text
question
-> search_regulations
-> fetch_section
-> verify_citation
-> final_answer
```

LLM 只在 citation verification 之后使用。它接收 verified evidence 和系统生成的 citation 列表，不负责决定引用哪个法律依据。

如果 LLM 调用失败，Agent 会回退到 deterministic answer。

## Evaluation

项目把评测拆成三层：

| 层级 | 目的 |
|---|---|
| Retrieval evaluation | 检查是否找到了正确法规 section |
| Process evaluation | 检查 Agent 是否按预期工具流程执行 |
| LLM-as-Judge evaluation | 检查答案相关性、忠实度、引用支撑和法律谨慎性 |

当前本地验证结果：

```text
154 passed
```

生成的评测报告包括：

```text
reports/title12_hybrid_retrieval_eval.md
reports/title12_agent_process_eval.md
reports/title12_llm_judge_eval.md
```

LLM-as-Judge 只作为辅助信号，不是 ground truth。更可信的答案质量评测建议使用：

```text
answer model = qwen2.5:7b-instruct
judge model  = deepseek-v4-pro
```

## Quickstart

安装依赖：

```powershell
cd D:\pythonProject\rag_law_clean
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe -m pip install -r requirements.environment.txt
```

设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
```

本地生成资产不会提交到仓库。当前验证环境使用的 embedding model 路径是：

```text
D:\pythonProject\rag_law\bge-large-en-v1.5
```

## CLI Demo

不启用 LLM：

```powershell
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\ask_agent.py `
  "What does 12 CFR 211.31 apply to?" `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5
```

启用本地 Ollama LLM：

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

CLI 会输出 answer、citations、fetched sections、citation verification status、top evidence、termination reason 和 trace path。

## API Demo

启动 FastAPI 服务：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
$env:RAG_LAW_EMBEDDING_MODEL = "D:\pythonProject\rag_law\bge-large-en-v1.5"
$env:RAG_LAW_USE_LLM = "true"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  -m uvicorn rag_law.api:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

提问：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question":"What does 12 CFR 211.31 apply to?"}'
```

查询 trace：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/trace/<trace_id>
```

## Security

不要提交真实 API key。

本地使用 `.env`，公开模板使用 `.env.example`：

```env
JUDGE_API_KEY=
JUDGE_BASE_URL=https://api.deepseek.com
JUDGE_MODEL=deepseek-v4-pro

ANSWER_BASE_URL=http://localhost:11434/v1
ANSWER_MODEL=qwen2.5:7b-instruct
ANSWER_API_KEY=ollama
```

`.env` 已被 Git 忽略。

## Limitations

- 仓库不包含本地 eCFR corpus、FAISS index 或 embedding model。
- 系统绑定 `2025-09-01` 的 Title 12 固定快照。
- 当前评测集规模较小，不能当作广泛法律 QA 泛化能力证明。
- LLM-as-Judge 是辅助评测方法，不是 ground truth。
- Agent workflow 仍是 deterministic，没有使用 LLM planner 自动选择工具。
- 本项目不构成法律建议。
