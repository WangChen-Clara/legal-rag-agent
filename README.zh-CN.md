# Legal RAG Agent

[English](README.md) | [简体中文](README.zh-CN.md)

面向 eCFR Title 12 金融监管法规问答的可追踪 Legal RAG Agent 系统。

这个项目不是普通法律聊天机器人，而是围绕固定法规快照和 verified evidence 构建的可信 RAG/Agent 系统。核心链路是：

```text
固定 eCFR 快照
-> citation-aware retrieval
-> fetch section
-> verify citation
-> 基于 verified evidence 的 LLM 答案生成
-> trace
-> retrieval / process / answer-quality evaluation
-> CLI 和 FastAPI 服务
```

项目重点不是让模型自由发挥，而是把法律 RAG 中最容易出错的引用、版本、证据支撑和可追踪问题显式工程化。

## 当前状态

已实现：

- 固定 eCFR Title 12 法规快照工作流
- BGE Large + FAISS 稠密检索
- citation-aware retrieval，显式 CFR 条款优先召回
- 实验性 lexical + hybrid retrieval
- 只读工具层：
  - `search_regulations`
  - `fetch_section`
  - `verify_citation`
- verified Agent loop：
  - `search_regulations`
  - `fetch_section`
  - `verify_citation`
  - `final_answer`
- 可选 OpenAI-compatible LLM 答案生成
- LLM-as-Judge 答案质量评测
- FastAPI 服务：
  - `GET /health`
  - `POST /ask`
  - `GET /trace/{trace_id}`
- 结构化 Agent trace 输出
- retrieval、process、LLM judge、tool、CLI、API 测试

最近一次本地测试结果：

```text
154 passed
```

## 为什么要做 citation verification

法律 RAG 和普通文档问答最大的区别是：答案听起来合理不够，必须知道依据来自哪里。

系统需要确认：

- 检索到的是哪个法规 section
- 来自哪个固定版本日期
- source URL 是否与固定快照匹配
- citation 是否安全可展示
- 最终答案是否只引用 verified evidence

因此，本项目把 LLM 放在检索和引用验证之后。LLM 只负责组织答案，不负责决定法律依据。

## 数据快照

- 数据源：eCFR Title 12
- 固定版本日期：`2025-09-01`
- citation URL 示例：

```text
https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134
```

本项目不是实时法律检索系统，也不构成法律建议。

## 系统架构

更多架构说明见：

- `docs/architecture.md`
- `docs/architecture_zh.md`
- `docs/agent_system_plan_zh.md`

高层流程：

```text
用户问题
  |
  v
search_regulations
  |
  v
fetch_section
  |
  v
verify_citation
  |
  v
LLM answer generation 或 deterministic fallback
  |
  v
最终答案 + citations + trace
```

最终 citation 列表由系统从 verified sections 中生成，不让模型自行编造引用。

## 本地资产说明

大型生成资产不会提交到仓库：

- `.venv/`
- `.conda-gpu/`
- `models/`
- `data/raw/`
- `data/canonical/`
- `data/indexes/`
- `data/alignment/`
- `data/recovered/`
- `*.index`
- `*.npy`
- 大型 JSON 报告

当前本地验证使用的 embedding model 路径：

```text
D:\pythonProject\rag_law\bge-large-en-v1.5
```

## 环境安装

安装依赖：

```powershell
cd D:\pythonProject\rag_law_clean
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe -m pip install -r requirements.environment.txt
```

设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
```

## 密钥配置

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

在 PowerShell 中加载 `.env`：

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -and -not $_.StartsWith("#")) {
    $name, $value = $_ -split "=", 2
    Set-Item -Path "Env:$name" -Value $value
  }
}
```

`.env` 已被 Git 忽略。

## CLI Demo

不启用 LLM 的 Agent 运行方式：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\ask_agent.py `
  "What does 12 CFR 211.31 apply to?" `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5
```

启用本地 Ollama LLM 答案生成：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"

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

CLI 会输出：

- answer
- citations
- fetched sections
- citation verification status
- top evidence
- termination reason
- trace JSON path

## LLM 答案生成

默认本地 answer model：

```text
qwen2.5:7b-instruct via Ollama
```

代码使用 OpenAI-compatible chat-completions 接口。可以通过以下参数替换为其他兼容 API：

```text
--llm-base-url
--llm-model
--llm-api-key
```

如果 LLM 调用失败，Agent 会回退到 deterministic answer。系统控制的 citation 列表仍然保留。

## LLM-as-Judge 评测

运行 LLM-as-Judge：

```powershell
cd D:\pythonProject\rag_law_clean

Get-Content .env | ForEach-Object {
  if ($_ -and -not $_.StartsWith("#")) {
    $name, $value = $_ -split "=", 2
    Set-Item -Path "Env:$name" -Value $value
  }
}

$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\evaluate_title12_llm_judge.py `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5 `
  --judge-base-url $env:JUDGE_BASE_URL `
  --judge-model $env:JUDGE_MODEL `
  --judge-api-key-env JUDGE_API_KEY
```

输出：

```text
reports/title12_llm_judge_eval.json
reports/title12_llm_judge_eval.md
```

Judge 指标：

```text
answer_relevance
faithfulness
citation_support
legal_caution
overall
pass/fail
issues
```

重要说明：

如果 answer model 和 judge model 是同一个模型，评测结果只能说明 LLM-as-Judge 链路可运行，不能当作严肃质量结论。更可信的配置是：

```text
answer model = qwen2.5:7b-instruct
judge model  = deepseek-v4-pro
```

## FastAPI 服务

启动 API：

```powershell
cd D:\pythonProject\rag_law_clean

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

查看 trace：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/trace/<trace_id>
```

API 使用懒加载，导入 `rag_law.api:app` 时不会立刻加载 index 和 embedding model。

## 评测脚本

Hybrid retrieval 评测：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\evaluate_title12_hybrid_retrieval.py `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5
```

Agent process 评测：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"

D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe `
  D:\pythonProject\rag_law_clean\scripts\evaluate_title12_agent_process.py `
  --device cpu `
  --model D:\pythonProject\rag_law\bge-large-en-v1.5
```

## 测试

运行全量测试：

```powershell
$env:PYTHONPATH = "D:\pythonProject\rag_law_clean\src"
D:\pythonProject\rag_law_clean\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

最近一次本地结果：

```text
154 passed
```

## 设计取舍

- 固定法规快照，而不是实时法律搜索
- verified citations，而不是模型自由生成引用
- 工具化 section 读取，而不是无边界塞上下文
- LLM 只在引用验证后组织答案
- LLM 失败时可回退 deterministic answer
- 评测拆成 retrieval、process、answer quality 三层
- FastAPI 只是薄服务层，复用同一个 Agent core

## 当前限制

- 公开仓库不包含本地 eCFR corpus、FAISS index 和 embedding model。
- 当前评测集规模小，不能当成广泛法律 QA 泛化能力证明。
- LLM-as-Judge 只是辅助信号，不是 ground truth。
- 系统绑定 `2025-09-01` 的 Title 12 固定快照。
- Agent workflow 仍是 deterministic，没有使用 LLM planner 自动选择工具。
- 本项目不构成法律建议。

## Roadmap

近期：

- 在低峰时段用 `deepseek-v4-pro` 跑正式 LLM-as-Judge
- 如果 citation support 或 faithfulness 偏低，优化 answer prompt
- 增加更清晰的样例输出和展示材料
- 准备简历与面试讲解文档

暂缓：

- 复杂 multi-agent 角色拆分
- Docker / 云部署
- 前端 UI
- 大规模人工评测集
- 复杂法规版本比较
