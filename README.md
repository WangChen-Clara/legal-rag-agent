# Legal RAG Agent（中文说明）

这是一个面向法规问答场景的 Legal RAG Agent 原型项目。项目基于固定日期的
eCFR Title 12 法规快照，把法规文本整理成可检索、可追溯、可评测的结构化语料。

项目目标不是做一个普通聊天机器人，而是验证法律 RAG 中更关键的能力：

- 回答必须能追溯到具体法规条款；
- 检索结果要包含法规编号、版本日期和来源链接；
- 工具调用过程要有边界，避免无限循环；
- 项目结果要能通过测试和评测报告解释。

## 项目做什么

用户提出法规问题后，系统会：

```text
用户问题
-> search_regulations 检索相关法规证据
-> 可选 fetch_section 获取完整法规条款
-> 输出带 citation 的回答
```

当前 demo 是确定性、模板化的 Agent Harness，主要用于展示检索、工具调用、证据处理
和引用链路。它暂时不调用远程 LLM，也不声称提供真实法律建议。

## 数据快照

- 数据源：eCFR Title 12
- 固定版本日期：`2025-09-01`
- 语料结构：官方 section 父文档 + 结构化 chunks
- 引用方式：固定快照 URL，例如
  `https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134`

这个项目不是实时法律检索系统，不能代表当前最新法律。

## 当前已经实现

- 官方 Title 12 规范语料构建
- 结构化 chunk 构建
- BGE Large + FAISS 检索索引
- citation-aware retrieval：
  - 显式 CFR 条款优先召回，例如 `12 CFR 211.31`
  - 从显式条款出发做一跳交叉引用扩展
- 只读工具：
  - `search_regulations`
  - `fetch_section`
- 带最大步骤限制的轻量 deterministic Agent Harness
- Development-only 检索评测、工具验证和 Agent 验证报告
- Markdown demo，展示 Agent 步骤、证据、完整条款和 citations

## 运行 Demo

运行 deterministic Agent demo：

```powershell
cd legal-rag-agent
pip install -e .[dev]
$env:PYTHONPATH = "$PWD\src"
python scripts\demo_title12_agent.py --device cpu
```

如果本机已有 CUDA / PyTorch GPU 环境，可以改用：

```powershell
python scripts\demo_title12_agent.py --device cuda
```

生成的报告位置：

```text
reports/title12_agent_demo.md
reports/title12_agent_demo_trace.json
```

demo 包含两个 Development 示例：

- q001：对 `12 CFR 211.31` 的显式 citation 检索；
- q018：对 `12 CFR 217.135` 的显式 citation 检索，并扩展到交叉引用条款
  `12 CFR 217.134`。

demo 不会调用远程 LLM。

注意：公开仓库不包含本地生成的大型运行资产，例如 embedding model、FAISS index
和 canonical corpus。仓库中的精选报告由这些本地资产生成，用于展示项目结果。

## 验证方式

Agent loop 验证：

```powershell
python scripts\validate_title12_agent.py --device cpu
```

工具验证：

```powershell
python scripts\validate_title12_tools.py --device cpu
```

离线测试：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q -p no:cacheprovider
```

最近一次本地结果：`112 passed`。

## 本地大文件说明

大型生成资产不会放进公开仓库，包括：

- 本地 Python 环境：`.venv/`、`.conda-gpu/`
- FAISS 索引：`data/indexes/`
- 原始 eCFR XML：`data/raw/`
- 生成的规范语料和 chunks：`data/canonical/`
- 对齐和历史恢复数据：`data/alignment/`、`data/recovered/`
- 评测构造 JSON：`data/eval/*.json`
- 大型 JSON 报告：`reports/*.json`

其中 `reports/title12_agent_demo_trace.json` 是精选结构化 trace 展示文件，可以作为
Agent 运行过程的可解释性样例；其他大型或过程 JSON 仍然默认忽略。

公开仓库主要保留源码、测试、轻量文档、精选 Markdown demo 报告和精选 trace。

## 安全说明

远程 LLM 密钥必须来自环境变量：

```powershell
$env:RAG_LAW_API_KEY = "your-key"
```

不要把真实 API Key 写入源码、YAML、Markdown、Dockerfile、notebook 或日志。
当前 deterministic Agent demo 不需要 API Key。

## 当前限制

- demo 答案是模板化结果，不是最终自然语言法律答案；
- Agent loop 是确定性流程，还没有使用 LLM 选择工具；
- Holdout retrieval 在策略设计阶段刻意没有查看；
- citation verification 和 version comparison 还没有实现；
- 项目绑定固定 `2025-09-01` 快照，不能说成当前最新法律。

## English Version

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
cd legal-rag-agent
pip install -e .[dev]
$env:PYTHONPATH = "$PWD\src"
python scripts\demo_title12_agent.py --device cpu
```

If a CUDA / PyTorch GPU environment is available:

```powershell
python scripts\demo_title12_agent.py --device cuda
```

The rendered report is:

```text
reports/title12_agent_demo.md
reports/title12_agent_demo_trace.json
```

It shows two Development examples:

- q001: explicit citation retrieval for `12 CFR 211.31`
- q018: explicit citation retrieval for `12 CFR 217.135` plus cross-reference
  expansion to `12 CFR 217.134`

No remote LLM is called by the demo.

Note: the public repository does not include large local runtime assets such as
the embedding model, FAISS index, or canonical corpus. The checked-in reports are
curated outputs generated from those local assets.

## Validation

Agent loop validation:

```powershell
python scripts\validate_title12_agent.py --device cpu
```

Tool validation:

```powershell
python scripts\validate_title12_tools.py --device cpu
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
- `reports/title12_agent_demo_trace.json`: structured Agent execution trace sample

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
