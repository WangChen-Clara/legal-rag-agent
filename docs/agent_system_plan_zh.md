# Legal RAG Agent System Plan

## 项目目标

把当前项目从 Legal RAG demo 升级为可展示、可追踪、可评测的 Legal RAG Agent 系统。

最终定位：

> 面向 eCFR Title 12 金融监管法规问答的可信 Legal RAG Agent 系统，支持固定法规快照、citation-aware retrieval、工具化证据读取、引用校验、LLM 答案生成、结构化 Agent Trace、检索评测、流程评测和接口化服务。

项目不追求“大而全法律助手”，主线是补齐法律场景里最关键的可信闭环：

```text
固定法规快照
-> 结构化法规 section
-> 检索
-> 工具读取
-> 引用验证
-> LLM 基于 verified evidence 生成答案
-> Trace
-> Evaluation
-> API / Demo
-> README / 简历展示
```

## 当前 Roadmap

### Phase 1: Trusted Tool Layer

目标：让 Agent 不只具备 `search` / `fetch`，还具备法律场景中关键的 citation verification。

交付内容：

- `verify_citation` 工具
- `CitationVerificationResult` 结构化结果
- 检查 section 是否存在
- 检查 `version_date` 是否等于固定快照日期
- 检查 `source_url` 是否与 section 和 version 对应
- 检查 `safe_for_citation`
- 工具层测试

状态：已完成。

### Phase 2: Verified Agent Loop

目标：把 deterministic Agent 从“检索后回答”升级为“检索、读取、验证、回答”。

流程：

```text
search_regulations
-> fetch_section
-> verify_citation
-> final_answer
```

交付内容：

- Agent 调用 verification step
- `AgentState` 增加 citation verification results
- `to_dict()` / `to_trace_dict()` 输出 verification 信息
- final answer citation 只来自 verified evidence
- citation verification 失败时不输出伪 citation

状态：已完成。

### Phase 3: CLI Demo

目标：让项目可以通过命令行完整运行，而不是一组内部脚本。

交付内容：

- `scripts/ask_agent.py`
- 支持命令行输入问题
- 输出 answer、citations、top evidence、fetched sections、verification status 和 trace JSON path

状态：已完成。

### Phase 4: Retrieval + Process Evaluation

目标：不只评估 retrieval 是否找对，还评估 Agent 是否按可信流程执行。

已包含评测：

- retrieval evaluation
- hybrid retrieval experimental evaluation
- agent process evaluation

Agent process 指标：

```text
tool_success_rate
expected_section_found_rate
fetch_section_success_rate
citation_verified_rate
final_answer_citation_support_rate
average_steps
termination_reason_distribution
```

状态：已完成。

### Phase 5: LLM Answer Generation

目标：让 Agent 的最终回答不再只是模板拼接，而是由 LLM 基于 verified evidence 生成。

设计原则：

- LLM 只做 answer composer
- 检索、证据读取、引用验证仍由系统工具链控制
- citation 列表由系统从 verified sections 生成，不让 LLM 自行决定
- LLM 不可用时回退到 deterministic answer
- 测试使用 fake LLM，不依赖本地模型或外部 API

默认本地模型配置：

```text
Protocol: OpenAI-compatible chat completions
Base URL: http://localhost:11434/v1
Model: qwen2.5:7b-instruct
API key: ollama
```

状态：已完成。

### Phase 6: LLM-as-Judge Evaluation

目标：补齐答案质量评测，评估 LLM 生成答案是否相关、忠实于证据、引用是否支撑结论，以及法律场景下是否足够谨慎。

建议指标：

```text
answer_relevance
faithfulness
citation_support
legal_caution
overall
pass/fail
```

Judge 输入应只包含系统输出和 verified evidence：

```json
{
  "question": "...",
  "answer": "...",
  "citations": ["12 CFR ..."],
  "verified_evidence": [
    {
      "section": "...",
      "text": "...",
      "source_url": "...",
      "version_date": "2025-09-01"
    }
  ],
  "agent_trace": {}
}
```

Judge 输出应为结构化 JSON：

```json
{
  "answer_relevance": 4,
  "faithfulness": 5,
  "citation_support": 5,
  "legal_caution": 4,
  "overall": 4,
  "pass": true,
  "issues": []
}
```

模型选择原则：

- 本地开发 / 演示默认可以使用 `qwen2.5:7b-instruct`，因为它已在本机可用，且可复现。
- 但如果 answer generation 和 judge 都使用同一个模型，这只能证明“评测链路可运行”，不应作为严肃质量结论。
- 更严肃的评测应使用独立 judge 模型，最好比 answer model 更强，例如 Qwen-Max、DeepSeek、GPT 系列或其他 OpenAI-compatible API 模型。
- 测试环境必须使用 fake judge，保证 pytest 不依赖网络、密钥或本地 Ollama。

推荐落地方式：

```text
开发默认:
answer model = qwen2.5:7b-instruct
judge model  = qwen2.5:7b-instruct
用途 = 跑通链路和本地展示

正式评测:
answer model = qwen2.5:7b-instruct
judge model  = 独立更强模型
用途 = 更可信的答案质量评估
```

已落地文件：

- `src/rag_law/llm_judge.py`
- `scripts/evaluate_title12_llm_judge.py`
- `reports/title12_llm_judge_eval.json`
- `reports/title12_llm_judge_eval.md`

状态：已完成。

### Phase 7: FastAPI Service

目标：把 CLI Agent 包装成轻量服务，方便上层应用或演示调用。

建议接口：

```text
GET  /health
POST /ask
GET  /trace/{trace_id}
```

`POST /ask` 返回：

```json
{
  "answer": "...",
  "citations": ["12 CFR ..."],
  "trace_id": "...",
  "termination_reason": "completed",
  "citation_verifications": []
}
```

设计原则：

- 先做接口，不急着做 Docker / 云部署 / 前端
- trace 继续写入本地 `reports/agent_runs`
- API 不影响 CLI
- 服务层只做包装，不改变 Agent 核心逻辑

已落地文件：

- `src/rag_law/api.py`
- `tests/test_api.py`

状态：已完成。

### Phase 8: README / Docs / Resume Packaging

目标：把项目从实验脚本风格整理成完整应用系统展示。

README 建议结构：

```text
Problem
System Architecture
Data Pipeline
Agent Workflow
Tools
LLM Answer Generation
Evaluation
API Demo
Limitations
Roadmap
```

简历和面试材料：

- 2 行简历版
- 5 行简历版
- 2 分钟项目介绍
- 5 分钟技术讲解
- 高频追问 Q&A

状态：待实现。

## 暂不优先做

- 复杂 multi-agent 角色拆分
- 远程 LLM 自动工具规划
- Docker / 云部署
- 手工扩展大规模评测集
- 把 hybrid retrieval 接入默认主流程
- 前端 UI
- 复杂法规版本比较

## 当前展示口径

这个项目的核心不是“什么都能答的法律助手”，而是把法律 RAG 中最关键的可信链路做完整：

```text
固定版本语料
-> 可追踪检索
-> 工具化证据读取
-> 引用校验
-> LLM 基于 verified evidence 生成答案
-> 流程和质量评测
```

因此，项目展示时应强调：

- 法规版本固定，避免法规漂移
- citation 由系统验证，不让 LLM 编造
- LLM 只做答案组织
- trace 可复查每一步
- 评测分为 retrieval、process、answer quality 三层
