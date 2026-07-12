# 架构图

```mermaid
flowchart TD
    U["用户问题"] --> A["LegalRAGAgent<br/>确定性 Agent Harness"]

    A --> T1["search_regulations<br/>引用感知检索"]
    A --> T2["fetch_section<br/>完整条文读取"]
    A --> TR["Agent Trace<br/>步骤 / 证据 / 引用"]

    T1 --> R["FaissRetriever"]
    R --> C1["显式法规引用解析<br/>12 CFR 217.135"]
    R --> C2["交叉引用扩展<br/>一跳引用条文"]
    R --> C3["语义检索<br/>BGE Large embeddings + FAISS"]

    C1 --> IDX["FAISS 索引<br/>vector_db.index"]
    C2 --> IDX
    C3 --> IDX

    IDX --> META["元数据<br/>section / part / version / source_url"]
    META --> EV["RegulationEvidence<br/>结构化检索证据"]

    T2 --> SEC["规范条文库<br/>sections.jsonl"]
    SEC --> FULL["SectionRecord<br/>全文 + 引用元数据"]

    EV --> A
    FULL --> A

    A --> FA["FinalAnswer<br/>模板化回答 + 固定快照引用"]
    A --> TR

    RAW["eCFR Title 12 XML<br/>固定快照 2025-09-01"] --> P1["ecfr_parser<br/>法规 XML 解析"]
    P1 --> CAN["Canonical Corpus<br/>官方 section 语料"]
    CAN --> CH["Structured Chunks<br/>结构化切块"]
    CH --> EMB["Embedding 构建<br/>BGE Large"]
    EMB --> IDX
    CAN --> SEC

    EVAL["Development QA Set<br/>开发评测集"] --> ER["检索评测<br/>Hit Rate / Recall@K / MRR"]
    IDX --> ER
    ER --> REP["验证报告<br/>retrieval / agent / tools"]
```

这个项目是一个面向 eCFR Title 12 固定快照的 Legal RAG Agent 原型。用户问题先进入
确定性的 `LegalRAGAgent`，Agent 调用只读工具完成引用感知检索和完整条文读取。检索层
由三部分组成：显式 `12 CFR xxx` 引用优先召回、一跳交叉引用扩展，以及 BGE Large +
FAISS 语义检索。Agent 最终输出带固定快照 citation 的模板化回答，并记录结构化 trace，
包括工具调用步骤、检索证据、读取的完整条文和最终引用。

数据管线从 `2025-09-01` 的 eCFR Title 12 XML 固定快照开始。解析器先生成官方 section
级规范语料，随后结构化切块用于向量检索，embedding/indexing 步骤生成 FAISS 运行索引。
Development QA 数据和验证脚本用于计算 Hit Rate、Recall@K、MRR 等检索指标，并验证工具层
和 Agent 流程是否符合预期。
