# Architecture

```mermaid
flowchart TD
    U["User Question"] --> A["LegalRAGAgent<br/>deterministic agent harness"]

    A --> T1["search_regulations<br/>citation-aware search"]
    A --> T2["fetch_section<br/>full section lookup"]
    A --> TR["Agent Trace<br/>steps / evidence / citations"]

    T1 --> R["FaissRetriever"]
    R --> C1["Explicit Citation Parser<br/>12 CFR 217.135"]
    R --> C2["Cross-reference Expansion<br/>one-hop cited sections"]
    R --> C3["Semantic Retrieval<br/>BGE Large embeddings + FAISS"]

    C1 --> IDX["FAISS Index<br/>vector_db.index"]
    C2 --> IDX
    C3 --> IDX

    IDX --> META["Metadata<br/>section / part / version / source_url"]
    META --> EV["RegulationEvidence"]

    T2 --> SEC["Canonical Sections Store<br/>sections.jsonl"]
    SEC --> FULL["SectionRecord<br/>full text + citation metadata"]

    EV --> A
    FULL --> A

    A --> FA["FinalAnswer<br/>template answer + fixed-snapshot citations"]
    A --> TR

    RAW["eCFR Title 12 XML<br/>fixed snapshot 2025-09-01"] --> P1["ecfr_parser"]
    P1 --> CAN["Canonical Corpus<br/>official sections"]
    CAN --> CH["Structured Chunks"]
    CH --> EMB["Embedding Build<br/>BGE Large"]
    EMB --> IDX
    CAN --> SEC

    EVAL["Development QA Set"] --> ER["Retrieval Evaluation<br/>Hit Rate / Recall@K / MRR"]
    IDX --> ER
    ER --> REP["Reports<br/>retrieval / agent / tools validation"]
```

The system is a CLI-first Legal RAG Agent over a fixed eCFR Title 12 snapshot.
User questions enter a deterministic `LegalRAGAgent`, which calls read-only tools
for citation-aware retrieval and full-section lookup. Retrieval combines explicit
CFR citation matching, one-hop cross-reference expansion, and BGE Large + FAISS
semantic search. The agent returns a template-based cited answer and a structured
trace that records tool calls, retrieved evidence, fetched sections, and final
citations.

The data pipeline starts from the fixed `2025-09-01` eCFR Title 12 XML snapshot.
The parser builds canonical section records, the chunk builder produces structured
retrieval chunks, and the embedding/indexing step creates the FAISS runtime index.
Development QA data and validation scripts are used to report retrieval metrics
and validate the tool and agent flows.
