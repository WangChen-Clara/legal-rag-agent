# Title 12 Legal RAG Agent Demo

- Demo type: deterministic Agent Harness
- Snapshot date: `2025-09-01`
- Questions: 2
- Max steps: 4
- Max fetch sections: 2
- Holdout retrieval inspected: no
- LLM called: no

This demo shows the application-level flow: question -> agent steps -> tool calls
-> evidence -> final cited answer. The answer text is template-based; this report
is intended to demonstrate control flow, tool use, and citation plumbing.

## title12-dev-q001

**Question:** What investors do the provisions of 12 CFR 211.31's subpart apply to?

### Agent Steps

| Step | Action | Status | Detail |
|---:|---|---|---|
| 1 | search_regulations | completed | mode=citation_aware<br>evidence_count=10<br>sections=211.31, 211.10, 211.8, 211.10, 211.10 |
| 2 | fetch_section | completed | section=211.31<br>source_url=https://www.ecfr.gov/on/2025-09-01/title-12/section-211.31<br>safe_for_citation=True |
| 3 | final_answer | completed | citations=12 CFR 211.31 (2025-09-01) |

### Unique Sections Summary

211.31 (explicit_citation), 211.10 (semantic), 211.8 (semantic), 211.33 (semantic), 211.9 (semantic), 211.32 (semantic)

### Retrieved Evidence

| Rank | Section | Source | Version | URL | Preview |
|---:|---|---|---|---|---|
| 1 | 211.31 | explicit_citation | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.31 | Authority, purpose, and scope. (a) Authority. This subpart is issued by the Board of Governors of the Federal Reserve System (Board) under the authority of t... |
| 2 | 211.10 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.10 | ced such activities prior to March 27, 1991, and subject to the limitations in effect at that time (See 12 CFR part 211, revised January 1, 1991); or (ii) Li... |
| 3 | 211.8 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.8 | Investment Limit. Portfolio investments made under authority of this subpart shall be subject to the aggregate equity limit of § 211.10(a)(15)(iii). (iii) Lo... |
| 4 | 211.10 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.10 | f this section only if each of the bank holding company, member bank, and Edge or agreement corporation qualify as well-capitalized and well-managed. (ii) Li... |
| 5 | 211.10 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.10 | (A) With respect to foreign persons only; and (B) Subject to the limitations on owning or controlling shares of a company in section 4(c)(6) of the BHC Act (... |

### Fetched Sections

| Section | Heading | Version | URL | Safe for citation |
|---|---|---|---|---|
| 211.31 | § 211.31 Authority, purpose, and scope. | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-211.31 | True |

### Final Answer

Relevant evidence was found in 12 CFR 211.31 (2025-09-01). Use the cited fixed-snapshot sections to answer the question.

**Citations:** 12 CFR 211.31 (2025-09-01)

### Curated Answer Preview

This preview is curated for demonstration only; the current agent answer remains template-based.

The retrieved section indicates that the subpart applies to eligible investors under 12 CFR 211.31. The demo keeps the generated agent answer template-based, but this preview shows how the cited section can support a direct natural-language response.

## title12-dev-q018

**Question:** For double default treatment under 12 CFR 217.135, what kind of exposure may be hedged and what related section defines the eligible guarantee or credit derivative treatment?

### Agent Steps

| Step | Action | Status | Detail |
|---:|---|---|---|
| 1 | search_regulations | completed | mode=citation_aware<br>evidence_count=10<br>sections=217.135, 217.134, 217.142, 217.135, 217.135 |
| 2 | fetch_section | completed | section=217.135<br>source_url=https://www.ecfr.gov/on/2025-09-01/title-12/section-217.135<br>safe_for_citation=True |
| 3 | fetch_section | completed | section=217.134<br>source_url=https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134<br>safe_for_citation=True |
| 4 | final_answer | completed | citations=12 CFR 217.135 (2025-09-01), 12 CFR 217.134 (2025-09-01) |

### Unique Sections Summary

217.135 (explicit_citation), 217.134 (cross_reference), 217.142 (cross_reference), 217.135 (semantic), 324.135 (semantic), 3.135 (semantic)

### Retrieved Evidence

| Rank | Section | Source | Version | URL | Preview |
|---:|---|---|---|---|---|
| 1 | 217.135 | explicit_citation | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.135 | Guarantees and credit derivatives: double default treatment. (a) Eligibility and operational criteria for double default treatment. A Board-regulated institu... |
| 2 | 217.134 | cross_reference | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134 | Guarantees and credit derivatives: PD substitution and LGD adjustment approaches. (a) Scope. (1) This section applies to wholesale exposures for which: (i) C... |
| 3 | 217.142 | cross_reference | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.142 | Risk-based capital requirement for securitization exposures. (a) Hierarchy of approaches. Except as provided elsewhere in this section and in § 217.141: (1) ... |
| 4 | 217.135 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.135 | tion provider. If excessive correlation is present, the Board-regulated institution may not use the double default treatment for the hedged exposure. (b) Ful... |
| 5 | 217.135 | semantic | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.135 | (3) The hedged exposure is a wholesale exposure (other than a sovereign exposure). (4) The obligor of the hedged exposure is not: (i) An eligible double defa... |

### Fetched Sections

| Section | Heading | Version | URL | Safe for citation |
|---|---|---|---|---|
| 217.135 | § 217.135 Guarantees and credit derivatives: double default treatment. | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.135 | True |
| 217.134 | § 217.134 Guarantees and credit derivatives: PD substitution and LGD adjustment approaches. | 2025-09-01 | https://www.ecfr.gov/on/2025-09-01/title-12/section-217.134 | True |

### Final Answer

Relevant evidence was found in 12 CFR 217.135 (2025-09-01), 12 CFR 217.134 (2025-09-01). Use the cited fixed-snapshot sections to answer the question.

**Citations:** 12 CFR 217.135 (2025-09-01), 12 CFR 217.134 (2025-09-01)

### Curated Answer Preview

This preview is curated for demonstration only; the current agent answer remains template-based.

For double default treatment, the cited evidence points to a hedged wholesale exposure under 12 CFR 217.135 and the related guarantee or credit derivative treatment in 12 CFR 217.134. This preview is curated for demonstration; the agent's own answer remains template-based.

