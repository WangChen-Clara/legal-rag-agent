# TODO

## LLM-as-Judge 正式评测

- [ ] 晚上低峰时段运行 `evaluate_title12_llm_judge.py`
- [ ] 使用 `.env` 中的 `deepseek-v4-pro` 作为 judge model
- [ ] 检查输出报告：
  - `reports/title12_llm_judge_eval.json`
  - `reports/title12_llm_judge_eval.md`
- [ ] 重点分析指标：
  - `pass_rate`
  - `average_faithfulness`
  - `average_citation_support`
  - `average_legal_caution`
- [ ] 阅读每条 `issues`，判断问题来自：
  - answer prompt 不够约束
  - verified evidence 格式不够清晰
  - citation 不够细
  - judge 标准过严或不稳定

## 后续优化候选

- [ ] 如果 citation support 低，优化 answer prompt，要求每个关键结论带 citation
- [ ] 如果 faithfulness 低，压缩 verified evidence，减少模型过度概括
- [ ] 如果 legal caution 低，加入法律免责声明和证据不足处理规则
- [ ] 如果 judge 输出不稳定，考虑强化 JSON repair 或更换 judge model
