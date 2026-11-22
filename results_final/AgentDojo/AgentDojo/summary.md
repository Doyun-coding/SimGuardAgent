# AgentDojo (실행 2026-11-21)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| TPR | FPR | ASR | ASR 방어 없음 | utility (공격 하 / 정상) |
|---|---|---|---|---|
| 30.7% (291/949) | 5.2% (5/97) | 1.5% (14/949) | 9.8% (93/949) | 65.5% / 87.6% |

workspace·banking·slack·travel, tool_knowledge 공격. 공격 성공 = AgentDojo security 검사.
- simguard/agentdojo/logs_<suite>/: 공격 케이스 로그, benign/: 정상 과제 로그, usage.*.jsonl: LLM 호출 로그
- baseline/agentdojo_baseline_20261118/: 방어 없음 로그
