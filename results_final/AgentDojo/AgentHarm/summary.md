# AgentHarm (실행 2026-11-21)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| 과제 | 샘플 | BLOCK | 지표 |
|---|---|---|---|
| harmful | 176 | 61 (TPR 34.7%) | ASR 5.5% (유해 점수 0.055), 완전 성공 2.8%, 거절 51.7% |
| benign | 176 | 57 (FPR 32.4%) | 정상 점수 0.562 |
| chat | 44 | 2 | – |

방어 없음: 유해 점수 0.293 (29.3%), 완전 성공 21.0%. 공식 inspect 에이전트 + simguard.protect, 채점 gpt-4o-mini (AgentHarm 공식).
FPR이 높은 원인: 정상 과제가 유해 과제와 비슷하게 쓰여 ToolCallJudge의 scope_violation·unauthorized_action·harmful_output이 걸림.
- simguard/agentharm/summary_*.json, verdicts_*.jsonl (판정 상세), *.eval (inspect 로그)
