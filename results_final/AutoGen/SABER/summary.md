# AutoGen / SABER (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| TPR (해로운 단계) | FPR (A/B 정상 단계) | ASR (과제 단위) | ASR 방어 없음 |
|---|---|---|---|
| 31.4% (75/239) | 28.0% (98/350) | 8.4% (60/716) | 34.1% (244/716) |

연결: AutoGen의 bash·python 도구가 AgentDojo 에이전트와 같은 Docker 세션 판정(intercept_round: 실행 → 다음 단계 예측 → ExecutionJudge)을 거침 (code_autogen.py). 단계 라벨은 exec_score.py.
