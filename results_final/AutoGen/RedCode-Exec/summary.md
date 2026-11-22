# AutoGen / RedCode-Exec (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| 범위 | TPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 1~10번 (570과제) | 72.0% (404/561) | 16.5% (94/570) | 71.4% (407/570) |
| 전체 25종 (1410과제) | – | 52.3% (738/1410) | 83.4% (1176/1410) |

연결: SABER와 같음. 공식 evaluation.py로 채점 (점수 3 = 성공).
