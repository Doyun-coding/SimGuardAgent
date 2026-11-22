# AutoGen / GA_Jailbreak (실행 2026-11-17 (채점 2026-11-18))

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| TPR | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 10.8% (150/1395) | 3.1% (44/1400) | 3.2% (45/1395) | 7.0% (97/1395) |

agents/autogen/run_autogen_simguard.py로 돌린 기존 결과 (현재 BehaviorJudge의 검색 문서 없는 프롬프트와 같음). ASR은 HarmBench 방식 (Qwen3.5-9B 채점).
