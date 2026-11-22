# AutoGen / AgentDojo (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| TPR | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 30.5% (289/949) | 5.2% (5/97) | 1.3% (12/949) | 8.7% (83/949) |

연결: AutoGen을 AgentDojo 파이프라인 요소로 넣음 (agentdojo_autogen.py). AgentDojo 함수가 AutoGen 도구가 되고, AgentDojo의 benchmark_suite_with_injections가 같은 과제 쌍(v1.2.2, tool_knowledge)을 돌려 AgentDojo 검사로 채점. SimGuard는 simguard.protect로 AutoGen의 모델 호출을 판정.
