# AutoGen / BIPIA (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| TPR | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 29.4% (97/330) | 4.7% (7/150) | 6.7% (22/330) | 23.9% (79/330) |

연결: AgentDojo 에이전트와 같은 샘플(bipia_samples.py). 외부 문서는 시스템 프롬프트에 넣고, simguard.retrieved_context(문서)로 감싸 BehaviorJudge의 RAG 점검을 켬.
