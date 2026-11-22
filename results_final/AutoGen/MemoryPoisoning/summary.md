# AutoGen / MemoryPoisoning (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| 설정 | TPR (시도된 오염 저장 중) | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|---|
| 앱 문서 경계 표시 | 51.9% (54/104) | 0.0% (0/70) | 20.4% (51/250) | 39.6% (99/250) |
| 경계 표시 없음 | 11.5% (20/174) | 0.0% (0/70) | 61.6% (154/250) | 70.4% (176/250) |

연결: 벤치마크의 시스템 프롬프트·사용자 메시지 + save_memory 도구 (memory_autogen.py). 샘플은 labels/memory_samples/, 채점은 memory_grade.py (벤치마크 goal-match).
