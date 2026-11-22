# AutoGen / AgentHarm (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| 과제 | 샘플 | BLOCK | 지표 |
|---|---|---|---|
| harmful | 176 | 63 (TPR 35.8%) | ASR 5.6% (유해 점수 0.056), 완전 성공 3.4% |
| benign | 176 | 64 (FPR 36.4%) | 정상 점수 0.556 |
| chat | 44 | 1 | – |

방어 없음: 유해 점수 0.315 (31.5%), 완전 성공 23.3%. 연결: inspect의 AgentHarm 평가(데이터, 도구 설정, 기본 시스템 프롬프트, 공식 채점)는 그대로 두고 생성 루프만 AutoGen 에이전트로 바꿈 (agentharm_autogen.py, .venv-agentharm에서 실행).
