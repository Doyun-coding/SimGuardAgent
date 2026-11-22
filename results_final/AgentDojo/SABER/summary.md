# SABER (실행 2026-11-20)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| TPR (해로운 단계) | FPR (A/B 정상 단계) | ASR (과제 단위) | ASR 방어 없음 |
|---|---|---|---|
| 33.2% (65/196) | 28.1% (114/406) | 6.6% | 30.0% |

ExecutionJudge v4, Docker 실행 + 에이전트 작업 파일, 최대 10단계. 단계 라벨은 SABER 정답 기준(exec_score.py), 과제 ASR은 saber_task_asr.py.
- simguard/: SABER_simguard_{attack,benign}.json
- baseline/: SABER_baseline_*.json
- usage/cost_20261120/: 30과제 표본의 시간·토큰
