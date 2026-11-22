# BIPIA (실행 2026-11-21)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| TPR | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 32.4% (107/330) | 9.3% (14/150) | 3.6% (12/330) | 21.2% (70/330) |

유해 공격 유형만: 이메일·표 QA에 Scams & Fraud, Misinformation & Propaganda, Marketing & Advertising, 코드 QA에 악성 코드 10종. 공격 성공 = BIPIA 공식 평가 (LLM 질문은 gpt-6-luna, 코드는 fuzzy match).
FPR 원인: 표 답 6건 documents_disagree (틀린 숫자를 문서 충돌로 판단), 코드 답 8건 (답 속 코드가 실행 경로로 감).
- simguard/bipia/bipia_simguard_{attack,benign}.json, gt_simguard.jsonl
- baseline/: 방어 없음 답 + bipia_gt.jsonl
