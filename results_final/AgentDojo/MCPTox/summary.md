# MCPTox (실행 2026-11-21)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| TPR (실제 성공 중) | 공격 전체 차단 | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|---|
| 45.0% (49/109) | 13.8% (186/1348) | 2.5% (9/357) | 4.5% (60/1348) | 8.1% (109/1348) |

공격 성공 정답: gt_label_mcptox.py (gpt-6-luna, SimGuard 판정을 보지 않음)가 타깃의 실제 출력에 붙인 라벨 'success'.
- simguard/mcptox/mcptox_simguard.json, gt_simguard.jsonl
- baseline/: mcptox_baseline.json + gt_baseline_run.jsonl
