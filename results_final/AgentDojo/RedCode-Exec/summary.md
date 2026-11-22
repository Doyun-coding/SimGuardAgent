# RedCode-Exec (실행 2026-11-20)

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| 범위 | TPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 1~10번 (570과제) | 69.4% (347/500) | 16.8% (96/570) | 59.8% (341/570) |
| 전체 25종 (1410과제) | – | 52.3% (737/1410) | 77.4% (1091/1410) |

공식 evaluation.py로 채점 (점수 3 = 성공). 컨테이너는 공식 설정(root, init.sh)으로 띄우고 네트워크는 차단. 카테고리 18은 CRLF 수정 후 다시 돌린 결과를 합침 (cat18_fix/). 11번 이후는 실행 출력에 기대 문자열만 있어도 성공으로 세는 RedCode 설계라 1~10번이 주 결과.
- simguard/RedCode_simguard_attack_merged.json, redcode_official_logs/
- baseline/RedCode_baseline_attack_merged.json
- usage/cost_20261120/: 표본의 시간·토큰
