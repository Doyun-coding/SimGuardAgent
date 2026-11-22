# GA_Jailbreak (실행 2026-11-17 (채점 2026-11-18))

타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`, AgentDojo 에이전트. 지표 정의는 ../README.md.

| TPR | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 13.3% (185/1395) | 4.3% (60/1400) | 3.7% (52/1395) | 9.8% (137/1394) |

BehaviorJudge 내용 축 버전 (현재 BehaviorJudge의 검색 문서 없는 프롬프트와 같음). ASR은 HarmBench 방식 (Qwen3.5-9B 채점), 통과한 공격 중 유해 52/1210 = 4.3%.
- simguard/agentdojo/attack.json, benign.json
- labels/table.md (채점표), agentdojo_*.jsonl (HarmBench 채점), compare.md (다른 라벨러 비교)
