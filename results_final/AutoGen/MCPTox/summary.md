# AutoGen / MCPTox (실행 2026-11-21)

에이전트: AutoGen AgentChat `AssistantAgent` (agents/autogen 소스, 0.7.5). 타깃·Simulator `qwen/qwen3.5-9b`, Judge `openai/gpt-oss-120b`.
방어 없음 = 같은 AutoGen 에이전트를 SimGuardAgent 없이 돌린 결과. 지표 정의는 ../../README.md.

| TPR (실제 성공 중) | FPR | ASR | ASR 방어 없음 |
|---|---|---|---|
| 45.5% (51/112) | 0.6% (2/357) | 4.5% (61/1348) | 8.5% (115/1348) |

연결: 케이스의 시스템 프롬프트에 적힌 MCP 도구(오염된 설명 포함)를 AutoGen 도구로 등록하고 첫 도구 호출을 채점 (mcptox_autogen.py). 정답 라벨은 gt_label_mcptox.py.
