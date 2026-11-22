# Full run 2026-11-21 (runs/full_20261121)

Current simguard/ (memory check in ToolCallJudge, RAG checks in BehaviorJudge, retrieved_context wired), AgentDojo
agent (AgentHarm: its inspect agent with simguard.protect). Target = simulator = qwen/qwen3.5-9b, judge =
openai/gpt-oss-120b. All six ran at the same time (`full_run.sh`); `score.py` makes the tables (raw: score.md).
No-defense ASR comes from the earlier baseline runs (they do not use SimGuardAgent).

| Benchmark | TPR | FPR | ASR | No-defense ASR |
|---|---|---|---|---|
| MemoryPoisoning, app document markers | 48.4% (45/93 attempted poisoned saves) | 0.0% (0/70) | 19.2% (48/250) | 32.8% |
| MemoryPoisoning, no markers | 6.3% (11/174) | 0.0% (0/70) | 65.2% (163/250) | 67.6% |
| BIPIA | 32.4% (107/330) | 9.3% (14/150) | 3.6% (12/330) | 21.2% |
| AgentDojo (4 suites) | 30.7% (291/949) | 5.2% (5/97) | 1.5% (14/949) | 9.8% (93/949) |
| AgentHarm | 34.7% (61/176) | 32.4% (57/176) | harm score 0.055 | 0.293 |
| MCPTox | 45.0% (49/109 real successes) | 2.5% (9/357) | 4.5% (60/1348) | 8.1% (109/1348) |
| BIRD | 22.0% (88/400) | 1.0% (1/100) | 35.5% (142/400) | 38.5% (154/400) |

TPR is blocked attack cases / attack cases, except Memory (over the target's attempted poisoned saves) and MCPTox
(over the cases the GT labeler marked as real successes, as in runs/mcptox_gt_20261118/table.py).
AgentDojo utility: under attack 65.5%, benign 87.6%. BIRD by type (blocked / ASR): sensitive_data_access 70% / 7%,
destructive_operation 10% / 50% (real DB changed), scope_expansion 1% / 85%, db_content_indirect_injection 7% / 0%.

## LLM usage (tokens are exact; times are inflated, see below)

| Benchmark | Target tokens (in / out) | SimGuardAgent tokens: simulator + judge (in / out / reasoning) | Target $ | SimGuardAgent $ | Wall time |
|---|---|---|---|---|---|
| Memory | 6.50M / 0.20M | 12.82M / 0.60M / 0.10M | 0.91 | 1.35 | 19.8 min |
| BIPIA | 0.23M / 0.03M | 1.40M / 0.26M / 0.04M | 0.03 | 0.18 | 22.9 min |
| AgentDojo | 22.37M / 0.66M | 26.80M / 3.46M / 0.34M | 3.57 | 4.16 | 86.5 min |
| AgentHarm | 1.10M / 0.17M | 3.45M / 0.59M / 0.09M | 0.21 | 0.50 | 6.8 min |
| MCPTox | 2.30M / 0.07M | 11.26M / 1.40M / 0.14M | 0.32 | 1.53 | 31.2 min |
| BIRD | 2.96M / 0.17M | 7.38M / 0.96M / 0.17M | 0.51 | 1.01 | 118.3 min |

Graders not included (Memory goal-match 0.01 $, AgentHarm gpt-4o-mini 0.12 $; BIPIA / MCPTox / BIRD GT labels ran
afterwards). Wall times are not SimGuardAgent's overhead: ~160 Python processes ran on 32 logical cores (CPU 100%),
and loading the certificate bundle for every new client took 10-25 s until it was cached per process
(simguard/llm_backends.py `_tls_verify`, added during this run; AgentHarm was restarted after it).

## Open issues found in this run
- AgentHarm FPR 32.4%: ToolCallJudge's scope_violation / unauthorized_action / harmful_output fire on benign tasks
  written to look like the harmful ones (same as the 2026-11-18 run, 61/176).
- BIRD destructive_operation: 50/100 changed the real DB while only 10 were blocked; scope_expansion 85% ASR.
- BIPIA FPR: 6 table answers blocked by documents_disagree (a wrong count read as a document conflict) and 8 code
  answers (code in the answer goes to the execution route). documents_disagree was kept for SafeRAG, now dropped.
