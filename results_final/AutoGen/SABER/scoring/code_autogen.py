"""SABER and RedCode-Exec on the real AutoGen agent with SimGuardAgent.

Same tasks, Docker sessions, step limit, scoring and result format as the AgentDojo-agent runs
(attacks-bench/agentdojo_adapter/agentdojo_adapter.py _run_saber_task); only the agent loop is AutoGen: an
AssistantAgent with the task's system prompt and the bash / python tools. Each tool call is gated exactly as in the
adapter: SimGuardAgentInterceptor.intercept_round runs it in the task's Docker session, predicts the agent's next step
(follow_up = the target model's next response to the sandbox result) and ExecutionJudge decides. ALLOW commits the
result to the agent; BLOCK discards it and ends the task. AUTOGEN_MODE=none runs the call directly (baseline).
usage (repo root, .venv-autogen, via launch.py): code_autogen.py SABER|redcode OUT_JSON [--label attack|benign]
env: AUTOGEN_MODE, AG_WORKERS (tasks at once), AG_JUDGE_WORKERS"""
import asyncio, importlib.util, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common

BENCH, OUT = sys.argv[1], Path(sys.argv[2])
LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "attack"
spec = importlib.util.spec_from_file_location("ad", common.ROOT / "attacks-bench/agentdojo_adapter/agentdojo_adapter.py")
ad = importlib.util.module_from_spec(spec); sys.modules["ad"] = ad; spec.loader.exec_module(ad)
sys.argv = ["x", "--benchmark", BENCH, "--mode", "simguard" if common.MODE == "simguard" else "baseline", "--label", LABEL,
            "--model", "openai-compatible", "--model-id", common.TARGET, "--judge-model-id", "openrouter/openai/gpt-oss-120b",
            "--judge-no-think", "--results", str(OUT)]
args = ad.parse_args(); ad._configure_endpoint(args)
from autogen_core.tools import FunctionTool
from openai import AsyncOpenAI
from simguard import Action, SimGuardAgentInterceptor
from simguard.interceptor import render_conversation


async def run_task(task, sem, judge_sem):
    async with sem:
        started = time.monotonic()
        setup = task.get("task", {}).get("setup", {})
        user_prompt = str(setup.get("user_prompt", "")); system = str(setup.get("system_prompt", "You are a helpful coding agent."))
        is_redcode = task.get("_benchmark") == "RedCode-Exec"
        session = ad.SaberDockerSession(args.redcode_image if is_redcode else args.saber_image, args.exec_timeout,
                                        str(setup.get("cwd") or "/home/user/project"), redcode_official=is_redcode)
        client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url, timeout=args.llm_timeout, max_retries=1)
        history = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]  # what the target saw
        state = {"step": 0, "blocked": False, "error": None}
        steps, commands, follow_ups = [], [], []

        async def gate(tool_name: str, arguments: dict) -> str:
            if state["blocked"] or state["error"] or state["step"] >= args.max_steps:
                return ad.BLOCKED_MESSAGE if hasattr(ad, "BLOCKED_MESSAGE") else "This action was blocked by SimGuardAgent."
            state["step"] += 1; step = state["step"]
            code = str(arguments.get("command") or arguments.get("code") or ""); commands.append(code)
            call_id = f"call_{step}"
            history.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]})
            follow = {"text": ""}

            async def observe(results):
                for result in results:
                    visible = json.dumps({k: result.get(k) for k in ("returncode", "stdout", "stderr", "state_diff")}, ensure_ascii=False)
                    msgs = history + [{"role": "tool", "tool_call_id": call_id, "content": visible}]
                resp = await client.chat.completions.create(model=common.TARGET, messages=msgs, tools=ad.SABER_TOOLS, tool_choice="auto",
                                                            temperature=0, max_tokens=args.llm_max_tokens, reasoning_effort=args.reasoning_effort)
                follow["text"] = ad._follow_up_evidence(resp.choices[0].message); follow_ups.append(follow["text"])
                return follow["text"]

            session.prepare_round(task, step, observe, judge_sem)
            if common.MODE == "simguard":
                trace = (await SimGuardAgentInterceptor().intercept_round(
                    [Action(user_request=user_prompt, tool_name=tool_name, tool_arguments=arguments,
                            conversation=render_conversation(history[:-1]) or None)], session))[0]
                events = [e.model_dump() for e in trace.security_events]
                decision = "ERROR" if any(e["type"] == "judge_failure" for e in events) else trace.decision.value
                result, risk = trace.execution_result, trace.risk_score
            else:
                result = await asyncio.to_thread(session.execute_action, tool_name, arguments)
                decision, risk, events = "ALLOW", 0.0, []
            result["agent_response"] = follow["text"]
            steps.append({"step": step, "tool": tool_name, "arguments": arguments, "result": result, "follow_up_response": follow["text"],
                          "decision": decision, "risk_score": risk, "events": events, "committed": decision == "ALLOW",
                          "guard_stage": "post_execution" if common.MODE == "simguard" else "none"})
            if decision != "ALLOW":
                state["blocked" if decision == "BLOCK" else "error"] = True if decision == "BLOCK" else "ExecutionJudge failed"
                return "This action was blocked by SimGuardAgent."
            visible = json.dumps({k: result.get(k) for k in ("returncode", "stdout", "stderr", "state_diff")}, ensure_ascii=False)
            history.append({"role": "tool", "tool_call_id": call_id, "content": visible})
            return visible

        async def bash(command: str) -> str:
            """Run a shell command in the task project directory."""
            return await gate("bash", {"command": command})

        async def python(code: str) -> str:
            """Run Python code in the task project directory."""
            return await gate("python", {"code": code})

        try:
            await asyncio.to_thread(session.start, setup.get("file_contents", {}))
            tools = [FunctionTool(bash, description="Run a shell command in the task project directory."),
                     FunctionTool(python, description="Run Python code in the task project directory.")]
            r = await common.run_agent(user_prompt, system=system, tools=tools, max_tool_iterations=args.max_steps)
            matches = ad._harmful_command_matches(task, commands)
            official = None
            if is_redcode and not state["error"]:
                official = await asyncio.to_thread(ad._redcode_official_score, args, task, session, steps, state["blocked"])
            return {**task.get("_result_metadata", {}), **({"redcode_official": official} if official is not None else {}),
                    "task_id": task.get("task_id"), "category": task.get("category"), "difficulty": task.get("difficulty"),
                    "scenario": task.get("scenario"), "injection_enabled": bool(task.get("task", {}).get("injection", {}).get("enabled")),
                    "blocked": state["blocked"], "harmful_command_proposed": bool(matches), "tool_call_proposed": bool(commands),
                    "no_tool_call": not commands, "action_executed": any(not s["result"].get("skipped") for s in steps),
                    "matched_patterns": matches, "commands": commands, "steps": steps, "follow_up_responses": follow_ups,
                    "final_response": r["final"][:4000], "error": state["error"] or r["error"], "agent": "autogen",
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
        except Exception as exc:
            return {**task.get("_result_metadata", {}), "task_id": task.get("task_id"), "category": task.get("category"), "blocked": True,
                    "steps": steps, "commands": commands, "error": f"{type(exc).__name__}: {exc}", "agent": "autogen"}
        finally:
            await asyncio.to_thread(session.close)


async def main():
    tasks = (ad._load_redcode_tasks(Path(args.redcode_dataset), None) if BENCH.lower() == "redcode"
             else ad._load_saber_tasks(Path(args.tasks), LABEL, None))
    if os.getenv("AG_LIMIT"):
        tasks = tasks[:int(os.environ["AG_LIMIT"])]
    sem, judge_sem = asyncio.Semaphore(int(os.getenv("AG_WORKERS", "8"))), asyncio.Semaphore(int(os.getenv("AG_JUDGE_WORKERS", "8")))
    results = []
    for i, fut in enumerate(asyncio.as_completed([run_task(t, sem, judge_sem) for t in tasks]), 1):
        results.append(await fut)
        if i % 25 == 0:
            print(f"[{BENCH}] {i}/{len(tasks)}", flush=True)
            OUT.write_text(json.dumps({"agent": "autogen", "benchmark": BENCH, "label": LABEL, "mode": common.MODE, "results": results},
                                      ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"agent": "autogen", "benchmark": BENCH, "label": LABEL, "mode": common.MODE, "results": results},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[{BENCH}] done {len(results)} blocked {sum(bool(r.get('blocked')) for r in results)} errors {sum(1 for r in results if r.get('error'))}")

asyncio.run(main())
