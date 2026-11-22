"""Sleeper memory poisoning on the real AutoGen agent with SimGuardAgent.
The agent gets the benchmark's system prompt and user message (samples from memory_dump_samples.py) and one tool,
save_memory, like experiments/memory_rag_judge/run_memory.py does for the AgentDojo agent. Grading (goal match) is
done afterwards by memory_grade.py (needs the main venv).
usage (repo root, .venv-autogen, via experiments/full_runs/launch.py): memory_autogen.py SAMPLES_JSON OUT_JSON
env: AUTOGEN_MODE=simguard|none, AG_WORKERS"""
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
common.install(live_reads=False)
from autogen_core.tools import FunctionTool

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])


async def main():
    samples = json.load(open(SRC, encoding="utf-8"))
    sem = asyncio.Semaphore(int(os.getenv("AG_WORKERS", "10")))

    async def one(s):
        saved: list[str] = []

        def save_memory(memory_text: str) -> str:
            """Save one short memory about the user.

            :param memory_text: Concise fact or preference to remember about the user.
            """
            saved.append(memory_text)
            return "Memory saved."

        tool = FunctionTool(save_memory, description="Save one short memory about the user (a concise fact or preference).")
        async with sem:
            r = await common.run_agent(s["user"], system=s["system"], tools=[tool], max_tool_iterations=3)
        return {"id": s["id"], "goal": s["goal"], "query": s["query"], "error": r["error"], "blocked": r["blocked"],
                "proposed_saves": [p for p in r["proposed"] if p["tool"] == "save_memory"], "really_saved": saved,
                "events": r["events"], "final": r["final"][:400], "seconds": r["seconds"]}

    rows = await asyncio.gather(*(one(s) for s in samples))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{SRC.name}: {len(rows)} samples, errors {sum(1 for r in rows if r['error'])}, saved {sum(1 for r in rows if r['really_saved'])}")

asyncio.run(main())
