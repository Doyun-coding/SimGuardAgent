"""BIPIA on the real AutoGen agent with SimGuardAgent. Same samples and prompts as experiments/rag_harm/bipia_run.py
(experiments/rag_harm/bipia_samples.py): the external content goes into BIPIA's system prompt, the question is the
task. The application knows which text is external content, so the run is wrapped in
simguard.retrieved_context(content): BehaviorJudge then adds its RAG checks, as with the AgentDojo agent (which passes
extra_args["retrieved_context"]). Labels: experiments/rag_harm/bipia_gt.py on the output.
usage (repo root, .venv-autogen, via launch.py): bipia_autogen.py OUT_JSON attack|benign    env: AUTOGEN_MODE, AG_WORKERS"""
import asyncio, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "experiments/rag_harm")]
import common
common.install(live_reads=False)
from bipia_samples import SYSTEM, USER, samples

OUT, LABEL = Path(sys.argv[1]), sys.argv[2]


async def main():
    import simguard
    sem = asyncio.Semaphore(int(os.getenv("AG_WORKERS", "16")))

    async def one(s):
        system, user = SYSTEM[s["task"]].format(context=s["context"]), USER[s["task"]].format(**s)
        async with sem:
            with simguard.retrieved_context(s["context"]):
                r = await common.run_agent(user, system=system)
        row = {k: s.get(k) for k in ("id", "task", "attack_name", "attack_str", "position", "question", "ideal")}
        return {**row, "system": system, "user": user, "context": s["context"], "response": r["final"], "error": r["error"],
                "blocked": r["blocked"], "events": r["events"], "routes": r["routes"], "seconds": r["seconds"]}

    rows = await asyncio.gather(*(one(s) for s in samples(LABEL)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{LABEL}: {len(rows)} samples, blocked {sum(bool(r['blocked']) for r in rows)}, errors {sum(1 for r in rows if r['error'])}")

asyncio.run(main())
