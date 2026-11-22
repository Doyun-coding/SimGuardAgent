"""Judge recorded, undefended BIPIA answers offline (same method as experiments/memory_rag_judge/search.py).
The external content is the retrieved context; the judge sees the user message, that content and the answer.
usage (repo root, via run_with_env.py): judge_eval.py run|table VARIANT
env: JUDGE_MODULE (current_judges | conditional_judges, from experiments/memory_rag_judge), RAG_MODE, WORKERS
"""
import asyncio, collections, importlib, json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd(); sys.path[:0] = [str(ROOT), str(ROOT / "experiments/memory_rag_judge")]
sys.stdout.reconfigure(encoding="utf-8")
RUNS = ROOT / "runs/rag_harm_20261121"
CMD, VARIANT = sys.argv[1], sys.argv[2]
J = importlib.import_module(os.getenv("JUDGE_MODULE", "conditional_judges"))
OUT = RUNS / f"answers_bipia_{VARIANT}.jsonl"


def items():
    gt = {json.loads(l)["id"]: json.loads(l)["success"] for l in open(RUNS / "bipia_gt.jsonl", encoding="utf-8")}
    rows = []
    for f, group in (("bipia_baseline_attack.json", "attack"), ("bipia_baseline_benign.json", "benign")):
        for r in json.load(open(RUNS / f, encoding="utf-8")):
            if r.get("error"):
                continue
            rows.append({"key": r["id"], "group": group, "task": r["task"],
                         "category": (r["attack_name"] or "benign").rsplit("-", 1)[0],
                         "success": gt.get(r["id"]) if group == "attack" else None,
                         "question": r["user"], "documents": r["context"], "answer": r["response"] or ""})
    return rows


async def run():
    todo = items()
    done = {json.loads(l)["key"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    todo = [x for x in todo if x["key"] not in done]
    print(f"bipia/{VARIANT}: todo={len(todo)}", flush=True)
    judge, sem = J.GroundedAnswerJudge(), asyncio.Semaphore(int(os.getenv("WORKERS", "16")))

    async def one(x):
        async with sem:
            a = await judge.judge(x["question"], x["documents"], x["answer"])
        return {**{k: x[k] for k in ("key", "group", "task", "category", "success")}, "answers": a}

    with OUT.open("a", encoding="utf-8") as fh:
        for fut in asyncio.as_completed([one(x) for x in todo]):
            fh.write(json.dumps(await fut, ensure_ascii=False) + "\n"); fh.flush()


def table():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    pct = lambda a, b: f"{100 * a / b:5.1f}% ({a}/{b})" if b else "-"
    for name, pol in J.RAG_POLICIES.items():
        blk = lambda r: pol(r["answers"])
        att = [r for r in rows if r["group"] == "attack"]; ben = [r for r in rows if r["group"] == "benign"]
        succ = [r for r in att if r["success"]]; left = [r for r in succ if not blk(r)]
        by = collections.defaultdict(lambda: [0, 0])
        for r in succ:
            by[f"{r['task']}:{r['category'][:12]}"][0] += blk(r); by[f"{r['task']}:{r['category'][:12]}"][1] += 1
        fb = collections.defaultdict(lambda: [0, 0])
        for r in ben:
            fb[r["task"]][0] += blk(r); fb[r["task"]][1] += 1
        print(f"{name:22s} TPR {pct(len(succ) - len(left), len(succ))}  FPR {pct(sum(map(blk, ben)), len(ben))}  "
              f"ASR {pct(len(left), len(att))} | " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by.items()))
              + " | FP " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in fb.items())
              + f" | blocked failed-attacks {sum(blk(r) for r in att if not r['success'])}")


if CMD == "run":
    asyncio.run(run())
else:
    table()
