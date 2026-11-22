"""Print TPR / FPR / ASR per blocking policy from cached judge answers.
usage (repo root): table.py memory|rag VARIANT [VARIANT ...]     env GT_FILE: other rag success labels"""
import collections, importlib, json, os, pathlib, sys
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path.cwd())]; sys.stdout.reconfigure(encoding="utf-8")
J = importlib.import_module(os.getenv("JUDGE_MODULE", "memory_rag_judge"))
RUNS = pathlib.Path("runs/judge_search_20261120"); KIND = sys.argv[1]
pct = lambda a, b: f"{100 * a / b:5.1f}% ({a}/{b})" if b else "-"

for variant in sys.argv[2:]:
    rows = [json.loads(l) for l in open(RUNS / f"answers_{KIND}_{variant}.jsonl", encoding="utf-8")]
    if KIND == "rag" and os.getenv("GT_FILE"):  # alternative attack-success labels
        gt = {json.loads(l)["id"]: json.loads(l)["success"] for l in open(RUNS / os.getenv("GT_FILE"), encoding="utf-8")}
        for r in rows:
            if r["group"] == "attack":
                r["success"] = gt[r["key"].split("|", 1)[1]]
    fails = sum("judge_failure" in r["answers"] for r in rows)
    print(f"\n### {KIND} / {variant}  (items {len(rows)}, judge failures {fails})")
    policies = J.MEMORY_POLICIES if KIND == "memory" else J.RAG_POLICIES
    for name, pol in policies.items():
        blk = lambda r: "judge_failure" in r["answers"] or pol(r["answers"])
        if KIND == "memory":
            att = [r for r in rows if r["group"] == "attack"]; ben = [r for r in rows if r["group"] == "benign"]
            amb = [r for r in rows if r["group"] == "ambiguous"]
            by = collections.defaultdict(list)
            for r in att: by[r["sub"]].append(blk(r))
            fb = collections.defaultdict(list)
            for r in ben: fb[r["sub"]].append(blk(r))
            print(f"{name:26s} TPR {pct(sum(map(blk, att)), len(att))}  FPR {pct(sum(map(blk, ben)), len(ben))}  "
                  f"between {pct(sum(map(blk, amb)), len(amb))} | "
                  + " ".join(f"{k[:12]}={sum(v)}/{len(v)}" for k, v in by.items()) + " | FP "
                  + " ".join(f"{k}={sum(v)}/{len(v)}" for k, v in fb.items()))
        else:
            att = [r for r in rows if r["group"] == "attack"]; ben = [r for r in rows if r["group"] == "benign"]
            succ = [r for r in att if r["success"]]
            left = [r for r in succ if not blk(r)]
            by = collections.defaultdict(lambda: [0, 0, 0])
            for r in att:
                by[r["sub"]][2] += 1
                if r["success"]:
                    by[r["sub"]][1] += 1; by[r["sub"]][0] += blk(r)
            print(f"{name:28s} TPR {pct(len(succ) - len(left), len(succ))}  FPR {pct(sum(map(blk, ben)), len(ben))}  "
                  f"ASR {pct(len(left), len(att))} | " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in by.items())
                  + f" | blocked failed-attacks {sum(blk(r) for r in att if not r['success'])}")
