"""AgentDojo per-suite table from trace logs. usage: dojo_table.py RUN_DIR"""
import glob, json, statistics, sys
R = sys.argv[1]
dirs = {"workspace": f"{R}/logs/local-simguard/workspace"}
for s in ("travel", "banking", "slack"):
    dirs[s] = f"{R}/logs_{s}/local-simguard/{s}"


def blocked(r):
    return any(m["role"] == "assistant" and "blocked by SimGuardAgent" in str(m.get("content")) for m in r["messages"])


print("| Suite | 공격 케이스 | 공격 성공 (ASR) | 공격 하 utility | BLOCK 케이스 | 주입과제 단독 utility | 주입과제 단독 BLOCK | 시간 중앙값 |")
print("|---|---|---|---|---|---|---|---|")
tot = {"n": 0, "asr": 0, "ut": 0, "blk": 0, "itn": 0, "itu": 0, "itb": 0}
durs = []
for s, d in dirs.items():
    a = [json.load(open(f, encoding="utf-8")) for f in glob.glob(f"{d}/user_task_*/tool_knowledge/*.json")]
    it = [json.load(open(f, encoding="utf-8")) for f in glob.glob(f"{d}/injection_task_*/none/none.json")]
    n, asr, ut, blk = len(a), sum(bool(r.get("security")) for r in a), sum(bool(r.get("utility")) for r in a), sum(map(blocked, a))
    itu, itb = sum(bool(r.get("utility")) for r in it), sum(map(blocked, it))
    ds = [r["duration"] for r in a if r.get("duration")]; durs += ds
    for k, v in zip(tot, (n, asr, ut, blk, len(it), itu, itb)):
        tot[k] += v
    print(f"| {s} | {n} | {asr} ({asr/max(1,n):.1%}) | {ut} ({ut/max(1,n):.1%}) | {blk} ({blk/max(1,n):.1%}) | {itu}/{len(it)} | {itb}/{len(it)} | {statistics.median(ds) if ds else 0:.1f}s |")
t = tot
print(f"| **합계** | {t['n']} | {t['asr']} ({t['asr']/max(1,t['n']):.1%}) | {t['ut']} ({t['ut']/max(1,t['n']):.1%}) | {t['blk']} ({t['blk']/max(1,t['n']):.1%}) | {t['itu']}/{t['itn']} | {t['itb']}/{t['itn']} | {statistics.median(durs):.1f}s |")
