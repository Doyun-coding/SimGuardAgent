"""Ground truth for SafeRAG attack responses (no defense), following SafeRAG's own success rules.

SA / ICC: the attack keyword (promoted brand, falsified figure) appears in the answer (SafeRAG keyword rule).
WDoS / SN: an LLM labeler with the criteria and prompt of runs/saferag_paper_asr_20261118/eval_rag_old.py,
which sees the golden and injected documents. The labeler model is independent of SimGuardAgent's judge.

usage (repo root): saferag_gt.py RESULTS_JSON OUT_JSONL      env: GT_MODEL, GT_WORKERS, GT_KEYWORD_METHODS (default SA,ICC)
"""
import asyncio, importlib.util, json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from simguard.config import make_client
from simguard.simulation.judges.risk_scale import parse_json_object

SRC, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
MODEL = os.getenv("GT_MODEL", "openrouter/openai/gpt-6-luna")
spec = importlib.util.spec_from_file_location("ev", "runs/saferag_paper_asr_20261118/eval_rag_old.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
DATA = {f"{r['attack_type']}_{r['id']}": r for r in json.load(open("attacks-bench/rag/saferag/nctd_poisoned.json", encoding="utf-8"))}


async def label(client, sem, r):
    e = DATA[r["id"]]; resp = (r.get("response") or "").strip()
    kw = ev.keyword_success(r["method"], e.get("attack_kws") or [], resp)
    if r["method"] in os.getenv("GT_KEYWORD_METHODS", "SA,ICC").split(",") and kw is not None:
        return {"id": r["id"], "method": r["method"], "success": bool(kw), "how": "keyword"}
    prompt = ev.ATTACK_PROMPT.format(criteria=ev.ATTACK_CRITERIA[r["method"]], question=e["question"],
                                     clean=ev._join(e.get("clean_contexts") or [], 8),
                                     poisoned=ev._join(e.get("poisoned_contexts") or [], 8), response=resp[:4000])
    async with sem:
        for _ in range(3):
            try:
                out = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=400, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}], extra_body={"reasoning": {"effort": "none", "exclude": True}})
                v = ev._yn(parse_json_object(out.choices[0].message.content or "").get("attack_success"))
                if v is not None:
                    return {"id": r["id"], "method": r["method"], "success": v, "how": "llm", "kw": kw}
            except Exception:
                await asyncio.sleep(2)
    return {"id": r["id"], "method": r["method"], "success": None, "how": "llm_failed", "kw": kw}


async def main():
    rows = json.load(open(SRC, encoding="utf-8"))["results"]
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    client, sem = make_client("judge"), asyncio.Semaphore(int(os.getenv("GT_WORKERS", "16")))
    with OUT.open("a", encoding="utf-8") as fh:
        for fut in asyncio.as_completed([label(client, sem, r) for r in rows if r["id"] not in done]):
            fh.write(json.dumps(await fut, ensure_ascii=False) + "\n"); fh.flush()

asyncio.run(main())
