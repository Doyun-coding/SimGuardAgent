"""
LlamaGuard 4 (12B) GA_Jailbreak 평가
openrouter/meta-llama/llama-guard-4-12b 사용

결과: results/attacks-bench/jailbreak/llamaguard4__GA_Jailbreak__attack.json
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL    = "openrouter/meta-llama/llama-guard-4-12b"
BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY  = os.getenv("OPENAI_API_KEY", "")
WORKERS  = 10


# ─────────────────────────────────────────────────────────────────────────────

async def classify(client: httpx.AsyncClient, task: str) -> tuple[bool, str]:
    """(blocked, raw_output)"""
    resp = await client.post(
        BASE_URL + "/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": task}],
            "max_tokens": 50,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    blocked = content.lower().startswith("unsafe")
    return blocked, content


async def worker(sem, client, entry, results, idx, total):
    async with sem:
        try:
            blocked, raw = await classify(client, entry["task"])
            entry["blocked"] = blocked
            entry["raw"]     = raw
            entry["error"]   = None
        except Exception as e:
            entry["blocked"] = False
            entry["raw"]     = None
            entry["error"]   = str(e)[:200]
        results.append(entry)
        done = len(results)
        if done % 50 == 0 or done == total:
            n_block = sum(1 for r in results if r.get("blocked"))
            print(f"  [{done}/{total}]  BLOCK={n_block}  TPR={n_block/done*100:.1f}%")


async def main(benchmark: str, label: str, limit: int | None, workers: int, result: str | None = None):
    from run_benchmark import BENCHMARKS, _load_ga_jailbreak

    if benchmark not in BENCHMARKS:
        print(f"알 수 없는 벤치마크: {benchmark}. 선택 가능: {list(BENCHMARKS)}")
        sys.exit(1)

    loader, subdir = BENCHMARKS[benchmark]
    if benchmark == "GA_Jailbreak":
        entries = loader(limit, label=label)
    else:
        entries = loader(limit)

    auto_name = f"llamaguard4__{benchmark}{'__' + label if benchmark == 'GA_Jailbreak' else ''}.json"
    if result:
        p = Path(result)
        if result.endswith("/") or result.endswith("\\") or p.is_dir():
            out_file = p / auto_name
        else:
            out_file = p
    else:
        out_file = Path("results/attacks-bench") / subdir / auto_name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"LlamaGuard 4  모델={MODEL}")
    print(f"벤치마크: {benchmark}  label={label}  데이터: {len(entries)}건  workers={workers}\n")

    results = []
    sem     = asyncio.Semaphore(workers)
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {API_KEY}"},
        verify=False,
    ) as client:
        tasks = [
            worker(sem, client, e, results, i, len(entries))
            for i, e in enumerate(entries)
        ]
        await asyncio.gather(*tasks)

    # 저장
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "results": results}, f, ensure_ascii=False, indent=2)

    # 요약
    total   = len(results)
    blocked = sum(1 for r in results if r.get("blocked"))
    errors  = sum(1 for r in results if r.get("error"))
    tpr     = blocked / (total - errors) * 100 if total - errors else 0

    print(f"\n{'='*60}")
    print(f"모델: {MODEL}")
    print(f"전체: {total}건  BLOCK: {blocked}건  TPR: {tpr:.1f}%  오류: {errors}건")
    print(f"결과: {out_file}")
    print(f"{'='*60}")

    # 카테고리별 TPR
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"block": 0, "allow": 0})
    for r in results:
        cat = r.get("category", "unknown")
        if r.get("error"):
            continue
        if r.get("blocked"):
            by_cat[cat]["block"] += 1
        else:
            by_cat[cat]["allow"] += 1

    print("\n카테고리별 TPR:")
    for cat, v in sorted(by_cat.items(), key=lambda x: -(x[1]["block"] + x[1]["allow"])):
        n = v["block"] + v["allow"]
        print(f"  {cat:<40} {v['block']:>4}/{n:<4}  {v['block']/n*100:.1f}%")


if __name__ == "__main__":
    import argparse
    from run_benchmark import BENCHMARKS
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="GA_Jailbreak", choices=list(BENCHMARKS))
    p.add_argument("--label",     default="attack", choices=["attack", "benign", "all"])
    p.add_argument("--limit",     type=int, default=50)
    p.add_argument("--workers",   type=int, default=WORKERS)
    p.add_argument("--result",    default=None,
                   help="결과 저장 경로 (기본값: results/... 자동 생성)")
    args = p.parse_args()
    asyncio.run(main(args.benchmark, args.label, args.limit, args.workers, args.result))
