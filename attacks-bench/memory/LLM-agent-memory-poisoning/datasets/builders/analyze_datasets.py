#!/usr/bin/env python3
"""
Analyze JSON dataset files under datasets/.

Expects: datasets/in_dist/ and datasets/out_of_dist/ (each containing .json files).
Reports: structure, entry counts, field stats, and distributions.
"""
import json
import sys
from pathlib import Path
from collections import Counter

DATASETS_DIR = Path(__file__).resolve().parents[1]
SUBDIRS = ("in_dist", "out_of_dist")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def analyze_file(path: Path) -> dict:
    data = load_json(path)
    out = {
        "path": path.name,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "root_type": type(data).__name__,
        "count": None,
        "structure": {},
        "stats": {},
        "distributions": {},
    }
    if isinstance(data, list):
        out["count"] = len(data)
        if not data:
            return out
        entry = data[0]
        out["structure"] = {k: type(v).__name__ for k, v in entry.items()}
        # Nested structure
        for key in ("document", "goal", "query", "preexisting_memories"):
            if key not in entry:
                continue
            v = entry[key]
            if isinstance(v, dict):
                out["structure"][f"{key}.keys"] = list(v.keys())
            elif isinstance(v, list):
                out["structure"][f"{key}.len"] = len(v)
                if v and isinstance(v[0], dict):
                    out["structure"][f"{key}.item_keys"] = list(v[0].keys())

        # Stats
        doc_text_lens = []
        goal_text_lens = []
        has_memories = "preexisting_memories" in entry
        num_memories = []
        domains = []
        subcategory_ids = []
        category_ids = []
        goal_source_splits = []  # when present (e.g. in out_of_dist)
        goal_domain_seeds = []   # for unique list

        for e in data:
            doc = e.get("document") or {}
            goal = e.get("goal") or {}
            t = doc.get("text") or ""
            doc_text_lens.append(len(t))
            gt = goal.get("goal_text") or ""
            goal_text_lens.append(len(gt))
            domains.append(doc.get("domain") or "?")
            subcategory_ids.append(goal.get("subcategory_id") or "?")
            category_ids.append(goal.get("category_id") or "?")
            ds = goal.get("domain_seed")
            if ds:
                goal_domain_seeds.append(ds)
            if "goal_source_split" in goal:
                goal_source_splits.append(goal.get("goal_source_split") or "?")
            if has_memories:
                pre = e.get("preexisting_memories") or {}
                mems = pre.get("memories") if isinstance(pre.get("memories"), list) else []
                num_memories.append(len(mems))

        out["stats"] = {
            "document.text len": f"min={min(doc_text_lens)} max={max(doc_text_lens)} avg={sum(doc_text_lens)//len(doc_text_lens)}",
            "goal.goal_text len": f"min={min(goal_text_lens)} max={max(goal_text_lens)} avg={sum(goal_text_lens)//len(goal_text_lens)}",
        }
        if num_memories:
            out["stats"]["preexisting_memories count"] = f"min={min(num_memories)} max={max(num_memories)} avg={sum(num_memories)/len(num_memories):.1f}"

        out["distributions"] = {
            "document.domain": dict(Counter(domains)),
            "goal.subcategory_id": dict(Counter(subcategory_ids)),
            "goal.category_id": dict(Counter(category_ids)),
        }
        if goal_source_splits:
            out["distributions"]["goal.goal_source_split"] = dict(Counter(goal_source_splits))
        out["unique_goal_domain_seeds"] = sorted(set(goal_domain_seeds))
    return out


def _print_file_report(info: dict, rel_path: str) -> None:
    """Print structure, stats, and distributions for one file."""
    print(f"\n--- {rel_path} ---")
    print(f"  Size:     {info['size_mb']:.2f} MB")
    print(f"  Type:     {info['root_type']}")
    print(f"  Entries:  {info['count']}")

    if info["structure"]:
        print("  Structure (first entry):")
        for k, v in info["structure"].items():
            if k.endswith(".keys") or k.endswith(".item_keys"):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")

    if info["stats"]:
        print("  Stats:")
        for k, v in info["stats"].items():
            print(f"    {k}: {v}")

    if info["distributions"]:
        print("  Distributions:")
        for name, dist in info["distributions"].items():
            items = sorted(dist.items(), key=lambda x: -x[1])[:10]
            parts = [f"{k}={v}" for k, v in items]
            print(f"    {name}: {', '.join(parts)}")

    if info.get("unique_goal_domain_seeds"):
        seeds = info["unique_goal_domain_seeds"]
        print("  Unique goal.domain_seed:")
        print(f"    ({len(seeds)}): {', '.join(seeds)}")


def main() -> None:
    if not DATASETS_DIR.exists():
        print(f"Directory not found: {DATASETS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Collect (subdir_name, path) for each JSON file
    groups: dict[str, list[Path]] = {}
    for sub in SUBDIRS:
        sub_path = DATASETS_DIR / sub
        if not sub_path.is_dir():
            continue
        json_files = sorted(sub_path.glob("*.json"))
        if json_files:
            groups[sub] = json_files

    if not groups:
        print(f"No subdirs {SUBDIRS} with .json files under {DATASETS_DIR}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("datasets/ — in-distribution and out-of-distribution analysis")
    print("=" * 70)

    all_results: list[tuple[str, Path, dict]] = []  # (subdir, path, info)

    for subdir in SUBDIRS:
        if subdir not in groups:
            continue
        paths = groups[subdir]
        print(f"\n{'=' * 70}")
        print(f"  {subdir.upper().replace('_', ' ')}")
        print("=" * 70)
        for path in paths:
            rel = f"{subdir}/{path.name}"
            info = analyze_file(path)
            all_results.append((subdir, path, info))
            _print_file_report(info, rel)

    # Overall summary
    print("\n" + "=" * 70)
    print("Summary: split -> file -> entry count")
    print("=" * 70)
    for subdir in SUBDIRS:
        sub_results = [(p, i) for s, p, i in all_results if s == subdir]
        if not sub_results:
            continue
        total = sum(i["count"] or 0 for _, i in sub_results)
        print(f"\n  {subdir}/")
        for path, info in sub_results:
            print(f"    {path.name}: {info['count']} entries")
        print(f"    -> total: {total} entries")
    grand_total = sum(i["count"] or 0 for _, _, i in all_results)
    print(f"\n  ALL: {grand_total} entries")
    print("=" * 70)


if __name__ == "__main__":
    main()
