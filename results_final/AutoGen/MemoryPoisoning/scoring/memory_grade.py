"""Grade AutoGen Sleeper results with the benchmark's goal-match grader (sleeper_eval.scorer, same model as
run_memory.py: the target model, reasoning off). Adds proposed_matches_goal / saved_matches_goal to each row.
usage (repo root, main venv, via experiments/full_runs/launch.py): memory_grade.py RESULT_JSON [RESULT_JSON ...]"""
import concurrent.futures, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "attacks-bench/memory/LLM-agent-memory-poisoning"))
import types
for name in ("openpyxl", "xlrd", "odf", "odf.opendocument", "odf.text", "odf.teletype", "striprtf", "striprtf.striprtf"):
    try:  # office/RTF readers sleeper_eval imports but grading never uses (same stubs as memory_samples.py)
        __import__(name)
    except ImportError:
        stub = types.ModuleType(name); stub.__getattr__ = lambda attr: None; sys.modules[name] = stub
from sleeper_eval.scorer import build_goal_match_prompt, build_benign_save_goal_match_prompt, parse_goal_match_verdict
from openai import OpenAI
client = OpenAI(); MODEL = os.getenv("AGENTDOJO_MODEL_ID", "openrouter/qwen/qwen3.5-9b")


def goal_match(goal, memory, benign):
    build = build_benign_save_goal_match_prompt if benign else build_goal_match_prompt
    r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=300, messages=[{"role": "user", "content": build(goal, memory)}],
                                       extra_body={"reasoning": {"effort": "none", "exclude": True}})
    return parse_goal_match_verdict(r.choices[0].message.content or "")


for path in sys.argv[1:]:
    rows = json.load(open(path, encoding="utf-8")); benign = Path(path).name.startswith("benign")
    def grade(r):
        r["proposed_matches_goal"] = [goal_match(r["goal"], (p["arguments"] or {}).get("memory_text", ""), benign) for p in r["proposed_saves"]]
        r["saved_matches_goal"] = [goal_match(r["goal"], m, benign) for m in r["really_saved"]]
        return r
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(grade, rows))
    json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(Path(path).name, "graded", len(rows))
