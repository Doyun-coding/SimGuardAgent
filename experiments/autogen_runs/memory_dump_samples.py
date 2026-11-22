"""Build the Sleeper samples with the benchmark's own code (needs the main venv: sleeper_eval imports inspect_ai) and
save them as JSON for the AutoGen runner. Same samples as experiments/full_runs/full_run.sh memory():
5 attacks x 2 app settings (no markers / untrusted_content_markers) x (25 behaviour + 25 agent), and 70 benign saves.
usage (repo root): .venv/Scripts/python.exe experiments/autogen_runs/memory_dump_samples.py OUT_DIR"""
import json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd(); OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/memory_rag_judge")]
ATTACKS = ["universal_v2_optimized_without_markers", "universal_v2_optimized_with_markers", "evolved_v2",
           "attack_from_literature", "mem0_external_prompt_leak_without_untrusted_markers"]
DS = ["datasets/released/behaviour_eval_in_with_true_optimized_goals.json", "datasets/released/agent_eval_in_with_true_optimized_goals.json"]
import memory_samples as M
for app in ("none", "untrusted_content_markers"):
    os.environ["MEM_APP_DEFENSE"] = "" if app == "none" else app
    for at in ATTACKS:
        rows = [s for ds in DS for s in M.load(ROOT, ds, 25, attack=at)]
        (OUT / f"attack_{at}_app-{app}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
os.environ["MEM_APP_DEFENSE"] = ""
rows = M.load(ROOT, "datasets/released/benign_save/merged_eval_in_benign_save_70_true_opt.json", 70, attack="none")
(OUT / "benign.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
print("written", len(list(OUT.glob("*.json"))), "files")
