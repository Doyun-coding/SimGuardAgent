"""Copy the AutoGen results into results_final/AutoGen/<Benchmark>/ (run from the repo root), same layout as
results_final/AgentDojo/: simguard/, baseline/ (AutoGen without SimGuardAgent), labels/, scoring/, summary.md.
BIRD's per-case database copies are deleted by the runner and never copied."""
import shutil
from pathlib import Path

R = Path("runs/autogen_20261121"); OUT = Path("results_final/AutoGen"); E = Path("experiments/autogen_runs")
SCORE = [E / "autogen_score.py", E / "common.py", Path("experiments/full_runs/launch.py"), Path("experiments/full_runs/usage_site/sitecustomize.py")]
PLAN = {
    "AgentDojo": {"simguard": [R / "agentdojo"], "baseline": [R / "agentdojo_baseline"], "scoring": SCORE + [E / "agentdojo_autogen.py"]},
    "AgentHarm": {"simguard": [R / "agentharm"], "baseline": [R / "agentharm_baseline"], "scoring": SCORE + [E / "agentharm_autogen.py"]},
    "MCPTox": {"simguard": [R / "mcptox"], "baseline": [R / "mcptox_baseline"],
               "scoring": SCORE + [E / "mcptox_autogen.py", Path("experiments/full_runs/gt_label_mcptox.py")]},
    "MemoryPoisoning": {"simguard": [R / "memory"], "baseline": [R / "memory_baseline"], "labels": [R / "memory_samples"],
                        "scoring": SCORE + [E / "memory_autogen.py", E / "memory_grade.py", E / "memory_dump_samples.py"]},
    "BIPIA": {"simguard": [R / "bipia"], "baseline": [R / "bipia_baseline"],
              "scoring": SCORE + [E / "bipia_autogen.py", Path("experiments/rag_harm/bipia_samples.py"), Path("experiments/rag_harm/bipia_gt.py")]},
    "BIRD": {"simguard": [R / "bird"], "baseline": [R / "bird_baseline"], "scoring": SCORE + [E / "bird_autogen.py", Path("experiments/full_runs/gt_label_bird.py")]},
    "SABER": {"simguard": [R / "saber"], "baseline": [R / "saber_baseline"],
              "scoring": SCORE + [E / "code_autogen.py", Path("experiments/full_runs/exec_score.py"), Path("experiments/full_runs/saber_task_asr.py")]},
    "RedCode-Exec": {"simguard": [R / "redcode"], "baseline": [R / "redcode_baseline"],
                     "scoring": SCORE + [E / "code_autogen.py", Path("attacks-bench/RedCode/evaluation/RedCode_Exec/evaluation.py")]},
    "GA_Jailbreak": {"simguard": [Path("runs/ga_contentaxes_20261117/autogen")],
                     "labels": [Path("runs/ga_qwen_judge_20261118/table.md"), Path("runs/ga_qwen_judge_20261118/autogen_attack.jsonl"),
                                Path("runs/ga_qwen_judge_20261118/autogen_benign.jsonl")],
                     "scoring": [Path("runs/ga_qwen_judge_20261118/table.py"), Path("agents/autogen/run_autogen_simguard.py")]},
}
for bench, parts in PLAN.items():
    for part, sources in parts.items():
        for src in sources:
            if not src.exists():
                print(f"MISSING {bench}/{part}: {src}"); continue
            dst = OUT / bench / part / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=lambda d, n: [x for x in n if x.endswith("_agent_dbs")])
            else:
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    print("ok", bench)
