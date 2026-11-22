"""Copy the final result data of every benchmark into results_final/<Benchmark>/ (run from the repo root).
Each folder: simguard/ (SimGuardAgent run), baseline/ (no-defense run used for the ASR comparison), labels/ (attack
success ground truth), usage/ (LLM call logs, when measured), scoring/ (the script that made the numbers).
BIRD's per-case agent database copies (bird_simguard_agent_dbs, ~535 MB) are not result data and are skipped."""
import shutil
from pathlib import Path

SCRATCH = Path(r"C:/Users/inktr/AppData/Local/Temp/claude/04c7bc1874d3543e/a483fd15-d89f-4aff-a028-ef5f76c85a3e/scratchpad")
F = Path("runs/full_20261121")
OUT = Path("results_final")
SKIP = {"bird_simguard_agent_dbs", "bird_baseline_agent_dbs", "smoke_agent_dbs"}

PLAN = {
    "AgentDojo": {
        "simguard": [F / "agentdojo"],
        "baseline": [Path("runs/agentdojo_baseline_20261118")],
        "scoring": [Path("experiments/full_runs/score.py")],
    },
    "AgentHarm": {
        "simguard": [F / "agentharm"],
        "baseline": [Path("runs/agentharm_baseline_20261118")],
        "scoring": [Path("experiments/full_runs/score.py"), Path("experiments/full_runs/run_agentharm_logged.py")],
    },
    "MCPTox": {
        "simguard": [F / "mcptox"],
        "baseline": [Path("runs/mcptox_120b_20261118/mcptox_baseline.json"), Path("runs/mcptox_gt_20261118/gt_baseline_run.jsonl")],
        "scoring": [Path("experiments/full_runs/score.py"), Path("experiments/full_runs/gt_label_mcptox.py")],
    },
    "MemoryPoisoning": {
        "simguard": [F / "memory"],
        "baseline": sorted(Path("runs/judge_search_20261120").glob("mem_base_*.json")),
        "scoring": [Path("experiments/full_runs/score.py"), Path("experiments/memory_rag_judge/run_memory.py"),
                    Path("experiments/memory_rag_judge/memory_samples.py")],
    },
    "BIPIA": {
        "simguard": [F / "bipia"],
        "baseline": [Path("runs/rag_harm_20261121/bipia_baseline_attack.json"), Path("runs/rag_harm_20261121/bipia_baseline_benign.json"),
                     Path("runs/rag_harm_20261121/bipia_gt.jsonl")],
        "scoring": [Path("experiments/full_runs/score.py"), Path("experiments/rag_harm/bipia_run.py"), Path("experiments/rag_harm/bipia_gt.py")],
    },
    "BIRD": {
        "simguard": [F / "bird"],
        "baseline": [Path("runs/bird_baseline_20261120")],
        "scoring": [Path("experiments/full_runs/score.py"), Path("experiments/full_runs/gt_label_bird.py")],
    },
    "GA_Jailbreak": {
        "simguard": [Path("runs/ga_contentaxes_20261117/agentdojo")],
        "labels": [Path("runs/ga_qwen_judge_20261118/table.md"), Path("runs/ga_qwen_judge_20261118/agentdojo_attack.jsonl"),
                   Path("runs/ga_qwen_judge_20261118/agentdojo_benign.jsonl"), Path("runs/ga_contentaxes_20261117/compare.md")],
        "scoring": [Path("runs/ga_qwen_judge_20261118/table.py"), Path("runs/ga_qwen_judge_20261118/rejudge_qwen.py")],
    },
    "SABER": {
        "simguard": [Path("runs/exec_20261120_judgev4/SABER_simguard_attack.json"), Path("runs/exec_20261120_judgev4/SABER_simguard_benign.json"),
                     Path("runs/exec_20261120_judgev4/SABER_simguard_attack.log"), Path("runs/exec_20261120_judgev4/SABER_simguard_benign.log")],
        "baseline": [Path("runs/exec_20261120_judgev4/SABER_baseline_attack.json"), Path("runs/exec_20261120_judgev4/SABER_baseline_benign.json")],
        "usage": [Path("runs/cost_20261120")],
        "scoring": [SCRATCH / "exec_score.py", SCRATCH / "saber_task_asr.py"],
    },
    "RedCode-Exec": {
        "simguard": [Path("runs/exec_20261120_redcode_official/RedCode_simguard_attack_merged.json"),
                     Path("runs/exec_20261120_redcode_official/RedCode_simguard_attack.log"),
                     Path("runs/exec_20261120_redcode_official/cat18_fix"), Path("runs/exec_20261120_redcode_official/redcode_official_logs")],
        "baseline": [Path("runs/exec_20261120_redcode_official/RedCode_baseline_attack_merged.json"),
                     Path("runs/exec_20261120_redcode_official/RedCode_baseline_attack.log")],
        "usage": [Path("runs/cost_20261120")],
        "scoring": [Path("attacks-bench/RedCode/evaluation/RedCode_Exec/evaluation.py")],
    },
}


def copy(src: Path, dst_dir: Path) -> None:
    dst = dst_dir / src.name
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=lambda d, names: [n for n in names if n in SKIP])
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


for bench, parts in PLAN.items():
    for part, sources in parts.items():
        for src in sources:
            if not src.exists():
                print(f"MISSING {bench}/{part}: {src}")
                continue
            copy(src, OUT / bench / part)
    print("ok", bench)
