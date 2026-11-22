#!/usr/bin/env bash
# All SimGuardAgent benchmarks with the current simguard/, through the AgentDojo agent where the benchmark allows.
# target = simulator = qwen/qwen3.5-9b, judge = openai/gpt-oss-120b (experiments/full_runs/launch.py).
# Every LLM call is logged per benchmark (usage.<pid>.jsonl); each folder gets started.txt / finished.txt.
# usage (repo root): bash experiments/full_runs/full_run.sh RUN_DIR
R=$1; mkdir -p "$R"; export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
L=experiments/full_runs/launch.py; PY=.venv/Scripts/python.exe
stamp() { date "+%Y-%m-%d %H:%M:%S" > "$1"; }
J="--model-id openrouter/qwen/qwen3.5-9b --judge-model-id openrouter/openai/gpt-oss-120b --judge-no-think"

agentdojo() {
  local D=$R/agentdojo; mkdir -p $D/benign; stamp $D/started.txt
  for spec in workspace:40 banking:16 slack:21 travel:20; do s=${spec%%:*}; n=${spec##*:}
    for k in $(seq 0 9); do uts=(); for i in $(seq $k 10 $((n-1))); do uts+=(-ut "user_task_$i"); done
      [ ${#uts[@]} -eq 0 ] && continue
      USAGE_LOG=$D/usage $PY $L attacks-bench/agentdojo_adapter/agentdojo_adapter.py --benchmark $s --mode simguard --label all $J \
        --logdir $D/logs_$s --results $D/${s}_part$k.json "${uts[@]}" > $D/run_${s}_part$k.out 2>&1 &
    done
    for k in 0 1 2 3; do uts=(); for i in $(seq $k 4 $((n-1))); do uts+=(-ut "user_task_$i"); done
      USAGE_LOG=$D/usage $PY $L attacks-bench/agentdojo_adapter/agentdojo_adapter.py --benchmark $s --mode simguard --label benign $J \
        --logdir $D/benign/logs_$s --results $D/benign/${s}_part$k.json "${uts[@]}" > $D/benign/run_${s}_part$k.out 2>&1 &
    done
  done
  wait; stamp $D/finished.txt
}

agentharm() {
  local D=$R/agentharm; mkdir -p $D; stamp $D/started.txt
  for t in harmful benign chat; do
    USAGE_LOG=$D/usage AH_WORKERS=20 AGENTHARM_TASK=$t .venv-agentharm/Scripts/python.exe $L experiments/full_runs/run_agentharm_logged.py "$PWD" test_public $D > $D/run_$t.out 2>&1 &
  done
  wait; stamp $D/finished.txt
}

mcptox() {
  local D=$R/mcptox; mkdir -p $D; stamp $D/started.txt
  USAGE_LOG=$D/usage $PY $L attacks-bench/mcptox/run_mcptox_simguard.py --data attacks-bench/MCPTox-Benchmark/response_all.json \
    --mode simguard --label all --model openrouter/qwen/qwen3.5-9b --workers 40 --results $D/mcptox_simguard.json > $D/simguard.log 2>&1
  stamp $D/finished.txt
}

memory() {
  local D=$R/memory; mkdir -p $D; stamp $D/started.txt
  for at in universal_v2_optimized_without_markers universal_v2_optimized_with_markers evolved_v2 attack_from_literature mem0_external_prompt_leak_without_untrusted_markers; do
    for app in none untrusted_content_markers; do
      USAGE_LOG=$D/usage MEM_APP_DEFENSE=$([ $app = none ] && echo "" || echo $app) MEM_DEFENSE=simguard MEM_WORKERS=10 MEM_MODE=attack MEM_ATTACK=$at \
        $PY $L experiments/memory_rag_judge/run_memory.py . $D/attack_${at}_app-$app.json 25 > $D/attack_${at}_app-$app.log 2>&1 &
    done
  done
  USAGE_LOG=$D/usage MEM_DEFENSE=simguard MEM_WORKERS=10 MEM_MODE=benign $PY $L experiments/memory_rag_judge/run_memory.py . $D/benign.json 70 > $D/benign.log 2>&1 &
  wait; stamp $D/finished.txt
}

bipia() {
  local D=$R/bipia; mkdir -p $D; stamp $D/started.txt
  for label in attack benign; do
    USAGE_LOG=$D/usage BIPIA_DEFENSE=simguard BIPIA_WORKERS=16 $PY $L experiments/rag_harm/bipia_run.py $D/bipia_simguard_$label.json $label > $D/$label.log 2>&1 &
  done
  wait; stamp $D/finished.txt
}

bird() {
  local D=$R/bird; mkdir -p $D; stamp $D/started.txt
  USAGE_LOG=$D/usage $PY $L attacks-bench/database/run_bird_agent_simguard.py --model openrouter/qwen/qwen3.5-9b --workers 16 \
    --results $D/bird_simguard.json > $D/simguard.log 2>&1
  stamp $D/finished.txt
}

stamp $R/started.txt
agentdojo & agentharm & mcptox & memory & bipia & bird &
wait
stamp $R/finished.txt
echo "ALL DONE"
