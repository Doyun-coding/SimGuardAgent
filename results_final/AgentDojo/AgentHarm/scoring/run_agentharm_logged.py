"""AgentHarm (harmful, benign, chat) with SimGuardAgent as it is in ./simguard, verdict details in verdicts.jsonl.
usage: run_agentharm_logged.py ROOT val|test_public LOG_DIR [current|old|none]
  old  = previous ToolCallJudge (4 indicators + ToolEmu, max) loaded from tool_call_judge_indicators_toolemu.py
  none = no SimGuardAgent (baseline)"""
import collections, json, os, pathlib, sys, threading, time, warnings
ROOT = pathlib.Path(sys.argv[1]); SPLIT = sys.argv[2]; LOG_DIR = pathlib.Path(sys.argv[3]); MODE = sys.argv[4] if len(sys.argv) > 4 else "current"
assert MODE in ("current", "old", "none", "taskonly"), MODE
# taskonly = simulators/judges see only this task's steps after the latest user message (the earlier input scope)
sys.path[:0] = [str(ROOT)]; os.chdir(ROOT); warnings.filterwarnings("ignore")
LOG_DIR.mkdir(parents=True, exist_ok=True)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
os.environ["JUDGE_NO_THINK"] = "1"

TARGET = os.environ["AGENTDOJO_MODEL_ID"]
GRADER = "openrouter/openai/gpt-4o-mini"
BASE = (os.getenv("OPENAI_COMPATIBLE_BASE_URL") or os.getenv("CRTS_API_BASE") or os.getenv("OPENAI_BASE_URL")).rstrip("/")
KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy")
os.environ.update(GW_BASE_URL=BASE, GW_API_KEY=KEY, OPENAI_COMPATIBLE_BASE_URL=BASE, OPENAI_COMPATIBLE_API_KEY=KEY,
                  SIMULATOR_MODEL=os.getenv("SIMULATOR_MODEL_OVERRIDE") or TARGET, JUDGE_MODEL=os.getenv("JUDGE_MODEL_OVERRIDE") or TARGET)
print(f"target={TARGET} simulator={os.environ['SIMULATOR_MODEL']} judge={os.environ['JUDGE_MODEL']}", flush=True)

import simguard

if MODE == "old":
    import importlib.util, simguard.simulation.backends.tool_call_simulator as TCS
    spec = importlib.util.spec_from_file_location("tool_call_judge_old", pathlib.Path(__file__).with_name("tool_call_judge_indicators_toolemu.py"))
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    TCS.ToolCallJudge = old.ToolCallJudge

if MODE == "taskonly":
    import simguard.interceptor as INT
    _full = INT.render_conversation
    def _task_only(messages, *a, **k):
        users = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
        return _full(messages[users[-1] + 1:] if users else messages, *a, **k)
    INT.render_conversation = _task_only

conv_chars = []
import simguard.interceptor as _INT
_render = _INT.render_conversation
def _measured(messages, *a, **k):
    text = _render(messages, *a, **k); conv_chars.append(len(text)); return text
_INT.render_conversation = _measured

lock = threading.Lock()
verdicts = collections.Counter()
out = (LOG_DIR / (f"verdicts_{os.getenv('AGENTHARM_TASK')}.jsonl" if os.getenv("AGENTHARM_TASK") else "verdicts.jsonl")).open("a", encoding="utf-8")

def on_result(result):
    verdicts.update([result.decision.value])
    rows = []
    for t in result.traces:
        a = t.action
        rows.append({
            "backend": t.backend_used.value, "decision": t.decision.value, "risk": t.risk_score,
            "tool": getattr(a, "tool_name", None),
            "live": (getattr(t.live_result, "metadata", None) or {}).get("live_mode") if t.live_result else None,
            "tool_result": str(getattr(t.live_result, "result", ""))[:600] if t.live_result else None, "args": str(getattr(a, "tool_arguments", ""))[:600],
            "user_request": a.user_request or "",
            "events": [{"type": e.type, "evidence": e.evidence[:600]} for e in t.security_events],
            "axis_detail": t.axis_detail,
        })
    with lock:
        out.write(json.dumps({"decision": result.decision.value, "risk": result.risk_score, "error": result.error,
                              "traces": rows}, ensure_ascii=False, default=str) + "\n")
        out.flush()

# AGENTHARM_TASK=harmful|benign|chat: run one task per process. harmful and benign tools share 75 names with
# different implementations, so each process registers only its own task's tools with protect(tools=...).
ONLY = os.getenv("AGENTHARM_TASK")
import inspect_evals.agentharm.benchmark.harmful_tools as _HT, inspect_evals.agentharm.benchmark.benign_tools as _BT
_mod = {"harmful": _HT, "benign": _BT}.get(ONLY)
REAL_TOOLS = {n: getattr(_mod, n)() for n in _mod.__all__} if _mod else None  # {이름: 도구}
if MODE != "none":
    simguard.protect(target_model=TARGET, on_result=on_result, tools=REAL_TOOLS)
    print(f"task={ONLY} real tools registered: {len(REAL_TOOLS or [])}", flush=True)

from inspect_ai import eval as inspect_eval
from inspect_evals.agentharm import agentharm, agentharm_benign

judge = f"openai-api/gw/{GRADER}"
tasks = {
    "harmful": lambda: agentharm(split=SPLIT, refusal_judge=judge, semantic_judge=judge),
    "benign": lambda: agentharm_benign(split=SPLIT, refusal_judge=judge, semantic_judge=judge),
    "chat": lambda: agentharm(split=SPLIT, refusal_judge=judge, semantic_judge=judge, chat_dataset=True),
}
task_names = [ONLY] if ONLY else list(tasks)
tasks = [tasks[n]() for n in task_names]
started = time.time()
logs = inspect_eval(tasks, model=f"openai-api/gw/{TARGET}", max_connections=int(os.getenv("AH_WORKERS", "20")), max_samples=int(os.getenv("AH_WORKERS", "20")), temperature=0, limit=int(os.getenv("AH_LIMIT")) if os.getenv("AH_LIMIT") else None,
                    max_tokens=2048, extra_body={"reasoning": {"effort": "none", "exclude": True}},
                    log_dir=str(LOG_DIR), display="none")
elapsed = time.time() - started
print(f"\n[{MODE}] SimGuardAgent verdicts: {dict(verdicts)}  ({elapsed:.0f}s)")
summary = {}
for name, log in zip(task_names, logs):
    samples = log.samples or []
    blocked = sum(any(m.role == "assistant" and simguard.BLOCKED_MESSAGE in (m.text or "") for m in s.messages) for s in samples)
    metrics = {k: round(m.value, 4) for sc in (log.results.scores if log.results else [])
               for k, m in sc.metrics.items() if not any(c in k for c in ("_avg_", "category"))}
    summary[name] = {"status": log.status, "samples": len(samples), "samples_with_simguard_block": blocked,
                     "sample_errors": sum(1 for s in samples if s.error), "metrics": metrics}
    print(f"[{name}] samples={len(samples)} blocked={blocked} {metrics}")
conv = {"calls": len(conv_chars), "avg_chars": round(sum(conv_chars) / max(1, len(conv_chars))), "max_chars": max(conv_chars, default=0)}
print(f"conversation passed to simulators/judges: {conv}")
(LOG_DIR / (f"summary_{ONLY}.json" if ONLY else "summary.json")).write_text(json.dumps({"mode": MODE, "split": SPLIT, "verdicts": dict(verdicts), "seconds": round(elapsed),
                                                  "conversation": conv,
                                                  "tasks": summary}, indent=2), encoding="utf-8")
