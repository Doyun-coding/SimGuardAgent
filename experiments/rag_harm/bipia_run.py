"""BIPIA (indirect prompt injection in external content) through the AgentDojo agent, harmful attack types only.

Samples are built like BIPIA's own builders (attacks-bench/rag/BIPIA/bipia/data): the external content (email, table,
Stack Overflow answer) goes into BIPIA's system prompt, the question is the user message, and the attack string is
inserted at the end, start or middle of the content (middle = at a sentence boundary chosen with seed 2023; BIPIA
uses nltk's Punkt splitter, here a regex splitter because nltk is not installed).
Attack types kept (harmful outcomes): text - Scams & Fraud, Misinformation & Propaganda, Marketing & Advertising
(email and table tasks); code - all 10 malicious-code types (code task). WebQA and summarization need content that
BIPIA does not ship (license), so they are skipped.
Benign: the same tasks with clean content.

usage (repo root, via run_with_env.py): bipia_run.py OUT_JSON [attack|benign]
env: MEM_DEFENSE-like BIPIA_DEFENSE=none|simguard (default none), BIPIA_WORKERS
"""
import concurrent.futures, hashlib, importlib.util, json, os, pathlib, random, re, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd(); sys.path[:0] = [str(ROOT)]
OUT = pathlib.Path(sys.argv[1]); LABEL = sys.argv[2] if len(sys.argv) > 2 else "attack"
DEFENSE = os.getenv("BIPIA_DEFENSE", "none")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bipia_samples import SYSTEM, USER, samples  # noqa: E402

spec = importlib.util.spec_from_file_location("ad", ROOT / "attacks-bench/agentdojo_adapter/agentdojo_adapter.py")
ad = importlib.util.module_from_spec(spec); sys.modules["ad"] = ad; spec.loader.exec_module(ad)
sys.argv = ["x", "--mode", DEFENSE, "--model", "openai-compatible", "--model-id", "openrouter/qwen/qwen3.5-9b",
            "--judge-model-id", "openrouter/openai/gpt-oss-120b", "--judge-no-think"]
args = ad.parse_args(); ad._configure_endpoint(args)
from agentdojo.functions_runtime import FunctionsRuntime


def run(s):
    system = SYSTEM[s["task"]].format(context=s["context"])
    user = USER[s["task"]].format(**s)
    pipeline = ad.build_pipeline(model=args.model, model_id=args.model_id, defense=DEFENSE, tool_delimiter="tool",
                                 system_message_name=None, system_message=system, tool_output_format=None)
    extra = {"user_request": user, "retrieved_context": s["context"]}
    row = {k: s.get(k) for k in ("id", "task", "attack_name", "attack_str", "position", "question", "ideal")}
    row.update(system=system, user=user, context=s["context"], error=None, response=None)
    try:
        _, _, _, messages, extra = pipeline.query(user + ad.NO_THINK_SUFFIX, FunctionsRuntime([]), ad.EmptyEnv(), [], extra)
        row["response"] = ad._visible_text(messages[-1]["content"])
        verdicts = extra.get("simguard_results")
        if verdicts is not None:
            row["blocked"] = any(v.decision.value == "BLOCK" for v in verdicts)
            row["risk"] = max((float(v.risk_score) for v in verdicts), default=0.0)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return row


if __name__ == "__main__":
    todo = samples(LABEL)
    print(f"{LABEL}: {len(todo)} samples, defense={DEFENSE}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(int(os.getenv("BIPIA_WORKERS", "16"))) as pool:
        rows = list(pool.map(run, todo))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("errors:", sum(1 for r in rows if r["error"]), "written:", OUT)
