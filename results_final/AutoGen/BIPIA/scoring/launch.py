"""Run a benchmark script with the lab endpoint and models set, logging every chat-completion call.

Environment: loads .env (never printed) and sets target = simulator = qwen/qwen3.5-9b, judge = openai/gpt-oss-120b,
like the scratchpad run_with_env.py. Then usage_site/sitecustomize.py (on PYTHONPATH, so child processes too) patches the OpenAI chat-completions classes (sync and async) so each call
appends one line to USAGE_LOG.<pid>.jsonl: start time, latency, model, prompt/completion/reasoning tokens, cost, and
the caller's role read from the call stack:
  judge      simguard/simulation/judges/*
  simulator  any other simguard/* frame (simulators, plan extraction)
  grader     a benchmark grader function (GRADER_FUNCS), e.g. Sleeper's goal-match grader
  other      everything else: the target agent, and graders on another model (told apart by model name)
Requests and responses are not changed.
usage: launch.py SCRIPT [ARGS...]       env: USAGE_LOG (path prefix; unset = no logging)
"""
import os, runpy, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path.cwd()
load_dotenv(ROOT / ".env", override=True)
base = (os.getenv("OPENAI_COMPATIBLE_BASE_URL") or os.getenv("CRTS_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
TARGET = os.getenv("RUN_TARGET_MODEL", "openrouter/qwen/qwen3.5-9b")
os.environ.update(
    OPENAI_BASE_URL=base, OPENAI_API_KEY=key, OPENAI_COMPATIBLE_BASE_URL=base, OPENAI_COMPATIBLE_API_KEY=key,
    SIMULATOR_MODEL=os.getenv("RUN_SIMULATOR_MODEL", TARGET),
    JUDGE_MODEL=os.getenv("RUN_JUDGE_MODEL", "openrouter/openai/gpt-oss-120b"),
    PLAN_MODEL=TARGET, AGENTDOJO_MODEL_ID=TARGET, JUDGE_NO_THINK="1",
    JUDGE_MODEL_OVERRIDE=os.getenv("RUN_JUDGE_MODEL", "openrouter/openai/gpt-oss-120b"),
)


SITE = Path(__file__).resolve().parent / "usage_site"
os.environ["PYTHONPATH"] = os.pathsep.join([str(SITE), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
if "usage_site" not in str(getattr(sys.modules.get("sitecustomize"), "__file__", "")):
    runpy.run_path(str(SITE / "sitecustomize.py"))  # this process started before PYTHONPATH was set

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
