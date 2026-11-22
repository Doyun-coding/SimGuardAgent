"""Usage logging hook, loaded automatically by every Python process that has this folder on PYTHONPATH
(launch.py puts it there, so subprocesses such as BIRD's per-case agents are logged too). See launch.py."""
import json, os, threading, time, traceback
from pathlib import Path

GRADER_FUNCS = {"goal_match"}  # benchmark graders that call the target model (Sleeper's goal-match grader)


def _role() -> str:
    stack = traceback.extract_stack(limit=60)
    if any(frame.name in GRADER_FUNCS for frame in stack):
        return "grader"
    for frame in reversed(stack):
        p = frame.filename.replace("\\", "/")
        if "/simguard/simulation/judges/" in p:
            return "judge"
        if "/simguard/integrations/" in p:  # simguard.protect wraps the target's own LLM calls: keep looking outward
            continue
        if "/simguard/" in p and "/experiments/" not in p:
            return "simulator"
    return "other"


def _install(prefix: str) -> None:
    import openai.resources.chat.completions as chat_mod
    path = Path(f"{prefix}.{os.getpid()}.jsonl"); path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def record(model, role, started, resp, error):
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        row = {"t": round(started, 3), "sec": round(time.time() - started, 3), "model": str(model), "role": role,
               "error": error, "prompt_tokens": getattr(usage, "prompt_tokens", None),
               "completion_tokens": getattr(usage, "completion_tokens", None),
               "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
               "cost": (getattr(usage, "model_extra", None) or {}).get("cost") if usage else None}
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    sync_create, async_create = chat_mod.Completions.create, chat_mod.AsyncCompletions.create

    def create(self, *a, **kw):
        role, started = _role(), time.time()
        try:
            resp = sync_create(self, *a, **kw)
        except Exception as exc:
            record(kw.get("model"), role, started, None, type(exc).__name__); raise
        record(kw.get("model"), role, started, resp, None)
        return resp

    async def acreate(self, *a, **kw):
        role, started = _role(), time.time()
        try:
            resp = await async_create(self, *a, **kw)
        except Exception as exc:
            record(kw.get("model"), role, started, None, type(exc).__name__); raise
        record(kw.get("model"), role, started, resp, None)
        return resp

    chat_mod.Completions.create, chat_mod.AsyncCompletions.create = create, acreate



if os.getenv("USAGE_LOG"):
    try:
        _install(os.environ["USAGE_LOG"])
    except ImportError:
        pass  # a process without openai
