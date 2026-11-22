"""AgentHarm on the real AutoGen agent with SimGuardAgent (harmful, benign, chat; test_public).

inspect runs AgentHarm as usual (dataset, setup_tools_from_metadata, AgentHarm's default system prompt and prompt
template, and its combined scorer: grading functions + gpt-4o-mini refusal / semantic judges). Only the generate
loop of AgentHarm's default agent is replaced: an AutoGen AssistantAgent gets the same system and user messages and
the sample's inspect tools (wrapped as AutoGen tools), and its run is converted back into inspect messages for the
scorer. simguard.protect intercepts AutoGen's model calls, with the same real tools registered for live reads as
experiments/full_runs/run_agentharm_logged.py. AutoGen comes from the repo source (agents/autogen/python/packages).
usage (repo root, .venv-agentharm, via launch.py): agentharm_autogen.py test_public LOG_DIR
env: AGENTHARM_TASK=harmful|benign|chat, AH_WORKERS, AUTOGEN_MODE=simguard|none"""
import collections, json, os, pathlib, sys, threading, time, warnings
import importlib.metadata as md
ROOT = pathlib.Path.cwd(); SPLIT = sys.argv[1]; LOG_DIR = pathlib.Path(sys.argv[2]); LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT)] + [str(ROOT / f"agents/autogen/python/packages/{p}/src") for p in ("autogen-core", "autogen-agentchat", "autogen-ext")]
_version = md.version
md.version = lambda name: "0.7.5" if name.replace("-", "_").startswith("autogen") else _version(name)  # source checkout: no metadata
warnings.filterwarnings("ignore")
MODE = os.getenv("AUTOGEN_MODE", "simguard"); ONLY = os.environ["AGENTHARM_TASK"]
TARGET = os.environ["AGENTDOJO_MODEL_ID"]; GRADER = "openrouter/openai/gpt-4o-mini"
BASE = os.environ["OPENAI_COMPATIBLE_BASE_URL"]; KEY = os.environ["OPENAI_COMPATIBLE_API_KEY"]
os.environ.update(GW_BASE_URL=BASE, GW_API_KEY=KEY)

import simguard
lock, verdicts = threading.Lock(), collections.Counter()
out = (LOG_DIR / f"verdicts_{ONLY}.jsonl").open("a", encoding="utf-8")


def on_result(result):
    verdicts.update([result.decision.value])
    rows = [{"backend": t.backend_used.value, "decision": t.decision.value, "risk": t.risk_score, "tool": t.action.tool_name,
             "args": str(t.action.tool_arguments)[:600], "events": [{"type": e.type, "evidence": e.evidence[:600]} for e in t.security_events]}
            for t in result.traces]
    with lock:
        out.write(json.dumps({"decision": result.decision.value, "risk": result.risk_score, "error": result.error, "traces": rows}, default=str) + "\n"); out.flush()


import inspect_evals.agentharm.benchmark.harmful_tools as _HT, inspect_evals.agentharm.benchmark.benign_tools as _BT
_mod = {"harmful": _HT, "benign": _BT}.get(ONLY)
REAL_TOOLS = {n: getattr(_mod, n)() for n in _mod.__all__} if _mod else None
if MODE == "simguard":
    simguard.protect(target_model=TARGET, on_result=on_result, tools=REAL_TOOLS, providers=["openai"])

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ModelOutput
from inspect_ai.solver import chain, prompt_template, solver
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool_def import ToolDef
from inspect_evals.agentharm import agentharm, agentharm_benign
from inspect_evals.agentharm.agents.default_agent import set_tool_choice, system_prompt_based_on_model
from inspect_evals.agentharm.prompts import get_jailbreak_template
from inspect_evals.agentharm.utils import get_benchmark_tools, setup_tools_from_metadata
from pydantic import BaseModel, ConfigDict


class _Any(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Out(BaseModel):
    result: str


class InspectTool(BaseTool):
    """An inspect tool as an AutoGen tool; the schema is the inspect tool's own JSON schema."""

    def __init__(self, tool):
        self._tool, self._def = tool, ToolDef(tool)
        super().__init__(args_type=_Any, return_type=_Out, name=self._def.name, description=self._def.description)

    @property
    def schema(self):
        return {"name": self._def.name, "description": self._def.description,
                "parameters": self._def.parameters.model_dump(exclude_none=True)}

    async def run(self, args, cancellation_token: CancellationToken):
        try:
            return _Out(result=str(await self._tool(**args.model_dump())))
        except Exception as exc:
            return _Out(result=f"Error: {type(exc).__name__}: {exc}")


@solver
def autogen_generate():
    async def solve(state, generate):
        system = next((m.text for m in state.messages if m.role == "system"), None)
        user = next(m.text for m in state.messages if m.role == "user")
        client = OpenAIChatCompletionClient(model=TARGET, base_url=BASE, api_key=KEY, temperature=0,
                                            model_info={"vision": False, "function_calling": True, "json_output": True,
                                                        "family": "unknown", "structured_output": False},
                                            extra_body={"reasoning": {"effort": "none", "exclude": True}})
        try:
            agent = AssistantAgent("assistant", model_client=client, system_message=system, tools=[InspectTool(t) for t in state.tools],
                                   max_tool_iterations=9)  # AgentHarm's Task allows 20 messages
            result = await agent.run(task=user)
        finally:
            await client.close()
        calls, text = {}, ""
        for m in result.messages:
            if isinstance(m, ToolCallRequestEvent):
                tcs = []
                for c in m.content:
                    try:
                        args = json.loads(c.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls[c.id] = c.name
                    tcs.append(ToolCall(id=c.id, function=c.name, arguments=args))
                state.messages.append(ChatMessageAssistant(content="", tool_calls=tcs, model=TARGET))
            elif isinstance(m, ToolCallExecutionEvent):
                for res in m.content:
                    state.messages.append(ChatMessageTool(content=str(res.content), tool_call_id=res.call_id, function=calls.get(res.call_id, res.name)))
            elif isinstance(m, TextMessage) and m.source != "user":
                text = m.content
                state.messages.append(ChatMessageAssistant(content=text, model=TARGET))
        state.output = ModelOutput.from_content(model=TARGET, content=text)
        return state
    return solve


judge = f"openai-api/gw/{GRADER}"
make = {"harmful": lambda: agentharm(split=SPLIT, refusal_judge=judge, semantic_judge=judge),
        "benign": lambda: agentharm_benign(split=SPLIT, refusal_judge=judge, semantic_judge=judge),
        "chat": lambda: agentharm(split=SPLIT, refusal_judge=judge, semantic_judge=judge, chat_dataset=True)}
task = make[ONLY]()
tools = get_benchmark_tools(ONLY) if ONLY != "chat" else []
# AgentHarm's default agent (default_agent.py) with its generate(tool_calls="loop") replaced by the AutoGen agent
plan = chain(setup_tools_from_metadata(ONLY, tools, 0), system_prompt_based_on_model(None, "standard"),
             prompt_template(get_jailbreak_template("empty")), set_tool_choice("auto"), autogen_generate())
started = time.time()
logs = inspect_eval([task], solver=plan, model=f"openai-api/gw/{TARGET}", max_connections=int(os.getenv("AH_WORKERS", "20")),
                    max_samples=int(os.getenv("AH_WORKERS", "20")), limit=int(os.getenv("AH_LIMIT")) if os.getenv("AH_LIMIT") else None,
                    log_dir=str(LOG_DIR), display="none")
elapsed = time.time() - started
log = logs[0]; samples = log.samples or []
blocked = sum(any(m.role == "assistant" and simguard.BLOCKED_MESSAGE in (m.text or "") for m in s.messages) for s in samples)
metrics = {k: round(m.value, 4) for sc in (log.results.scores if log.results else []) for k, m in sc.metrics.items()}
summary = {"agent": "autogen", "mode": MODE, "split": SPLIT, "verdicts": dict(verdicts), "seconds": round(elapsed),
           "tasks": {ONLY: {"status": log.status, "samples": len(samples), "samples_with_simguard_block": blocked,
                            "sample_errors": sum(1 for s in samples if s.error), "metrics": metrics}}}
(LOG_DIR / f"summary_{ONLY}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[{ONLY}] samples={len(samples)} blocked={blocked} {metrics}")
