"""AgentDojo on the real AutoGen agent with SimGuardAgent.

AutoGenPipeline is an AgentDojo pipeline element whose query() runs an AutoGen AssistantAgent: AgentDojo's own
functions (runtime.functions, acting on AgentDojo's environment) become AutoGen tools, AgentDojo's default system
message is the agent's system message, and the AutoGen run is converted back to AgentDojo chat messages. AgentDojo's
benchmark_suite_with_injections / _without_injections then run the same task pairs (v1.2.2, tool_knowledge attack) and
score utility / security with AgentDojo's checks, writing the same trace logs as the AgentDojo-agent runs, so
experiments/full_runs/score.py scores both the same way.
usage (repo root, .venv-autogen, via launch.py):
  agentdojo_autogen.py --suite banking --label attack|benign --logdir DIR [-ut user_task_0 ...]
env: AUTOGEN_MODE=simguard|none"""
import argparse, asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
common.install(live_reads=False)

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.functions_runtime import FunctionCall
from agentdojo.logging import Logger, OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string
from autogen_agentchat.messages import BaseChatMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel

VERSION = "v1.2.2"


class _Out(BaseModel):
    result: str


class DojoTool(BaseTool):
    """One AgentDojo function as an AutoGen tool; running it calls AgentDojo's runtime on the task environment."""

    def __init__(self, function, runtime, env):
        super().__init__(args_type=function.parameters, return_type=_Out, name=function.name, description=function.description)
        self._runtime, self._env = runtime, env

    async def run(self, args, cancellation_token: CancellationToken):
        result, error = self._runtime.run_function(self._env, self.name, args.model_dump(exclude_none=True))
        return _Out(result=f"Error: {error}" if error else str(result))


class AutoGenPipeline(BasePipelineElement):
    def __init__(self, name: str):
        self.name = name
        self.system = load_system_message(None)

    def query(self, query, runtime, env, messages=(), extra_args={}):
        tools = [DojoTool(f, runtime, env) for f in runtime.functions.values()]
        r = asyncio.run(common.run_agent(query, system=self.system, tools=tools, max_tool_iterations=15))
        out = [{"role": "system", "content": [text_content_block_from_string(self.system)]},
               {"role": "user", "content": [text_content_block_from_string(query)]}]
        calls: dict[str, FunctionCall] = {}
        for m in r["messages"]:
            if isinstance(m, ToolCallRequestEvent):
                fcs = []
                for c in m.content:
                    try:
                        args = json.loads(c.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls[c.id] = FunctionCall(function=c.name, args=args, id=c.id)
                    fcs.append(calls[c.id])
                out.append({"role": "assistant", "content": None, "tool_calls": fcs})
            elif isinstance(m, ToolCallExecutionEvent):
                for res in m.content:
                    out.append({"role": "tool", "content": [text_content_block_from_string(str(res.content))],
                                "tool_call_id": res.call_id, "tool_call": calls.get(res.call_id) or FunctionCall(function=res.name, args={}),
                                "error": str(res.content) if res.is_error else None})
            elif isinstance(m, BaseChatMessage) and m.source != "user":
                # TextMessage, or ToolCallSummaryMessage when AutoGen stops at max_tool_iterations
                out.append({"role": "assistant", "content": [text_content_block_from_string(m.to_text())], "tool_calls": None})
        if r["error"]:
            out.append({"role": "assistant", "content": [text_content_block_from_string(f"Agent error: {r['error']}")], "tool_calls": None})
        Logger.get().log(out)  # AgentPipeline.query does this after each element; the trace logs need the messages
        return query, runtime, env, out, {**extra_args, "autogen": {k: r[k] for k in ("blocked", "judgments", "events", "seconds")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--label", choices=["attack", "benign"], required=True)
    ap.add_argument("--logdir", type=Path, required=True); ap.add_argument("-ut", "--user-task", action="append", default=[])
    args = ap.parse_args()
    suite = get_suite(VERSION, args.suite)
    # the tool_knowledge attack words its injection by model name read from the pipeline name; "local" -> "Local
    # model", the same as the AgentDojo-agent runs (agentdojo_adapter.py build_pipeline)
    pipeline = AutoGenPipeline(f"local-autogen-{common.MODE}")
    args.logdir.mkdir(parents=True, exist_ok=True)
    with OutputLogger(str(args.logdir)):
        if args.label == "benign":
            res = benchmark_suite_without_injections(pipeline, suite, logdir=args.logdir, force_rerun=False,
                                                     user_tasks=args.user_task or None, benchmark_version=VERSION)
        else:
            res = benchmark_suite_with_injections(pipeline, suite, load_attack("tool_knowledge", suite, pipeline), logdir=args.logdir,
                                                  force_rerun=False, user_tasks=args.user_task or None, benchmark_version=VERSION)
    u = res["utility_results"]; s = res.get("security_results", {})
    print(json.dumps({"suite": args.suite, "label": args.label, "cases": len(u), "utility": sum(u.values()), "security": sum(s.values())}))


if __name__ == "__main__":
    main()
