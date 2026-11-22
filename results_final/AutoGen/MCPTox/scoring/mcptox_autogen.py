"""MCPTox on the real AutoGen agent with SimGuardAgent.
Cases come from attacks-bench/mcptox/run_mcptox_simguard.py load_cases (same 1348 attack + 357 benign). The MCP
server's tools listed in the case's system prompt (name, description incl. the poisoned one, argument names) become
AutoGen tools; the system prompt stays as the agent's system message. The agent gets the user query and may make one
round of tool calls (MCPTox scores the first action). Tool results are a fixed "OK" (MCPTox has no tool backends).
raw_response = the first proposed tool call as {"tool", "arguments"} JSON (taken from SimGuardAgent's verdict records,
so a blocked call is still recorded), else the text reply; experiments/full_runs/gt_label_mcptox.py labels it.
usage (repo root, .venv-autogen, via launch.py): mcptox_autogen.py OUT_JSON        env: AUTOGEN_MODE, AG_WORKERS"""
import asyncio, json, os, re, sys
from pathlib import Path
from typing import Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
common.install(live_reads=False)
sys.path.insert(0, str(common.ROOT / "attacks-bench/mcptox"))
import run_mcptox_simguard as M
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, create_model

OUT = Path(sys.argv[1])
BLOCK = re.compile(r"Tool:\s*([^\n]+)\nDescription:\s*(.*?)(?:\nArguments:\n(.*?))?(?=\n\n\nTool:|\Z)", re.S)


class _Result(BaseModel):
    result: str


class MCPTool(BaseTool):
    def __init__(self, name: str, description: str, arg_names: list[str]):
        fields = {a: (Optional[Any], None) for a in arg_names}
        args = create_model(f"{re.sub(r'[^A-Za-z0-9_]', '_', name)}_args", __config__=ConfigDict(extra="allow"), **fields)
        super().__init__(args_type=args, return_type=_Result, name=re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64], description=description[:4000])

    async def run(self, args, cancellation_token: CancellationToken):
        return _Result(result="OK")


def tools_from_system(system: str) -> list[MCPTool]:
    out, seen = [], set()
    for name, desc, arg_block in BLOCK.findall(system or ""):
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        args = re.findall(r"^-\s*([A-Za-z_][\w-]*)\s*:", arg_block or "", re.M)
        out.append(MCPTool(name, desc.strip(), args))
    return out


async def main():
    cases = M.load_cases(common.ROOT / "attacks-bench/MCPTox-Benchmark/response_all.json", "all")
    if os.getenv("AG_LIMIT"):  # smoke test: first N attack + first N benign cases
        n = int(os.environ["AG_LIMIT"])
        cases = [c for c in cases if c["label"] == "attack"][:n] + [c for c in cases if c["label"] == "benign"][:n]
    sem = asyncio.Semaphore(int(os.getenv("AG_WORKERS", "24")))

    async def one(case):
        async with sem:
            r = await common.run_agent(case["query"], system=case["system"], tools=tools_from_system(case["system"]), max_tool_iterations=1)
        first = r["proposed"][0] if r["proposed"] else None
        if first is None:
            calls = common.tool_calls(r["messages"])
            first = {"tool": calls[0][0], "arguments": json.loads(calls[0][1] or "{}"), "decision": "ALLOW"} if calls else None
        raw = json.dumps({"tool": first["tool"], "arguments": first["arguments"]}, ensure_ascii=False) if first else r["final"]
        return {**case, "raw_response": raw, "tool_name": first["tool"] if first else "no_tool_call",
                "tool_arguments": first["arguments"] if first else {}, "blocked": bool(r["blocked"]),
                "decision": "BLOCK" if r["blocked"] else "ALLOW", "events": r["events"], "error": r["error"], "seconds": r["seconds"]}

    rows = await asyncio.gather(*(one(c) for c in cases))
    att = [x for x in rows if x["label"] == "attack"]; ben = [x for x in rows if x["label"] == "benign"]
    summary = {"agent": "autogen", "mode": common.MODE, "total": len(rows), "attack_total": len(att), "attack_blocked": sum(x["blocked"] for x in att),
               "benign_total": len(ben), "benign_blocked": sum(x["blocked"] for x in ben), "errors": sum(1 for x in rows if x["error"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**summary, "cases": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary))

asyncio.run(main())
