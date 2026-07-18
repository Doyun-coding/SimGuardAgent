"""
GPT-Researcher 어댑터

서버: http://localhost:8000  (python agents/gpt-researcher/backend/run_server.py)
"""
from __future__ import annotations
import httpx
from .base import AgentAdapter

TIMEOUT = 180


class GPTResearcherAdapter(AgentAdapter):
    name = "gpt-researcher"

    def __init__(self, host: str = "http://localhost:8000"):
        self.url = f"{host}/report/"

    async def query(self, task: str, simguard: bool, client: httpx.AsyncClient = None) -> dict:
        _client = client or httpx.AsyncClient()
        _owned  = client is None
        try:
            r = await _client.post(self.url, json={
                "task":                  task,
                "report_type":           "research_report",
                "report_source":         "web",
                "tone":                  "Objective",
                "repo_name":             "",
                "branch_name":           "",
                "generate_in_background": False,
                "simguard":              simguard,
            }, timeout=TIMEOUT)
            return r.json()
        except httpx.TimeoutException:
            return {"error": "timeout", "report": None, "simguard_decision": None, "risk_score": None}
        except Exception as e:
            return {"error": str(e), "report": None, "simguard_decision": None, "risk_score": None}
        finally:
            if _owned:
                await _client.aclose()
