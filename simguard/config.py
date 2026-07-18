import os
import httpx
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv(usecwd=False))

RISK_THRESHOLD_BLOCK = 0.7

PLAN_MODEL    = os.getenv("PLAN_MODEL",    "qwen/qwen3.5-9b")
SHADOW_MODEL  = os.getenv("SHADOW_MODEL",  "qwen/qwen3.5-9b")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL",   "qwen/qwen3.5-9b")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "qwen/qwen3.5-9b")


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        http_client=httpx.AsyncClient(verify=False),
    )
