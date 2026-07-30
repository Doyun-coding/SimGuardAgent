import os
import httpx
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv(usecwd=False))

RISK_THRESHOLD_BLOCK = 0.7

PLAN_MODEL    = os.getenv("PLAN_MODEL",    "openrouter/qwen/qwen3.5-9b")
SHADOW_MODEL  = os.getenv("SHADOW_MODEL",  "openrouter/qwen/qwen3.5-9b")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL",   "openrouter/qwen/qwen3.5-9b")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "openrouter/qwen/qwen3.5-9b")


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=(
            os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("CRTS_API_BASE")
        ),
        http_client=httpx.AsyncClient(verify=False),
        timeout=float(os.getenv("JUDGE_TIMEOUT", "120")),
        # CodeJudge owns its schema/empty-response retry policy. Retrying again
        # inside the SDK multiplies timeout latency and hides infrastructure errors.
        max_retries=0,
    )
