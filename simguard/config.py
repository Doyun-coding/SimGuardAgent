import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=False))

RISK_THRESHOLD_BLOCK = 0.7

SIMULATOR_MODEL  = os.getenv("SIMULATOR_MODEL",  "openrouter/qwen/qwen3.5-9b")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL",   "openrouter/qwen/qwen3.5-9b")


def make_client(role: str | None = None):
    """내부 모델 클라이언트. role("simulator" | "judge")별로 부르는 방법(API, LiteLLM, GPU)을 고른다.

    설정은 simguard.llm_backends 참고. client.chat.completions.create(...)로 부른다.
    """
    from simguard import llm_backends

    return llm_backends.make_backend_client(role)
