"""
SimGuardAgent 내부 모델(Simulator, Judge)을 부르는 방법.

어떤 방법을 쓰든 Simulator와 Judge는 같은 모양으로 부른다:
    resp = await client.chat.completions.create(model=..., messages=..., max_tokens=..., ...)
    resp.choices[0].message.content, resp.usage.prompt_tokens, ...

설정 (.env 또는 환경 변수)
  SIMGUARD_LLM_BACKEND   모든 내부 모델의 기본 방법. 역할별로 SIMULATOR_BACKEND, JUDGE_BACKEND로 덮어쓴다.
    openai        OpenAI 호환 HTTP API (기본). OpenAI, OpenRouter, LiteLLM 프록시, vLLM·SGLang·TGI 서버,
                  Ollama(/v1), Anthropic·Gemini의 OpenAI 호환 주소 등. GPU 서버에 띄운 모델도 이 방법으로 부른다.
    litellm       LiteLLM SDK로 직접 부른다 (프록시 없이 "anthropic/claude-...", "ollama/qwen3" 같은 이름).
    ollama        Ollama 서버(이 PC나 GPU 서버)를 Ollama API로 부른다. thinking을 확실히 끌 수 있다.
                  모델 이름은 `ollama list`의 이름 (예: qwen3.5:9b).
    transformers  이 프로세스에서 Hugging Face 모델을 GPU로 직접 돌린다. 모델 이름은 HF id 또는 로컬 경로.
    vllm          이 프로세스에서 vLLM 엔진으로 GPU에서 직접 돌린다.
  SIMULATOR_MODEL, JUDGE_MODEL   역할별 모델 이름.

  openai·litellm
    {ROLE}_BASE_URL > OPENAI_COMPATIBLE_BASE_URL > OPENAI_BASE_URL > CRTS_API_BASE   주소
    {ROLE}_API_KEY  > OPENAI_COMPATIBLE_API_KEY  > OPENAI_API_KEY                    키
    (ROLE = SIMULATOR 또는 JUDGE. litellm은 주소·키를 주지 않으면 provider별 환경 변수를 쓴다.)

  ollama
    {ROLE}_BASE_URL > OLLAMA_HOST > http://localhost:11434     주소
    SIMGUARD_OLLAMA_NUM_CTX   문맥 길이 (기본 32768). Ollama 기본값은 짧아 긴 대화가 잘리므로 늘린다.
    SIMGUARD_OLLAMA_KEEP_ALIVE  모델을 GPU에 올려 둘 시간 (기본 30m)

  transformers·vllm (GPU)
    SIMGUARD_DEVICE        auto(기본, 여러 GPU에 나눠 올림) | cuda | cuda:1 | cpu      (transformers)
    SIMGUARD_DTYPE         auto(기본) | bfloat16 | float16 | float32
    SIMGUARD_GPU_MEMORY_UTILIZATION  vLLM이 쓸 GPU 메모리 비율 (기본 0.85)
    SIMGUARD_TENSOR_PARALLEL_SIZE    vLLM이 나눠 쓸 GPU 수 (기본 1). 쓸 GPU는 CUDA_VISIBLE_DEVICES로 고른다.
    SIMGUARD_MAX_MODEL_LEN           vLLM 최대 길이 (선택)
    SIMGUARD_TRUST_REMOTE_CODE       1이면 모델 저장소의 코드를 허용 (기본 0)
  같은 이름의 모델은 한 번만 올리고 Simulator와 Judge가 함께 쓴다. GPU 생성은 모델마다 한 번에 하나씩 한다.
  thinking은 끈다(채팅 템플릿에 enable_thinking=False). GPT-OSS처럼 끌 수 없는 모델은 생각 부분을 떼고 답만 쓴다.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
from collections.abc import Mapping
from typing import Any

BACKENDS = ("openai", "litellm", "ollama", "transformers", "vllm")


def backend_for(role: str | None) -> str:
    name = ""
    if role:
        name = os.getenv(f"{role.upper()}_BACKEND", "")
    name = (name or os.getenv("SIMGUARD_LLM_BACKEND", "") or "openai").strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown SimGuardAgent LLM backend {name!r}; choose one of {', '.join(BACKENDS)}")
    return name


def _role_env(role: str | None, name: str, *fallbacks: str) -> str | None:
    keys = ([f"{role.upper()}_{name}"] if role else []) + list(fallbacks)
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def base_url(role: str | None) -> str | None:
    return _role_env(role, "BASE_URL", "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL", "CRTS_API_BASE")


def api_key(role: str | None) -> str | None:
    return _role_env(role, "API_KEY", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY")


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- 응답 모양

def _completion(model: str, text: str, finish_reason: str, prompt_tokens: int, completion_tokens: int,
                reasoning: str = "") -> Any:
    """로컬 생성 결과를 OpenAI ChatCompletion 객체로 만든다."""
    from openai.types.chat import ChatCompletion

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    return ChatCompletion.model_validate({
        "id": "simguard-local",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason if finish_reason in {"stop", "length"} else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_HARMONY_FINAL = re.compile(r"(?:<\|channel\|>final<\|message\|>|assistantfinal)", re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """생성 결과에서 생각 부분과 답을 나눈다 (<think>…</think>, GPT-OSS final 채널)."""
    reasoning = ""
    match = None
    for match in _HARMONY_FINAL.finditer(text):
        pass
    if match is not None:
        reasoning, text = text[:match.start()], text[match.end():]
    if "</think>" in text:
        head, _, text = text.rpartition("</think>")
        reasoning += head.replace("<think>", "")
    text = _THINK.sub("", text).replace("<|return|>", "").replace("<|end|>", "")
    return text.strip(), reasoning.strip()


def _template_kwargs(model: str) -> dict:
    if "gpt-oss" in model.lower():
        return {"reasoning_effort": os.getenv("REASONING_EFFORT", "low")}
    return {"enable_thinking": False}


def _chat_messages(messages: list) -> list[dict]:
    rows = []
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        rows.append({"role": message.get("role", "user"), "content": content or ""})
    return rows


# ---------------------------------------------------------------- GPU 모델

_models: dict[tuple[str, str], Any] = {}
_models_lock = threading.Lock()


def _dtype(torch: Any = None) -> Any:
    name = os.getenv("SIMGUARD_DTYPE", "auto").strip().lower()
    if name == "auto" or torch is None:
        return name
    return getattr(torch, name)


class _TransformersModel:
    def __init__(self, name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust = _flag("SIMGUARD_TRUST_REMOTE_CODE")
        device = os.getenv("SIMGUARD_DEVICE", "auto").strip() or "auto"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
        options: dict[str, Any] = {"torch_dtype": _dtype(torch), "trust_remote_code": trust}
        if device == "auto":
            options["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(name, **options)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.lock = threading.Lock()

    def generate(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> tuple:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            **_template_kwargs(model),
        ).to(self.model.device)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=pad, **sampling)
        prompt_tokens = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finish = "length" if len(new_tokens) >= max_tokens else "stop"
        return text, finish, int(prompt_tokens), int(len(new_tokens))


class _VLLMModel:
    def __init__(self, name: str) -> None:
        from vllm import LLM

        options: dict[str, Any] = {
            "model": name,
            "dtype": _dtype(),
            "gpu_memory_utilization": float(os.getenv("SIMGUARD_GPU_MEMORY_UTILIZATION", "0.85")),
            "tensor_parallel_size": int(os.getenv("SIMGUARD_TENSOR_PARALLEL_SIZE", "1")),
            "trust_remote_code": _flag("SIMGUARD_TRUST_REMOTE_CODE"),
        }
        if os.getenv("SIMGUARD_MAX_MODEL_LEN"):
            options["max_model_len"] = int(os.environ["SIMGUARD_MAX_MODEL_LEN"])
        from simguard.integrations.llm_calls import bypass

        with bypass():          # 내부 모델 호출은 protect()가 판정하지 않는다
            self.llm = LLM(**options)
        self.lock = threading.Lock()

    def generate(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> tuple:
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        with self.lock:
            result = self.llm.chat(messages, params, use_tqdm=False, chat_template_kwargs=_template_kwargs(model))[0]
        output = result.outputs[0]
        return (output.text, output.finish_reason or "stop",
                len(result.prompt_token_ids or []), len(output.token_ids or []))


def _local_model(backend: str, name: str) -> Any:
    key = (backend, name)
    with _models_lock:
        if key not in _models:
            _models[key] = (_TransformersModel if backend == "transformers" else _VLLMModel)(name)
        return _models[key]


class _LocalCompletions:
    def __init__(self, backend: str) -> None:
        self._backend = backend

    async def create(self, *, model: str, messages: list, max_tokens: int | None = None,
                     temperature: float | None = 0, **_: Any) -> Any:
        # response_format, reasoning_effort, extra_body는 API 전용이다. 프롬프트가 JSON을 요구하고
        # 응답은 parse_json_object가 읽는다.
        rows = _chat_messages(messages)
        limit = int(max_tokens or 4096)

        def run() -> Any:
            local = _local_model(self._backend, model)
            text, finish, prompt_tokens, completion_tokens = local.generate(model, rows, limit, float(temperature or 0))
            answer, reasoning = split_reasoning(text)
            return _completion(model, answer, finish, prompt_tokens, completion_tokens, reasoning)

        return await asyncio.to_thread(run)


class _LiteLLMCompletions:
    def __init__(self, role: str | None) -> None:
        self._role = role

    async def create(self, **request: Any) -> Any:
        import litellm

        url, key = base_url(self._role), _role_env(self._role, "API_KEY")
        if url:
            request.setdefault("api_base", url)
        if key:
            request.setdefault("api_key", key)
        request.setdefault("timeout", float(os.getenv("JUDGE_TIMEOUT", "120")))
        request.setdefault("drop_params", True)       # provider가 모르는 인자(reasoning 등)는 버린다
        return await litellm.acompletion(**request)


class _OllamaCompletions:
    def __init__(self, role: str | None) -> None:
        self._host = _role_env(role, "BASE_URL", "OLLAMA_HOST") or "http://localhost:11434"

    async def create(self, *, model: str, messages: list, max_tokens: int | None = None,
                     temperature: float | None = 0, response_format: Any = None, **_: Any) -> Any:
        from ollama import AsyncClient

        # thinking은 끈다. GPT-OSS는 끌 수 없어 가장 낮은 단계로 둔다.
        think: Any = os.getenv("REASONING_EFFORT", "low") if "gpt-oss" in model.lower() else False
        json_mode = isinstance(response_format, Mapping) and response_format.get("type") in {"json_object", "json_schema"}
        # 클라이언트는 이벤트 루프에 묶이므로 호출마다 만든다.
        response = await AsyncClient(host=self._host, timeout=float(os.getenv("JUDGE_TIMEOUT", "120"))).chat(
            model=model,
            messages=_chat_messages(messages),
            think=think,
            format="json" if json_mode else None,
            options={"temperature": float(temperature or 0), "num_predict": int(max_tokens or 4096),
                     "num_ctx": int(os.getenv("SIMGUARD_OLLAMA_NUM_CTX", "32768"))},
            keep_alive=os.getenv("SIMGUARD_OLLAMA_KEEP_ALIVE", "30m"),
        )
        answer, reasoning = split_reasoning(response.message.content or "")
        return _completion(model, answer, "length" if response.done_reason == "length" else "stop",
                           response.prompt_eval_count or 0, response.eval_count or 0,
                           (response.message.thinking or "") + reasoning)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class LLMClient:
    """openai 외 방법의 클라이언트. client.chat.completions.create(...)만 있다."""

    def __init__(self, completions: Any) -> None:
        self.chat = _Chat(completions)


_ssl_context: Any = None
_ssl_lock = threading.Lock()


def _tls_verify() -> Any:
    """httpx의 verify 값. 인증서 묶음(certifi)은 프로세스마다 한 번만 읽어 SSLContext를 다시 쓴다.

    Interceptor는 판정마다 클라이언트를 새로 만든다 (클라이언트가 이벤트 루프에 묶이기 때문). httpx 기본값
    verify=True는 그때마다 certifi 묶음을 다시 읽는데, 프로세스가 많아 CPU가 바쁘면 한 번에 10~25초가 걸렸다.
    SIMGUARD_TLS_VERIFY=0이면 검증하지 않는다.
    """
    global _ssl_context
    if os.getenv("SIMGUARD_TLS_VERIFY", "1").lower() in {"0", "false", "no", "off"}:
        return False
    with _ssl_lock:
        if _ssl_context is None:
            import ssl
            import certifi

            _ssl_context = ssl.create_default_context(cafile=certifi.where())
        return _ssl_context


def make_backend_client(role: str | None) -> Any:
    backend = backend_for(role)
    if backend == "openai":
        import httpx
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=api_key(role) or "EMPTY",       # 키가 필요 없는 로컬 서버(vLLM, Ollama)도 있다
            base_url=base_url(role),
            http_client=httpx.AsyncClient(verify=_tls_verify()),
            timeout=float(os.getenv("JUDGE_TIMEOUT", "120")),
            # Judge가 형식 오류·빈 응답 재시도를 직접 한다. SDK 안에서 또 재시도하면 지연이 곱해진다.
            max_retries=0,
        )
    if backend == "litellm":
        return LLMClient(_LiteLLMCompletions(role))
    if backend == "ollama":
        return LLMClient(_OllamaCompletions(role))
    return LLMClient(_LocalCompletions(backend))
