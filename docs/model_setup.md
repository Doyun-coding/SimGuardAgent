# 모델 연결 설정

SimGuardAgent가 다루는 모델은 두 가지다. 둘은 따로 설정한다.

| | 무엇 | 설정 |
|---|---|---|
| 타겟 모델 | 보호할 에이전트가 부르는 모델 | `protect(providers=...)` 또는 `SIMGUARD_PROVIDERS` |
| 내부 모델 | Simulator, Judge | `SIMGUARD_LLM_BACKEND` (역할별 `SIMULATOR_BACKEND`, `JUDGE_BACKEND`) |

---

## 타겟 모델: 어떤 호출을 가로챌지

```python
import simguard
simguard.protect()                                  # 설치된 방법을 모두 (기본 "auto")
simguard.protect(providers=["anthropic"])           # 쓰는 방법만
```

`.env`에 `SIMGUARD_PROVIDERS=openai,ollama`처럼 써도 된다 (`protect(providers=...)`가 우선).

| 이름 | 가로채는 호출 | 예 |
|---|---|---|
| `openai` | OpenAI SDK `chat.completions.create` | OpenAI, OpenRouter, LiteLLM 프록시, GPU 서버의 vLLM·SGLang·TGI, Ollama `/v1` |
| `litellm` | `litellm.completion`, `acompletion` | LiteLLM SDK |
| `anthropic` | Anthropic SDK `messages.create` | Claude |
| `ollama` | `ollama.chat`, `Client.chat`, `AsyncClient.chat` | Ollama Python 클라이언트 |
| `vllm` | `vllm.LLM.chat` | 이 프로세스의 GPU에 올린 모델 |
| `transformers` | text-generation `pipeline(...)` 호출 | 이 프로세스의 GPU에 올린 모델 |

- 아직 import되지 않은 라이브러리는 import되는 순간 연결한다. 그래서 `protect()`가 vLLM·transformers를 미리 불러오지 않는다.
- 로컬 모델이 도구 호출을 글로 내면 도구 호출로 읽어서 판정한다. 읽는 형식은 `<tool_call>{...}</tool_call>`, `<|python_tag|>`, `<function=...>`, `[TOOL_CALLS]`, 그리고 출력 전체가 `{"name", "arguments"}` JSON인 경우다 (`simguard.parse_text_tool_calls`).
- 목록에 없는 방법(다른 SDK, 직접 만든 `generate` 함수)은 `LLMCall`로 알려 준다.

```python
simguard.protect(llm_calls=[simguard.LLMCall(
    owner=MyModel, name="generate",
    request=lambda args, kwargs: (kwargs["messages"], kwargs.get("tools")),       # OpenAI 형식 (messages, tools)
    outputs=lambda response, args, kwargs: [simguard.integrations.llm_providers.text_output(response.text)],
    block=lambda response, index: setattr(response, "text", simguard.BLOCKED_MESSAGE),
)])
```

- 지원하지 않는 호출: OpenAI Responses API, Anthropic `messages.stream()` 도우미, `model.generate()`를 토큰 단위로 직접 부르는 코드. 이런 호출은 `LLMCall`로 알려 준다.

---

## 내부 모델: Simulator와 Judge를 어떻게 부를지

```ini
# .env
SIMGUARD_LLM_BACKEND=openai        # 모든 내부 모델의 기본
JUDGE_BACKEND=vllm                 # (선택) Judge만 다르게
SIMULATOR_MODEL=...
JUDGE_MODEL=...
```

| 값 | 부르는 방법 | 모델 이름 |
|---|---|---|
| `openai` (기본) | OpenAI 호환 HTTP API. GPU 서버에 띄운 vLLM·SGLang·TGI·Ollama도 이 방법으로 부른다 | 서버가 받는 이름 |
| `litellm` | LiteLLM SDK로 직접 (프록시 없이) | `anthropic/claude-...`, `ollama/qwen3` 등 |
| `ollama` | Ollama 서버(이 PC 또는 GPU 서버)를 Ollama API로. thinking을 확실히 끈다 | `ollama list`의 이름 (`qwen3.5:9b`) |
| `transformers` | 이 프로세스에서 Hugging Face 모델을 GPU로 돌린다 | HF id 또는 로컬 경로 |
| `vllm` | 이 프로세스에서 vLLM 엔진으로 GPU에서 돌린다 | HF id 또는 로컬 경로 |

### API (`openai`, `litellm`)

| 설정 | 우선순위 |
|---|---|
| 주소 | `{ROLE}_BASE_URL` > `OPENAI_COMPATIBLE_BASE_URL` > `OPENAI_BASE_URL` > `CRTS_API_BASE` |
| 키 | `{ROLE}_API_KEY` > `OPENAI_COMPATIBLE_API_KEY` > `OPENAI_API_KEY` |

`ROLE`은 `SIMULATOR` 또는 `JUDGE`다. 예를 들어 Judge만 GPU 서버로 보내려면 `JUDGE_BASE_URL=http://gpu-server:8000/v1`을 쓴다. 키가 필요 없는 로컬 서버는 키를 비워 둬도 된다.

### Ollama (`ollama`)

```ini
SIMGUARD_LLM_BACKEND=ollama
SIMULATOR_MODEL=qwen3.5:9b
JUDGE_MODEL=qwen3.5:9b
```

| 설정 | 기본 | 뜻 |
|---|---|---|
| 주소 | `{ROLE}_BASE_URL` > `OLLAMA_HOST` > `http://localhost:11434` | 다른 PC의 GPU 서버도 된다 |
| `SIMGUARD_OLLAMA_NUM_CTX` | `32768` | 문맥 길이. Ollama 기본값은 짧아서 긴 대화가 잘린다 |
| `SIMGUARD_OLLAMA_KEEP_ALIVE` | `30m` | 모델을 GPU에 올려 둘 시간 |

동시에 처리할 요청 수는 Ollama 서버의 `OLLAMA_NUM_PARALLEL`로 정한다.

### GPU 직접 (`transformers`, `vllm`)

| 설정 | 기본 | 뜻 |
|---|---|---|
| `SIMGUARD_DEVICE` | `auto` | transformers: `auto`(여러 GPU에 나눠 올림), `cuda`, `cuda:1`, `cpu` |
| `SIMGUARD_DTYPE` | `auto` | `bfloat16`, `float16`, `float32` |
| `SIMGUARD_GPU_MEMORY_UTILIZATION` | `0.85` | vLLM이 쓸 GPU 메모리 비율 |
| `SIMGUARD_TENSOR_PARALLEL_SIZE` | `1` | vLLM이 나눠 쓸 GPU 수. 쓸 GPU는 `CUDA_VISIBLE_DEVICES`로 고른다 |
| `SIMGUARD_MAX_MODEL_LEN` | 모델 기본값 | vLLM 최대 길이 |
| `SIMGUARD_TRUST_REMOTE_CODE` | `0` | 모델 저장소의 코드 실행 허용 |

- 같은 이름의 모델은 한 번만 올리고 Simulator와 Judge가 함께 쓴다. 생성은 모델마다 한 번에 하나씩 한다.
- thinking은 끈다 (채팅 템플릿에 `enable_thinking=False`). GPT-OSS처럼 끌 수 없는 모델은 생각 부분을 떼고 답만 판정에 쓴다.
- `response_format`, `reasoning_effort` 같은 API 전용 인자는 쓰지 않는다. 프롬프트가 JSON을 요구하고, 응답에서 JSON을 읽는다.
- `transformers`, `vllm`, `torch`는 requirements에 없다. 쓰는 환경에만 설치한다.
