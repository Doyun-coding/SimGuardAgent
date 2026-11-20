"""
protect()가 가로채는 타겟 모델 호출 방법.

  openai        OpenAI SDK chat.completions.create. OpenAI 호환 주소면 모두 여기로 온다
                (OpenAI, OpenRouter, LiteLLM 프록시, GPU 서버의 vLLM·SGLang·TGI, Ollama /v1 등).
  litellm       litellm.completion / acompletion
  anthropic     Anthropic SDK messages.create
  ollama        ollama.chat, ollama.Client.chat, ollama.AsyncClient.chat
  vllm          vllm.LLM.chat (이 프로세스의 GPU에서 돌리는 모델)
  transformers  transformers text-generation pipeline 호출 (이 프로세스의 GPU에서 돌리는 모델)

이 밖의 방법(다른 SDK, 직접 만든 generate 함수 등)은 LLMCall로 알려 준다:

  simguard.protect(llm_calls=[simguard.LLMCall(
      owner=MyModel, name="generate",
      request=lambda args, kwargs: (kwargs["messages"], kwargs.get("tools")),   # OpenAI 형식 (messages, tools)
      outputs=lambda response, args, kwargs: [{"role": "assistant", "content": response.text}],
      block=lambda response, index: setattr(response, "text", simguard.BLOCKED_MESSAGE),
  )])

모델이 도구 호출을 글로 내는 경우(로컬 GPU 모델)는 parse_text_tool_calls로 도구 호출을 읽는다.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from simguard.interceptor import BLOCKED_MESSAGE


# ---------------------------------------------------------------- 공통

def plain(value: Any) -> Any:
    """SDK 객체를 판정에 넘길 dict와 list로 바꾼다."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def get(value: Any, name: str) -> Any:
    """SDK 객체와 dict를 같은 방식으로 읽는다."""
    if isinstance(value, Mapping):
        return value.get(name)
    found = getattr(value, name, None)
    if found is None:
        found = (getattr(value, "model_extra", None) or {}).get(name)
    return found


def put(value: Any, name: str, item: Any) -> None:
    if isinstance(value, dict):
        value[name] = item
    else:
        setattr(value, name, item)


def argument(args: tuple, kwargs: dict, name: str, position: int | None) -> Any:
    if name in kwargs:
        return kwargs[name]
    if position is not None and len(args) > position:
        return args[position]
    return None


def _kw_model(args: tuple, kwargs: dict) -> Any:
    return kwargs.get("model")


def _kw_stream(args: tuple, kwargs: dict) -> bool:
    return bool(kwargs.get("stream"))


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value if value is not None else {}, ensure_ascii=False)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                parts.append(part.get("text") or (f"[{part.get('type')}]" if part.get("type") else ""))
        return "\n".join(part for part in parts if part)
    return str(content)


@dataclass
class LLMCall:
    """protect()가 판정할 LLM 호출 하나.

    owner, name: 가로챌 함수가 있는 곳(모듈, 클래스, 객체)과 이름.
    request(args, kwargs) -> (messages, tools): 모델 입력. OpenAI 채팅 형식으로 바꿔 준다.
    outputs(response, args, kwargs) -> list: 선택지마다 출력 메시지 {"content", "tool_calls"}.
        선택지마다 입력이 다르면(배치 호출) (messages, output) 쌍으로 준다.
    block(response, index): index번째 선택지를 차단 문구로 바꾼다. 새 응답을 돌려줘도 된다.
    is_async: 함수가 async인지. None이면 함수를 보고 정한다.
    model(args, kwargs): 모델 이름. protect(target_model=...)와 비교한다.
    stream(args, kwargs) -> bool: 스트림 응답인지.
    merge_stream(chunks) -> {index: output}: 스트림 조각을 출력으로 합친다.
    replace_stream(chunks, blocked) -> chunks: BLOCK된 선택지를 차단 문구로 바꾼 조각.
        없으면 BLOCK일 때 SimGuardBlocked를 일으킨다.
    bypass_inner: 이 호출 안에서 일어나는 다른 LLM 호출은 판정하지 않는다 (같은 출력을 두 번 판정하지 않게).
    """

    owner: Any
    name: str
    request: Callable[[tuple, dict], tuple[list, list | None]]
    outputs: Callable[[Any, tuple, dict], list]
    block: Callable[[Any, int], Any]
    is_async: bool | None = None
    model: Callable[[tuple, dict], Any] = _kw_model
    stream: Callable[[tuple, dict], bool] | None = None
    merge_stream: Callable[[list], dict[int, dict]] | None = None
    replace_stream: Callable[[list, set[int]], list] | None = None
    bypass_inner: bool = False


# ---------------------------------------------------------------- 글로 낸 도구 호출

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)        # Hermes, Qwen
_PYTHON_TAG = re.compile(r"<\|python_tag\|>\s*(.*?)\s*(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL)  # Llama 3.1+
_FUNCTION_TAG = re.compile(r"<function=([\w.\-]+)>\s*(.*?)\s*</function>", re.DOTALL)        # Llama 3 함수 태그
_MISTRAL_TAG = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.DOTALL)                    # Mistral
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _decode_all(text: str) -> list[Any]:
    """'{...}', '[...]', '{...}; {...}' 같은 글에서 JSON 값을 차례로 읽는다."""
    decoder, values, position = json.JSONDecoder(), [], 0
    text = text.strip()
    while position < len(text):
        while position < len(text) and text[position] in " \t\r\n;,":
            position += 1
        if position >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return values if values else []
        values.append(value)
        position = end
    return values


def _as_call(value: Any, name: str | None = None) -> dict | None:
    if name is not None:
        return {"type": "function", "function": {"name": name, "arguments": _json(value)}}
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        value = value["function"]
    found = value.get("name")
    if not isinstance(found, str) or not found:
        return None
    if "arguments" in value:
        arguments = value["arguments"]
    elif "parameters" in value:
        arguments = value["parameters"]
    else:
        return None
    return {"type": "function", "function": {"name": found, "arguments": _json(arguments)}}


def _calls_from(values: list[Any]) -> list[dict]:
    calls = []
    for value in values:
        for item in value if isinstance(value, list) else [value]:
            call = _as_call(item)
            if call is not None:
                calls.append(call)
    return calls


def parse_text_tool_calls(text: str) -> tuple[str, list[dict]]:
    """모델이 글로 낸 도구 호출을 OpenAI tool_calls로 읽는다. 돌려주는 글에서는 도구 호출 부분을 뺀다.

    읽는 형식: <tool_call>{...}</tool_call>, <|python_tag|>{...}, <function=name>{...}</function>,
    [TOOL_CALLS] [...], 그리고 출력 전체가 {"name": ..., "arguments": ...} JSON(또는 그 목록)인 경우.
    """
    text = text or ""
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    for pattern in (_TOOL_CALL_TAG, _PYTHON_TAG, _MISTRAL_TAG):
        for match in pattern.finditer(text):
            found = _calls_from(_decode_all(match.group(1)))
            if found:
                calls.extend(found)
                spans.append(match.span())
    for match in _FUNCTION_TAG.finditer(text):
        values = _decode_all(match.group(2))
        calls.append(_as_call(values[0] if values else {}, name=match.group(1)))
        spans.append(match.span())
    if not calls:
        whole = text.strip()
        fenced = _FENCE.match(whole)
        body = fenced.group(1) if fenced else whole
        if body[:1] in "{[":
            values = _decode_all(body)
            found = _calls_from(values)
            # 출력 전체가 도구 호출일 때만 도구 호출로 본다 (설명 속 JSON 예시는 도구 호출이 아니다).
            if found and len(found) == sum(len(v) if isinstance(v, list) else 1 for v in values):
                return "", [dict(call, id=f"call_{index}") for index, call in enumerate(found)]
        return text, []
    content = text
    for start, end in sorted(spans, reverse=True):
        content = content[:start] + content[end:]
    return content.strip(), [dict(call, id=f"call_{index}") for index, call in enumerate(calls)]


def text_output(text: str) -> dict:
    """로컬 모델이 낸 글 하나를 판정할 출력 메시지로 만든다 (생각 부분과 도구 호출을 나눈다)."""
    from simguard.llm_backends import split_reasoning

    answer, reasoning = split_reasoning(text or "")
    content, calls = parse_text_tool_calls(answer)
    output: dict = {"role": "assistant", "content": content}
    if calls:
        output["tool_calls"] = calls
    if reasoning:
        output["reasoning"] = reasoning
    return output


# ---------------------------------------------------------------- OpenAI SDK, LiteLLM

def _openai_request(position: int | None) -> Callable:
    def request(args: tuple, kwargs: dict) -> tuple[list, list | None]:
        messages = argument(args, kwargs, "messages", position)
        tools = plain(kwargs.get("tools"))
        return list(plain(messages) or []), tools if isinstance(tools, list) and tools else None
    return request


def _openai_outputs(response: Any, args: tuple, kwargs: dict) -> list[dict]:
    outputs = []
    for choice in response.choices:
        output = plain(choice.message)
        outputs.append(output if isinstance(output, dict) else {"content": str(output)})
    return outputs


def _openai_block(response: Any, index: int) -> None:
    choice = response.choices[index]
    message = choice.message
    message.content = BLOCKED_MESSAGE
    for name in ("tool_calls", "function_call"):
        if getattr(message, name, None) is not None:
            setattr(message, name, None)
    choice.finish_reason = "stop"


def _openai_merge(chunks: list) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for chunk in chunks:
        for choice in get(chunk, "choices") or []:
            slot = merged.setdefault(get(choice, "index") or 0, {"content": "", "reasoning": "", "calls": {}})
            delta = get(choice, "delta")
            if delta is None:
                continue
            slot["content"] += get(delta, "content") or ""
            reasoning = get(delta, "reasoning_content") or get(delta, "reasoning")
            if isinstance(reasoning, str):
                slot["reasoning"] += reasoning
            for call in get(delta, "tool_calls") or []:
                entry = slot["calls"].setdefault(
                    get(call, "index") or 0,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                entry["id"] = get(call, "id") or entry["id"]
                function = get(call, "function")
                if function is not None:
                    entry["function"]["name"] += get(function, "name") or ""
                    entry["function"]["arguments"] += get(function, "arguments") or ""
    outputs = {}
    for index, slot in merged.items():
        output: dict = {"role": "assistant", "content": slot["content"]}
        if slot["calls"]:
            output["tool_calls"] = [slot["calls"][key] for key in sorted(slot["calls"])]
        if slot["reasoning"]:
            output["reasoning"] = slot["reasoning"]
        outputs[index] = output
    return outputs


def _openai_replace(chunks: list, blocked: set[int]) -> list:
    """BLOCK된 선택지는 차단 문구 한 번으로 바꾸고 나머지 내용은 내보내지 않는다."""
    emitted: set[int] = set()
    kept_chunks = []
    for chunk in chunks:
        choices = get(chunk, "choices") or []
        if not choices:
            kept_chunks.append(chunk)
            continue
        kept = []
        for choice in choices:
            index = get(choice, "index") or 0
            if index not in blocked:
                kept.append(choice)
            elif index not in emitted:
                emitted.add(index)
                delta = get(choice, "delta")
                put(delta, "content", BLOCKED_MESSAGE)
                if get(delta, "tool_calls") is not None:
                    put(delta, "tool_calls", None)
                put(choice, "finish_reason", "stop")
                kept.append(choice)
        if kept:
            put(chunk, "choices", kept)
            kept_chunks.append(chunk)
    return kept_chunks


def _openai_like(owner: Any, name: str, is_async: bool, position: int | None = None,
                 model: Callable = _kw_model, bypass_inner: bool = False) -> LLMCall:
    return LLMCall(owner, name, _openai_request(position), _openai_outputs, _openai_block, is_async=is_async,
                   model=model, stream=_kw_stream, merge_stream=_openai_merge, replace_stream=_openai_replace,
                   bypass_inner=bypass_inner)


def _openai_calls() -> list[LLMCall]:
    from openai.resources.chat.completions import AsyncCompletions, Completions

    return [_openai_like(Completions, "create", False), _openai_like(AsyncCompletions, "create", True)]


def _litellm_calls() -> list[LLMCall]:
    import litellm

    def model(args: tuple, kwargs: dict) -> Any:
        return argument(args, kwargs, "model", 0)

    # LiteLLM은 안에서 OpenAI SDK를 부를 수 있으므로 그 안쪽 호출은 다시 판정하지 않는다.
    return [_openai_like(litellm, "completion", False, 1, model, bypass_inner=True),
            _openai_like(litellm, "acompletion", True, 1, model, bypass_inner=True)]


# ---------------------------------------------------------------- Anthropic SDK

def _anthropic_request(args: tuple, kwargs: dict) -> tuple[list, list | None]:
    messages: list[dict] = []
    system = plain(kwargs.get("system"))
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for message in plain(kwargs.get("messages") or []):
        role, content = message.get("role", "user"), message.get("content")
        if not isinstance(content, list):
            messages.append({"role": role, "content": _text_of(content)})
            continue
        texts, calls = [], []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text") or "")
            elif kind in {"tool_use", "server_tool_use"}:
                calls.append({"id": block.get("id"), "type": "function",
                              "function": {"name": block.get("name"), "arguments": _json(block.get("input"))}})
            elif kind == "tool_result":
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                                 "content": _text_of(block.get("content"))})
            elif kind not in {"thinking", "redacted_thinking"}:
                texts.append(f"[{kind}]")
        if texts or calls:
            row: dict = {"role": role, "content": "\n".join(texts)}
            if calls:
                row["tool_calls"] = calls
            messages.append(row)
    tools = [{"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                               "parameters": tool.get("input_schema") or {}}}
             for tool in plain(kwargs.get("tools") or []) if isinstance(tool, dict) and tool.get("name")]
    return messages, tools or None


def _anthropic_output(blocks: list) -> dict:
    texts, calls, reasoning = [], [], []
    for block in blocks:
        kind = get(block, "type")
        if kind == "text":
            texts.append(get(block, "text") or "")
        elif kind in {"tool_use", "server_tool_use"}:
            calls.append({"id": get(block, "id"), "type": "function",
                          "function": {"name": get(block, "name"), "arguments": _json(plain(get(block, "input")))}})
        elif kind == "thinking":
            reasoning.append(get(block, "thinking") or "")
    output: dict = {"role": "assistant", "content": "".join(texts)}
    if calls:
        output["tool_calls"] = calls
    if reasoning:
        output["reasoning"] = "".join(reasoning)
    return output


def _anthropic_outputs(response: Any, args: tuple, kwargs: dict) -> list[dict]:
    return [_anthropic_output(list(get(response, "content") or []))]


def _anthropic_text_block() -> Any:
    from anthropic.types import TextBlock

    return TextBlock.model_construct(type="text", text=BLOCKED_MESSAGE, citations=None)


def _anthropic_block(response: Any, index: int) -> None:
    put(response, "content", [_anthropic_text_block()])
    put(response, "stop_reason", "end_turn")


def _anthropic_merge(events: list) -> dict[int, dict]:
    blocks: dict[int, dict] = {}
    for event in events:
        kind, index = get(event, "type"), get(event, "index")
        if kind == "content_block_start":
            block = plain(get(event, "content_block")) or {}
            blocks[index] = dict(block, _json="")
        elif kind == "content_block_delta" and index in blocks:
            delta = get(event, "delta")
            delta_kind = get(delta, "type")
            if delta_kind == "text_delta":
                blocks[index]["text"] = (blocks[index].get("text") or "") + (get(delta, "text") or "")
            elif delta_kind == "input_json_delta":
                blocks[index]["_json"] += get(delta, "partial_json") or ""
            elif delta_kind == "thinking_delta":
                blocks[index]["thinking"] = (blocks[index].get("thinking") or "") + (get(delta, "thinking") or "")
    ordered = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.get("_json"):
            try:
                block["input"] = json.loads(block["_json"])
            except ValueError:
                block["input"] = block["_json"]
        ordered.append(block)
    return {0: _anthropic_output(ordered)}


def _anthropic_replace(events: list, blocked: set[int]) -> list:
    from anthropic.types import RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent, TextDelta

    start = [event for event in events if get(event, "type") == "message_start"][:1]
    delta = [event for event in events if get(event, "type") == "message_delta"][-1:]
    stop = [event for event in events if get(event, "type") == "message_stop"][-1:]
    for event in delta:
        put(get(event, "delta"), "stop_reason", "end_turn")
    text = _anthropic_text_block()
    body = [
        RawContentBlockStartEvent.model_construct(type="content_block_start", index=0,
                                                  content_block=type(text).model_construct(type="text", text="", citations=None)),
        RawContentBlockDeltaEvent.model_construct(type="content_block_delta", index=0,
                                                  delta=TextDelta.model_construct(type="text_delta", text=BLOCKED_MESSAGE)),
        RawContentBlockStopEvent.model_construct(type="content_block_stop", index=0),
    ]
    return start + body + delta + stop


def _anthropic_calls() -> list[LLMCall]:
    from anthropic.resources.messages import AsyncMessages, Messages

    def build(owner: Any, is_async: bool) -> LLMCall:
        return LLMCall(owner, "create", _anthropic_request, _anthropic_outputs, _anthropic_block, is_async=is_async,
                       stream=_kw_stream, merge_stream=_anthropic_merge, replace_stream=_anthropic_replace)

    return [build(Messages, False), build(AsyncMessages, True)]


# ---------------------------------------------------------------- Ollama

def _ollama_calls_json(calls: Any) -> list[dict]:
    rows = []
    for index, call in enumerate(plain(calls) or []):
        function = call.get("function") or {}
        rows.append({"id": call.get("id") or f"call_{index}", "type": "function",
                     "function": {"name": function.get("name"), "arguments": _json(function.get("arguments"))}})
    return rows


def _ollama_request(offset: int) -> Callable:
    def request(args: tuple, kwargs: dict) -> tuple[list, list | None]:
        messages = []
        for message in plain(argument(args, kwargs, "messages", offset + 1) or []):
            row = {"role": message.get("role", "user"), "content": message.get("content") or ""}
            if message.get("tool_calls"):
                row["tool_calls"] = _ollama_calls_json(message["tool_calls"])
            if message.get("tool_name"):
                row["name"] = message["tool_name"]
            messages.append(row)
        tools = []
        for tool in kwargs.get("tools") or []:
            if callable(tool) and not isinstance(tool, Mapping):
                try:
                    from ollama._utils import convert_function_to_tool

                    tool = convert_function_to_tool(tool)
                except Exception:
                    tool = {"type": "function", "function": {"name": getattr(tool, "__name__", "tool"),
                                                             "description": getattr(tool, "__doc__", "") or ""}}
            tools.append(plain(tool))
        return messages, tools or None
    return request


def _ollama_output(message: Any) -> dict:
    output: dict = {"role": "assistant", "content": get(message, "content") or ""}
    calls = _ollama_calls_json(get(message, "tool_calls"))
    if calls:
        output["tool_calls"] = calls
    if get(message, "thinking"):
        output["reasoning"] = get(message, "thinking")
    return output


def _ollama_outputs(response: Any, args: tuple, kwargs: dict) -> list[dict]:
    return [_ollama_output(get(response, "message"))]


def _ollama_block(response: Any, index: int) -> None:
    message = get(response, "message")
    put(message, "content", BLOCKED_MESSAGE)
    for name in ("tool_calls", "thinking"):
        if get(message, name) is not None:
            put(message, name, None)
    if hasattr(response, "done_reason") or isinstance(response, dict):
        put(response, "done_reason", "stop")


def _ollama_merge(chunks: list) -> dict[int, dict]:
    content, thinking, calls = "", "", []
    for chunk in chunks:
        message = get(chunk, "message")
        if message is None:
            continue
        content += get(message, "content") or ""
        thinking += get(message, "thinking") or ""
        calls.extend(get(message, "tool_calls") or [])
    message = {"content": content, "thinking": thinking or None, "tool_calls": calls or None}
    return {0: _ollama_output(message)}


def _ollama_replace(chunks: list, blocked: set[int]) -> list:
    if not chunks:
        return chunks
    last = copy.deepcopy(chunks[-1])
    _ollama_block(last, 0)
    return [last]


def _ollama_calls() -> list[LLMCall]:
    import ollama

    def model(offset: int) -> Callable:
        return lambda args, kwargs: argument(args, kwargs, "model", offset)

    common = dict(outputs=_ollama_outputs, block=_ollama_block, stream=_kw_stream,
                  merge_stream=_ollama_merge, replace_stream=_ollama_replace)
    calls = [LLMCall(ollama.Client, "chat", _ollama_request(1), model=model(1), is_async=False, **common),
             LLMCall(ollama.AsyncClient, "chat", _ollama_request(1), model=model(1), is_async=True, **common)]
    if callable(getattr(ollama, "chat", None)):
        # ollama.chat은 import 때 묶인 기본 Client의 메서드라 따로 가로챈다.
        calls.append(LLMCall(ollama, "chat", _ollama_request(0), model=model(0), is_async=False,
                             bypass_inner=True, **common))
    return calls


# ---------------------------------------------------------------- vLLM (이 프로세스의 GPU)

def _conversations(value: Any) -> tuple[list[list[dict]], bool]:
    """대화 하나 또는 대화 목록. (대화들, 배치인지)."""
    value = plain(value)
    if isinstance(value, str):
        return [[{"role": "user", "content": value}]], False
    if isinstance(value, Mapping) and "messages" in value:
        return [list(value["messages"])], False
    if not value:
        return [[]], False
    first = value[0]
    if isinstance(first, Mapping) and "role" in first:
        return [list(value)], False
    rows = []
    for item in value:
        rows.extend(_conversations(item)[0])
    return rows, True


def _vllm_model(args: tuple, kwargs: dict) -> Any:
    engine = getattr(args[0], "llm_engine", None) if args else None
    return getattr(getattr(engine, "model_config", None), "model", None)


def _vllm_request(args: tuple, kwargs: dict) -> tuple[list, list | None]:
    tools = plain(argument(args, kwargs, "tools", None))
    return _conversations(argument(args, kwargs, "messages", 1))[0][0], tools or None


def _vllm_outputs(response: Any, args: tuple, kwargs: dict) -> list[tuple[list, dict]]:
    conversations = _conversations(argument(args, kwargs, "messages", 1))[0]
    items = []
    for conversation, result in zip(conversations, response):
        for completion in result.outputs:
            items.append((conversation, text_output(completion.text)))
    return items


def _vllm_block(response: Any, index: int) -> None:
    completions = [completion for result in response for completion in result.outputs]
    completions[index].text = BLOCKED_MESSAGE


def _vllm_calls() -> list[LLMCall]:
    from vllm.entrypoints.llm import LLM

    return [LLMCall(LLM, "chat", _vllm_request, _vllm_outputs, _vllm_block, is_async=False, model=_vllm_model)]


# ---------------------------------------------------------------- transformers pipeline (이 프로세스의 GPU)

def _hf_model(args: tuple, kwargs: dict) -> Any:
    model = getattr(args[0], "model", None) if args else None
    return getattr(model, "name_or_path", None)


def _hf_request(args: tuple, kwargs: dict) -> tuple[list, list | None]:
    return _conversations(argument(args, kwargs, "text_inputs", 1))[0][0], None


def _hf_generations(response: Any) -> list[dict]:
    if isinstance(response, list) and response and isinstance(response[0], list):
        return [generation for group in response for generation in group]
    return list(response) if isinstance(response, list) else [response]


def _hf_outputs(response: Any, args: tuple, kwargs: dict) -> list[tuple[list, dict]]:
    conversations, batch = _conversations(argument(args, kwargs, "text_inputs", 1))
    groups = response if batch else [response]
    prompt = argument(args, kwargs, "text_inputs", 1)
    items = []
    for conversation, group in zip(conversations, groups):
        for generation in group if isinstance(group, list) else [group]:
            generated = generation.get("generated_text") if isinstance(generation, Mapping) else generation
            if isinstance(generated, list):
                text = _text_of((generated[-1] or {}).get("content")) if generated else ""
            else:
                text = str(generated or "")
                if isinstance(prompt, str) and text.startswith(prompt):     # return_full_text=True
                    text = text[len(prompt):]
            items.append((conversation, text_output(text)))
    return items


def _hf_block(response: Any, index: int) -> None:
    generation = _hf_generations(response)[index]
    generated = generation.get("generated_text")
    if isinstance(generated, list) and generated and isinstance(generated[-1], dict):
        generated[-1]["content"] = BLOCKED_MESSAGE
    else:
        generation["generated_text"] = BLOCKED_MESSAGE


def _transformers_calls() -> list[LLMCall]:
    from transformers.pipelines.text_generation import TextGenerationPipeline

    return [LLMCall(TextGenerationPipeline, "__call__", _hf_request, _hf_outputs, _hf_block,
                    is_async=False, model=_hf_model)]


# ---------------------------------------------------------------- 목록

# 이름: (불러오면 연결할 모듈, 가로챌 호출을 만드는 함수)
PROVIDERS: dict[str, tuple[str, Callable[[], list[LLMCall]]]] = {
    "openai": ("openai.resources.chat.completions", _openai_calls),
    "litellm": ("litellm", _litellm_calls),
    "anthropic": ("anthropic.resources.messages", _anthropic_calls),
    "ollama": ("ollama", _ollama_calls),
    "vllm": ("vllm.entrypoints.llm", _vllm_calls),
    "transformers": ("transformers.pipelines.text_generation", _transformers_calls),
}
