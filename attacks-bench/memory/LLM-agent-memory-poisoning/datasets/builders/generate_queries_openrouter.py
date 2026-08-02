#!/usr/bin/env python3
"""Generate user queries for documents using OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import tiktoken


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input dataset JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("datasets/provenance/query_taxonomy.json"),
        help="Query taxonomy JSON path.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model id. Can be repeated.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional list of OpenRouter models. If set, overrides --model.",
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=3,
        help="Number of queries to generate per document.",
    )
    parser.add_argument(
        "--avg-output-tokens",
        type=float,
        default=300,
        help="Estimated completion tokens per document response.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum number of documents to process (0 = all).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="",
        help="Only generate queries for a specific doc_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (unused when processing sequentially).",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=4000,
        help="Max document characters sent to the model.",
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=200,
        help="Minimum document characters required to send.",
    )
    parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include binary placeholder docs (PDF/Excel) as text-only requests.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=[],
        help="Optional domain filter (e.g. --domains legal news).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--referer",
        type=str,
        default="http://localhost",
        help="HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="llm-agent-memory-poisoning-query-generator",
        help="X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for API calls.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Linear backoff base seconds for retries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents already present in output JSONL.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only generate queries for documents that are missing any model outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel workers for OpenRouter requests.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="Max in-flight documents (0 = 2x max-workers).",
    )
    parser.add_argument(
        "--chat-overhead-tokens",
        type=int,
        default=8,
        help="Approx overhead tokens per chat request.",
    )
    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip upfront cost estimate.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_taxonomy_prompt(taxonomy: dict[str, Any]) -> str:
    return json.dumps(taxonomy, ensure_ascii=False, indent=2)


def is_binary_doc(doc: dict[str, Any]) -> bool:
    text = str(doc.get("text", ""))
    return text in {"PDF_BINARY", "SPREADSHEET_BINARY"}


def build_messages(
    doc: dict[str, Any],
    taxonomy_text: str,
    queries_per_doc: int,
    max_doc_chars: int,
) -> list[dict[str, str]]:
    text = str(doc.get("text", ""))
    clipped = text[:max_doc_chars]
    annotations = doc.get("annotations", {}) if isinstance(doc.get("annotations"), dict) else {}
    language = annotations.get("language") or annotations.get("language_code") or "Unknown"
    doc_format = annotations.get("document_format") or "Unknown"

    system = (
        "You generate realistic user queries over an uploaded document. "
        "Use the provided taxonomy to label each query. "
        "Only generate queries that can be answered from the document text. "
        "Match the query language to the document language. "
        "Vary formality: some queries should be informal, casual, and may skip capitalization or punctuation. "
        "If a query is informal, mark it with a style tag. "
        "Vary query length: include a mix of short and longer queries. "
        "Return a single JSON object only (no markdown, no code fences)."
    )

    domain = str(doc.get("domain", "")).lower()
    code_hint = ""
    if domain == "code":
        code_hint = (
            "For code documents, prefer realistic software requests such as: "
            "refactor for readability/performance, rewrite in another language, "
            "identify bugs or vulnerabilities, suggest tests, explain behavior, "
            "or review style and structure. "
            "Avoid generic summary-only requests."
        )

    user = (
        "Taxonomy (JSON):\n"
        f"{taxonomy_text}\n\n"
        "Document metadata:\n"
        f"- language: {language}\n"
        f"- format: {doc_format}\n"
        f"- domain: {domain}\n\n"
        "Document:\n"
        f"{clipped}\n\n"
        "Task:\n"
        f"Generate {queries_per_doc} distinct user queries relevant to the document. "
        "Each query must map to a taxonomy subcategory (use subcategory id). "
        "Include a small number of nonsensical queries that are NOT answerable from the document. "
        f"{code_hint}\n"
        "Return a JSON object with:\n"
        "- doc_id (string)\n"
        "- queries: array of {query, category_id, category_name, style, is_nonsensical}\n"
        "Set is_nonsensical=true for nonsensical queries; false otherwise.\n"
        f"Exactly {queries_per_doc} queries. All fields must be present; no nulls.\n"
        "Style must be one of: formal, informal. Use informal for some queries.\n"
        "No explanations." 
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    referer: str,
    title: str,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": referer,
        "X-Title": title,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(retry_backoff * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenRouter request failed")


def fetch_models(api_key: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def find_model_pricing(models: dict[str, Any], model_id: str) -> tuple[float, float]:
    data = models.get("data")
    if not isinstance(data, list):
        raise ValueError("Models JSON missing data[]")
    for item in data:
        if item.get("id") == model_id:
            pricing = item.get("pricing", {})
            prompt = pricing.get("prompt") or pricing.get("input")
            completion = pricing.get("completion") or pricing.get("output")
            if prompt is None or completion is None:
                raise ValueError("Pricing not found for model")
            return (float(prompt), float(completion))
    raise ValueError(f"Model not found in pricing list: {model_id}")


def choose_encoding(model_id: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_id)
    except Exception:  # noqa: BLE001
        lowered = model_id.lower()
        if "gpt-4o" in lowered or "4o" in lowered:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")


def count_prompt_tokens(
    messages: list[dict[str, str]],
    encoding: tiktoken.Encoding,
    chat_overhead_tokens: int,
) -> int:
    content_tokens = sum(len(encoding.encode(msg.get("content", ""))) for msg in messages)
    return content_tokens + chat_overhead_tokens


def extract_response_text(resp: dict[str, Any]) -> str:
    try:
        return str(resp["choices"][0]["message"]["content"])  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return ""


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def log_bad_response(doc: dict[str, Any], model: str, raw_text: str, reason: str) -> None:
    doc_id = str(doc.get("doc_id", ""))
    preview = " ".join(raw_text.strip().split())[:160]
    print(
        f"[queries][bad_response] doc_id={doc_id} model={model} reason={reason} preview={preview}",
        flush=True,
    )


def load_existing_doc_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    doc_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def load_existing_doc_models(output_path: Path) -> dict[str, set[str]]:
    if not output_path.exists():
        return {}
    found: dict[str, set[str]] = {}
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            models = row.get("models")
            if not isinstance(models, list):
                continue
            model_ids: set[str] = set()
            for entry in models:
                mid = entry.get("model") if isinstance(entry, dict) else None
                if isinstance(mid, str) and mid:
                    model_ids.add(mid)
            if model_ids:
                found[doc_id] = found.get(doc_id, set()).union(model_ids)
    return found


def normalize_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
    doc_id = str(doc.get("doc_id", ""))
    if not doc_id:
        return None
    text = str(doc.get("text", ""))
    return {**doc, "doc_id": doc_id, "text": text}


def build_output_row(
    doc: dict[str, Any],
    models_payload: list[dict[str, Any]],
    queries_per_doc: int,
) -> dict[str, Any]:
    return {
        "doc_id": doc.get("doc_id"),
        "domain": doc.get("domain"),
        "source_path": doc.get("source_path"),
        "queries_per_doc": queries_per_doc,
        "models": models_payload,
    }


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY or --api-key.")

    data = load_json(args.input)
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise ValueError("Input dataset missing documents[]")

    taxonomy = load_json(args.taxonomy)
    taxonomy_text = build_taxonomy_prompt(taxonomy)

    if args.doc_id:
        docs = [doc for doc in docs if str(doc.get("doc_id", "")) == args.doc_id]
        if not docs:
            raise ValueError(f"doc_id not found in dataset: {args.doc_id}")

    if args.domains:
        allowed = set(args.domains)
        docs = [doc for doc in docs if doc.get("domain") in allowed]

    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    skip_ids: set[str] = set()
    existing_models: dict[str, set[str]] = {}
    if args.resume:
        skip_ids = load_existing_doc_ids(args.output)
    if args.only_missing:
        existing_models = load_existing_doc_models(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    written = 0

    if args.models:
        models = args.models
    else:
        if isinstance(args.model, list):
            models = args.model
        else:
            models = [args.model]
    models = [m for m in models if m]

    worklist: list[dict[str, Any]] = []
    for doc in docs:
        normalized = normalize_doc(doc)
        if not normalized:
            continue
        processed += 1
        if args.resume and normalized["doc_id"] in skip_ids:
            continue
        if args.only_missing:
            doc_models = existing_models.get(normalized["doc_id"], set())
            if doc_models and all(m in doc_models for m in models):
                continue
        if not args.include_binary and is_binary_doc(normalized):
            continue
        if len(normalized["text"]) < args.min_doc_chars:
            continue
        worklist.append(normalized)

    if not args.skip_estimate and worklist:
        models_json = fetch_models(api_key)
        total_cost = 0.0
        for model in models:
            prompt_rate, completion_rate = find_model_pricing(models_json, model)
            encoding = choose_encoding(model)
            total_prompt_tokens = 0
            for doc in worklist:
                messages = build_messages(
                    doc=doc,
                    taxonomy_text=taxonomy_text,
                    queries_per_doc=args.queries_per_doc,
                    max_doc_chars=args.max_doc_chars,
                )
                total_prompt_tokens += count_prompt_tokens(
                    messages=messages,
                    encoding=encoding,
                    chat_overhead_tokens=args.chat_overhead_tokens,
                )
            total_completion_tokens = args.avg_output_tokens * len(worklist)
            total_cost += total_prompt_tokens * prompt_rate + total_completion_tokens * completion_rate
        print(f"${total_cost:.4f}")

    def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        models_payload: list[dict[str, Any]] = []
        for model in models:
            messages = build_messages(
                doc=doc,
                taxonomy_text=taxonomy_text,
                queries_per_doc=args.queries_per_doc,
                max_doc_chars=args.max_doc_chars,
            )
            try:
                resp = call_openrouter(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    referer=args.referer,
                    title=args.title,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                )
                raw_text = extract_response_text(resp)
                parsed = parse_json_response(raw_text)
                if parsed is None:
                    reason = "invalid_json"
                    if raw_text.strip().startswith("```"):
                        reason = "markdown_fence"
                    log_bad_response(doc, model, raw_text, reason)
                elif not isinstance(parsed, dict):
                    log_bad_response(doc, model, raw_text, "not_object")
                models_payload.append(
                    {
                        "model": model,
                        "response": parsed,
                        "raw_response": raw_text if parsed is None else None,
                        "usage": resp.get("usage"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[queries][error] doc_id={doc.get('doc_id','')} model={model} error={exc}",
                    flush=True,
                )
                models_payload.append(
                    {
                        "model": model,
                        "response": None,
                        "raw_response": None,
                        "usage": None,
                        "error": str(exc),
                    }
                )
        return build_output_row(
            doc=doc,
            models_payload=models_payload,
            queries_per_doc=args.queries_per_doc,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out:
        max_inflight = args.max_inflight if args.max_inflight > 0 else args.max_workers * 2
        pending: set = set()
        future_to_idx: dict[Any, int] = {}
        results_by_idx: dict[int, dict[str, Any]] = {}
        next_to_write = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for idx, doc in enumerate(worklist):
                while len(pending) >= max_inflight:
                    done = next(as_completed(pending))
                    pending.remove(done)
                    doc_idx = future_to_idx.pop(done)
                    results_by_idx[doc_idx] = done.result()
                    while next_to_write in results_by_idx:
                        out.write(json.dumps(results_by_idx.pop(next_to_write), ensure_ascii=False) + "\n")
                        out.flush()
                        written += 1
                        print(f"[queries] wrote {written} docs", flush=True)
                        next_to_write += 1
                future = executor.submit(process_doc, doc)
                pending.add(future)
                future_to_idx[future] = idx

            for done in as_completed(pending):
                doc_idx = future_to_idx.pop(done)
                try:
                    results_by_idx[doc_idx] = done.result()
                except Exception as exc:  # noqa: BLE001
                    doc = worklist[doc_idx]
                    results_by_idx[doc_idx] = build_output_row(
                        doc=doc,
                        models_payload=[
                            {
                                "model": "unknown",
                                "response": None,
                                "raw_response": None,
                                "usage": None,
                                "error": str(exc),
                            }
                        ],
                        queries_per_doc=args.queries_per_doc,
                    )
                while next_to_write in results_by_idx:
                    out.write(json.dumps(results_by_idx.pop(next_to_write), ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    print(f"[queries] wrote {written} docs", flush=True)
                    next_to_write += 1

    print(f"[queries] processed={processed} written={written} output={args.output}")


if __name__ == "__main__":
    main()
