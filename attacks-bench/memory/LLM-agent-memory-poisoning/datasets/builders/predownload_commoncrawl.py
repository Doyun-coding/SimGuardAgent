#!/usr/bin/env python3
"""Predownload Common Crawl HTML into local JSONL cache files."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


SOURCES = (
    # {"domain": "web_html_random", "query": "*.com/*", "weight": 1.2},
    {"domain": "web_html_wikipedia", "query": "*.wikipedia.org/wiki/*", "weight": 0.8},
    # {"domain": "web_html_wikipedia_en", "query": "en.wikipedia.org/wiki/*", "weight": 0.6},
)

POPULAR_WIKIPEDIA_LANGS = (
    # "en",
    "de",
    "fr",
    "es",
    "ru",
    "ja",
    "it",
    "pt",
    "zh",
    "ar",
    "pl",
    "nl",
    "sv",
    "uk",
    "fa",
    "he",
    "tr",
    "cs",
    "ko",
    "vi",
    "id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Cache output directory.")
    parser.add_argument("--crawl", type=str, default="CC-MAIN-2024-10", help="Common Crawl ID.")
    parser.add_argument("--docs-per-source", type=int, default=2000, help="Rows to cache per source.")
    parser.add_argument("--max-pages", type=int, default=80, help="Index pages to scan.")
    parser.add_argument("--page-size", type=int, default=2000, help="Common Crawl index pageSize.")
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum HTML chars to keep.")
    parser.add_argument("--host-cap-random", type=int, default=4, help="Host cap for random web source.")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--http-retries", type=int, default=4, help="HTTP retry attempts per request.")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.7,
        help="Base backoff seconds for HTTP retries (multiplied by attempt index).",
    )
    parser.add_argument(
        "--wikipedia-languages",
        type=str,
        default=",".join(POPULAR_WIKIPEDIA_LANGS),
        help="Comma-separated Wikipedia language codes for web_html_wikipedia source.",
    )
    parser.add_argument(
        "--max-pages-per-wiki-language",
        type=int,
        default=10,
        help="Common Crawl index pages sampled per Wikipedia language.",
    )
    return parser.parse_args()


def is_meaningful_web_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if not parsed.netloc:
        return False
    path = (parsed.path or "/").strip().lower()
    if path in {"/", ""}:
        return False
    low_signal_paths = {
        "/index",
        "/index.html",
        "/home",
        "/home.html",
        "/login",
        "/signin",
        "/signup",
        "/register",
        "/search",
        "/about",
        "/contact",
        "/privacy",
        "/terms",
        "/sitemap",
    }
    if path in low_signal_paths:
        return False
    if parsed.query and len(parsed.query) > 200:
        return False
    return True


def parse_warc_html(payload: bytes) -> str | None:
    if payload.startswith(b"\x1f\x8b"):
        try:
            payload = gzip.decompress(payload)
        except OSError:
            return None
    sep = payload.find(b"\r\n\r\n")
    if sep == -1:
        return None
    body = payload[sep + 4 :]
    if body.startswith(b"HTTP/"):
        http_sep = body.find(b"\r\n\r\n")
        if http_sep == -1:
            return None
        body = body[http_sep + 4 :]
    html = body.decode("utf-8", errors="ignore").strip()
    return html if html else None


def index_records(
    crawl: str,
    query: str,
    max_records: int,
    max_pages: int,
    page_size: int,
    timeout: int,
    rng: random.Random,
    http_retries: int,
    retry_backoff_seconds: float,
) -> list[dict[str, Any]]:
    endpoint = f"https://index.commoncrawl.org/{crawl}-index"
    seen_urls: set[str] = set()
    out: list[dict[str, Any]] = []
    total_pages = get_commoncrawl_num_pages(
        endpoint=endpoint,
        query=query,
        page_size=page_size,
        timeout=timeout,
        http_retries=http_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    if total_pages is None or total_pages <= 0:
        page_order = list(range(max_pages))
    else:
        page_count = min(max_pages, total_pages)
        page_order = rng.sample(range(total_pages), k=page_count)

    for page in page_order:
        params: dict[str, Any] = {
            "url": query,
            "output": "json",
            "filter": ["status:200", "mime:text/html"],
            "page": page,
            "pageSize": page_size,
        }
        resp = http_get_with_retry(
            endpoint,
            timeout=timeout,
            retries=http_retries,
            backoff_seconds=retry_backoff_seconds,
            params=params,
        )
        lines = resp.text.splitlines()
        if not lines:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = str(record.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            if not is_meaningful_web_url(url):
                continue
            seen_urls.add(url)
            out.append(record)
            if len(out) >= max_records:
                return out
    return out


def get_commoncrawl_num_pages(
    endpoint: str,
    query: str,
    page_size: int,
    timeout: int,
    http_retries: int,
    retry_backoff_seconds: float,
) -> int | None:
    params: dict[str, Any] = {
        "url": query,
        "output": "json",
        "filter": ["status:200", "mime:text/html"],
        "showNumPages": "true",
        "pageSize": page_size,
    }
    try:
        resp = http_get_with_retry(
            endpoint,
            timeout=timeout,
            retries=http_retries,
            backoff_seconds=retry_backoff_seconds,
            params=params,
        )
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return None

    value = payload.get("pages")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_html(
    record: dict[str, Any],
    timeout: int,
    http_retries: int,
    retry_backoff_seconds: float,
) -> str | None:
    filename = record.get("filename")
    offset = record.get("offset")
    length = record.get("length")
    if not filename or offset is None or length is None:
        return None
    try:
        offset_i = int(offset)
        length_i = int(length)
    except (TypeError, ValueError):
        return None
    if length_i <= 0:
        return None
    end_i = offset_i + length_i - 1
    url = f"https://data.commoncrawl.org/{filename}"
    resp = http_get_with_retry(
        url,
        timeout=timeout,
        retries=http_retries,
        backoff_seconds=retry_backoff_seconds,
        headers={"Range": f"bytes={offset_i}-{end_i}"},
    )
    return parse_warc_html(resp.content)


def http_get_with_retry(
    url: str,
    timeout: int,
    retries: int,
    backoff_seconds: float,
    **kwargs: Any,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            sleep_for = max(0.0, backoff_seconds * (attempt + 1))
            time.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed with unknown error")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wiki_langs = [lang.strip() for lang in args.wikipedia_languages.split(",") if lang.strip()]
    if not wiki_langs:
        wiki_langs = list(POPULAR_WIKIPEDIA_LANGS)

    for source in SOURCES:
        domain = source["domain"]
        query = source["query"]
        target = args.docs_per_source
        print(f"[{domain}] collecting index records...")

        if domain == "web_html_wikipedia":
            records = []
            per_lang_records = max(300, (max(target * 5, 5000) // max(len(wiki_langs), 1)))
            for lang in wiki_langs:
                lang_query = f"{lang}.wikipedia.org/wiki/*"
                print(f"[{domain}] indexing language={lang} ...")
                lang_records = index_records(
                    crawl=args.crawl,
                    query=lang_query,
                    max_records=per_lang_records,
                    max_pages=args.max_pages_per_wiki_language,
                    page_size=args.page_size,
                    timeout=args.timeout,
                    rng=rng,
                    http_retries=args.http_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                records.extend(lang_records)
        else:
            records = index_records(
                crawl=args.crawl,
                query=query,
                max_records=max(target * 5, 5000),
                max_pages=args.max_pages,
                page_size=args.page_size,
                timeout=args.timeout,
                rng=rng,
                http_retries=args.http_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
        rng.shuffle(records)

        out_path = args.output_dir / f"{domain}.jsonl"
        kept = 0
        seen = 0
        host_usage: dict[str, int] = {}
        with out_path.open("w", encoding="utf-8") as fh:
            for record in records:
                url = str(record.get("url", "")).strip()
                host = urlparse(url).netloc.lower()
                enforce_host_cap = domain == "web_html_random"
                if enforce_host_cap and host:
                    used = host_usage.get(host, 0)
                    if used >= args.host_cap_random:
                        continue

                seen += 1
                if seen % 25 == 0:
                    print(f"[{domain}] attempted={seen} kept={kept}/{target}")
                try:
                    html = fetch_html(
                        record,
                        timeout=args.timeout,
                        http_retries=args.http_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )
                except Exception:
                    continue
                if not html:
                    continue
                html = " ".join(html.split())
                if len(html) < args.min_chars:
                    continue

                row = {
                    "text": html,
                    "metadata": {
                        "url": url,
                        "wiki_language_hint": host.split(".")[0] if host.endswith(".wikipedia.org") else None,
                        "filename": record.get("filename"),
                        "offset": record.get("offset"),
                        "length": record.get("length"),
                        "mime": record.get("mime"),
                        "status": record.get("status"),
                        "digest": record.get("digest"),
                        "crawl": args.crawl,
                    },
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if enforce_host_cap and host:
                    host_usage[host] = host_usage.get(host, 0) + 1
                if kept >= target:
                    break
        print(f"[{domain}] wrote {kept} rows to {out_path}")


if __name__ == "__main__":
    main()
