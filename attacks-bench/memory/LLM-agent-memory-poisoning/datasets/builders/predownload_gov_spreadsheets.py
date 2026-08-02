#!/usr/bin/env python3
"""Crawl government pages and download a small set of spreadsheet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


SEED_URLS = (
    "https://www.data.gov.uk/search?q=&filters%5Bpublisher%5D=&filters%5Btopic%5D=&filters%5Bformat%5D=XLS&sort=best",
)

FILE_EXTENSIONS = (".xlsx", ".xls", ".csv")
ALLOWED_SCHEMES = {"http", "https"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write files and manifest.")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum spreadsheet files to download.")
    parser.add_argument("--max-pages", type=int, default=120, help="Maximum HTML pages to crawl.")
    parser.add_argument("--max-links-per-page", type=int, default=250, help="Limit extracted links per crawled page.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--http-retries", type=int, default=4, help="HTTP retries per request.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.8, help="Linear backoff base seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--formats",
        type=str,
        default="xlsx,xls,csv",
        help="Comma-separated formats to download: xlsx,xls,csv",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="llm-agent-memory-poisoning-gov-sheet-scraper/0.1 (+research; contact: local)",
        help="User agent for HTTP requests.",
    )
    return parser.parse_args()


def is_gov_host(host: str) -> bool:
    host = host.lower()
    return (
        host.endswith(".gov")
        or host == "data.gov"
        or host.endswith(".data.gov")
        or host == "data.gov.uk"
        or host.endswith(".data.gov.uk")
        or host.endswith(".gov.uk")
    )


def is_data_gov_uk_host(host: str) -> bool:
    host = host.lower()
    return host == "www.data.gov.uk" or host == "data.gov.uk"


def normalize_url(base_url: str, href: str) -> str | None:
    if not href or href.startswith("#"):
        return None
    url = urljoin(base_url, href.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return None
    if not parsed.netloc:
        return None
    # Remove fragment for dedupe stability.
    if parsed.fragment:
        url = url.split("#", 1)[0]
    return url


def parse_allowed_extensions(formats_csv: str) -> tuple[str, ...]:
    tokens = [t.strip().lower() for t in formats_csv.split(",") if t.strip()]
    allowed = []
    valid = {"xlsx", "xls", "csv"}
    for token in tokens:
        if token not in valid:
            raise ValueError(f"Unsupported format '{token}'. Use any of: xlsx,xls,csv")
        allowed.append(f".{token}")
    if not allowed:
        raise ValueError("No formats selected. Provide at least one of: xlsx,xls,csv")
    return tuple(dict.fromkeys(allowed))


def looks_like_file_url(url: str, allowed_exts: tuple[str, ...]) -> bool:
    lower = url.lower()
    if any(lower.endswith(ext) for ext in allowed_exts):
        return True
    return False


def is_direct_spreadsheet_url(url: str, allowed_exts: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    lower_path = (parsed.path or "").lower()
    return any(lower_path.endswith(ext) for ext in allowed_exts)


def likely_html_content_type(content_type: str) -> bool:
    return "text/html" in content_type.lower() or "application/xhtml+xml" in content_type.lower()


def hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def safe_filename_from_url(url: str, fallback_ext: str = ".bin") -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = f"download-{hash_url(url)}{fallback_ext}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if "." not in name:
        name = f"{name}{fallback_ext}"
    return name


def http_get_with_retry(
    url: str,
    timeout: int,
    retries: int,
    backoff_seconds: float,
    headers: dict[str, str],
    stream: bool = False,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, stream=stream)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(max(0.0, backoff_seconds * (attempt + 1)))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed without exception")


def crawl_for_gov_spreadsheets(
    seed_urls: list[str],
    max_pages: int,
    max_links_per_page: int,
    timeout: int,
    retries: int,
    backoff_seconds: float,
    headers: dict[str, str],
    rng: random.Random,
    allowed_exts: tuple[str, ...],
) -> list[dict[str, str]]:
    visited_pages: set[str] = set()
    seen_file_urls: set[str] = set()
    queue: deque[str] = deque(seed_urls)
    discovered_files: list[dict[str, str]] = []

    while queue and len(visited_pages) < max_pages:
        page_url = queue.popleft()
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)

        parsed = urlparse(page_url)
        if not is_data_gov_uk_host(parsed.netloc):
            continue
        print(
            f"[gov-sheets][crawl] page {len(visited_pages)}/{max_pages} exploring: {page_url}",
            flush=True,
        )

        try:
            resp = http_get_with_retry(
                page_url,
                timeout=timeout,
                retries=retries,
                backoff_seconds=backoff_seconds,
                headers=headers,
            )
        except Exception:
            continue

        content_type = resp.headers.get("content-type", "")
        if not likely_html_content_type(content_type):
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)
        rng.shuffle(links)
        links = links[:max_links_per_page]

        for tag in links:
            href = tag.get("href")
            if not isinstance(href, str):
                continue
            url = normalize_url(page_url, href)
            if not url:
                continue
            u = urlparse(url)
            if looks_like_file_url(url, allowed_exts=allowed_exts) and is_direct_spreadsheet_url(
                url, allowed_exts=allowed_exts
            ):
                if url not in seen_file_urls:
                    seen_file_urls.add(url)
                    discovered_files.append({"file_url": url, "source_page": page_url})
                    if len(discovered_files) % 10 == 0:
                        print(
                            f"[gov-sheets][crawl] discovered files: {len(discovered_files)}",
                            flush=True,
                        )
                continue

            # Continue crawling only within data.gov.uk listing pages.
            if not is_data_gov_uk_host(u.netloc):
                continue
            if url not in visited_pages and url not in queue:
                queue.append(url)

    return discovered_files


def main() -> None:
    args = parse_args()
    if args.max_files <= 0:
        raise ValueError("--max-files must be > 0")
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be > 0")

    rng = random.Random(args.seed)
    allowed_exts = parse_allowed_extensions(args.formats)
    print(f"[gov-sheets] allowed formats: {','.join(ext.lstrip('.') for ext in allowed_exts)}", flush=True)
    headers = {"User-Agent": args.user_agent, "Accept": "*/*"}
    output_dir = args.output_dir
    files_dir = output_dir / "gov_spreadsheets_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gov_spreadsheets_manifest.jsonl"

    seed_urls = list(SEED_URLS)
    rng.shuffle(seed_urls)
    print(f"[gov-sheets] crawling up to {args.max_pages} pages from {len(seed_urls)} seeds", flush=True)
    discovered = crawl_for_gov_spreadsheets(
        seed_urls=seed_urls,
        max_pages=args.max_pages,
        max_links_per_page=args.max_links_per_page,
        timeout=args.timeout,
        retries=args.http_retries,
        backoff_seconds=args.retry_backoff_seconds,
        headers=headers,
        rng=rng,
        allowed_exts=allowed_exts,
    )
    print(f"[gov-sheets] discovered {len(discovered)} candidate file URLs", flush=True)

    rng.shuffle(discovered)
    kept = 0
    attempted = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for item in discovered:
            if kept >= args.max_files:
                break
            attempted += 1
            file_url = item["file_url"]
            source_page = item["source_page"]
            print(
                f"[gov-sheets][download] attempt {attempted} target {kept+1}/{args.max_files}: {file_url}",
                flush=True,
            )
            try:
                resp = http_get_with_retry(
                    file_url,
                    timeout=args.timeout,
                    retries=args.http_retries,
                    backoff_seconds=args.retry_backoff_seconds,
                    headers=headers,
                    stream=True,
                )
                content_type = resp.headers.get("content-type", "")
                ext = Path(urlparse(file_url).path).suffix.lower()
                if ext not in allowed_exts:
                    if "spreadsheetml" in content_type.lower():
                        ext = ".xlsx"
                    elif "excel" in content_type.lower():
                        ext = ".xls"
                    elif "csv" in content_type.lower():
                        ext = ".csv"
                    else:
                        ext = ".bin"
                if ext not in allowed_exts:
                    print(f"[gov-sheets][download] skipped disallowed format ({ext}): {file_url}", flush=True)
                    continue
                name = safe_filename_from_url(file_url, fallback_ext=ext)
                unique_name = f"{kept+1:03d}_{hash_url(file_url)}_{name}"
                local_path = files_dir / unique_name
                with local_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                size_bytes = local_path.stat().st_size
                if size_bytes < 512:
                    local_path.unlink(missing_ok=True)
                    print(f"[gov-sheets][download] skipped tiny file: {file_url}", flush=True)
                    continue
            except Exception:
                print(f"[gov-sheets][download] failed: {file_url}", flush=True)
                continue

            row = {
                "source_page": source_page,
                "file_url": file_url,
                "local_path": str(local_path),
                "content_type": resp.headers.get("content-type", ""),
                "size_bytes": size_bytes,
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())
            kept += 1
            print(f"[gov-sheets] kept {kept}/{args.max_files}: {file_url}", flush=True)

    print(
        f"[gov-sheets] wrote {kept} files -> {files_dir} | manifest -> {manifest_path} | attempted {attempted}",
        flush=True,
    )


if __name__ == "__main__":
    main()
