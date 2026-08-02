#!/usr/bin/env python3
"""Build a mixed-domain document dataset from Hugging Face corpora."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
import requests


@dataclass(frozen=True)
class DatasetSource:
    domain: str
    path: str
    name: str | None = None
    split: str = "train"
    text_fields: tuple[str, ...] = ("text",)
    metadata_fields: tuple[str, ...] = ()
    weight: float = 1.0
    source_type: str = "huggingface"


DEFAULT_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        domain="legal",
        path="HFforLegal/case-law",
        split="us",
        text_fields=("text", "document", "content", "case_text"),
        metadata_fields=("title", "court", "jurisdiction", "date"),
        weight=1.4,
    ),
    DatasetSource(
        domain="books",
        path="nikolina-p/gutenberg_clean_en_splits",
        split="train",
        text_fields=("text", "book_text", "content"),
        metadata_fields=("book_id", "title", "author"),
        weight=0.9,
    ),
    DatasetSource(
        domain="patents",
        path="NortheasternUniversity/big_patent",
        name="a",
        split="train",
        text_fields=("description", "abstract", "background"),
        metadata_fields=("title", "cpc_title"),
        weight=1.0,
    ),
    DatasetSource(
        domain="news",
        path="allenai/c4",
        name="realnewslike",
        split="train",
        text_fields=("text",),
        metadata_fields=("url", "timestamp"),
        weight=1.2,
    ),
    DatasetSource(
        domain="science",
        path="ccdv/pubmed-summarization",
        name="document",
        split="train",
        text_fields=("article", "text", "abstract"),
        metadata_fields=("id",),
        weight=1.0,
    ),
    DatasetSource(
        domain="code",
        path="jugalgajjar/MultiLang-Code-Parser-Dataset",
        split="train",
        text_fields=("code", "content", "text"),
        metadata_fields=("language", "avg_line_length", "line_count", "ast_node_count", "num_errors"),
        weight=1.3,
    ),
    DatasetSource(
        domain="emails",
        path="snoop2head/enron_aeslc_emails",
        split="train",
        text_fields=("body", "text", "email_body"),
        metadata_fields=("subject", "from", "to"),
        weight=1.1,
    ),
    DatasetSource(
        domain="finance_transcripts",
        path="kurry/sp500_earnings_transcripts",
        split="train",
        text_fields=("transcript", "text", "content"),
        metadata_fields=("ticker", "company", "date"),
        weight=0.9,
    ),
    DatasetSource(
        domain="finance_news",
        path="danidanou/Bloomberg_Financial_News",
        split="train",
        text_fields=("text", "content", "article"),
        metadata_fields=("title", "date"),
        weight=0.9,
    ),
    DatasetSource(
        domain="podcasts",
        path="RamAnanth1/lex-fridman-podcasts",
        split="train",
        text_fields=("transcript", "text", "content"),
        metadata_fields=("episode", "title"),
        weight=0.8,
    ),
    DatasetSource(
        domain="encyclopedia",
        path="EdoVaira/Encyclopedia-Britannica",
        split="train",
        text_fields=("text", "content"),
        metadata_fields=("title",),
        weight=1.0,
    ),
    DatasetSource(
        domain="math",
        path="math-ai/AutoMathText",
        split="train",
        text_fields=("text", "content"),
        weight=0.9,
    ),
    DatasetSource(
        domain="tweets",
        path="cardiffnlp/tweet_eval",
        name="sentiment",
        split="train",
        text_fields=("text", "tweet", "content"),
        metadata_fields=("label",),
        weight=0.8,
    ),
    DatasetSource(
        domain="web_html_random",
        path="commoncrawl",
        name="CC-MAIN-2024-10",
        split="*.com/*",
        text_fields=("html",),
        metadata_fields=("url", "filename", "mime", "status", "offset", "length", "digest", "crawl"),
        weight=0.8,
        source_type="commoncrawl",
    ),
    DatasetSource(
        domain="web_html_wikipedia",
        path="commoncrawl",
        name="CC-MAIN-2024-10",
        split="*.wikipedia.org/wiki/*",
        text_fields=("html",),
        metadata_fields=("url", "filename", "mime", "status", "offset", "length", "digest", "crawl"),
        weight=0.8,
        source_type="commoncrawl",
    ),
    DatasetSource(
        domain="web_html_wikipedia_en",
        path="commoncrawl",
        name="CC-MAIN-2024-10",
        split="en.wikipedia.org/wiki/*",
        text_fields=("html",),
        metadata_fields=("url", "filename", "mime", "status", "offset", "length", "digest", "crawl"),
        weight=1.2,
        source_type="commoncrawl",
    ),
    DatasetSource(
        domain="sec_filings",
        path="PleIAs/SEC",
        split="train",
        text_fields=("text", "content", "filing_text", "document"),
        metadata_fields=("title", "company", "cik", "filing_type", "date"),
        weight=1.0,
    ),
    DatasetSource(
        domain="gov_reports",
        path="ccdv/govreport-summarization",
        split="train",
        text_fields=("report", "text", "content", "document"),
        metadata_fields=("id", "summary"),
        weight=1.0,
    ),
    DatasetSource(
        domain="science_pdfs",
        path="JoaoCoelho/scientific_papers_citation_scores",
        split="train",
        text_fields=("text",),
        metadata_fields=("pdf_url", "title", "doi", "citation_count", "local_pdf_path"),
        weight=0.5,
        source_type="pdf_cache",
    ),
    DatasetSource(
        domain="gov_spreadsheets",
        path="data.gov.uk/spreadsheets",
        split="manifest",
        text_fields=("text",),
        metadata_fields=("file_url", "source_page", "local_path", "content_type", "size_bytes"),
        weight=0.4,
        source_type="spreadsheet_cache",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/generated/eval_dataset.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=240,
        help="Total number of documents to include across all sources.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Max characters kept per document.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Minimum characters required after extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel workers for loading/sampling sources.",
    )
    parser.add_argument(
        "--sample-pool-multiplier",
        type=int,
        default=40,
        help="Rows scanned per source ~= quota * multiplier (lower is faster).",
    )
    parser.add_argument(
        "--sample-pool-min",
        type=int,
        default=800,
        help="Minimum rows scanned per source (lower is faster).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="Optional domain filter. Example: --sources legal news code",
    )
    parser.add_argument(
        "--commoncrawl-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing predownloaded Common Crawl JSONL cache files.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing predownloaded Hugging Face source JSONL cache files.",
    )
    parser.add_argument(
        "--pdf-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing predownloaded PDF source JSONL cache files.",
    )
    parser.add_argument(
        "--spreadsheet-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing predownloaded spreadsheet manifest/files.",
    )
    return parser.parse_args()


def safe_token(value: str | None) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")
    return token.strip("_") or "none"


def hf_cache_file_for_source(cache_dir: Path, source: DatasetSource) -> Path:
    filename = (
        f"{safe_token(source.domain)}__"
        f"{safe_token(source.path)}__"
        f"{safe_token(source.name)}__"
        f"{safe_token(source.split)}.jsonl"
    )
    return cache_dir / filename


def extract_text(example: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def keep_metadata(example: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for field in fields:
        if field in example and isinstance(example[field], (str, int, float, bool)):
            kept[field] = example[field]
    return kept


def weighted_quotas(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        return [0] * len(weights)
    if not weights:
        return []
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("at least one source weight must be > 0")

    raw = [(weight / weight_sum) * total for weight in weights]
    quotas = [int(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (raw[idx] - quotas[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        quotas[idx] += 1
    return quotas


def effective_min_chars(source: DatasetSource, default_min_chars: int) -> int:
    # Tweets are short by nature; keep a lower floor to avoid near-empty caches.
    if source.domain == "tweets":
        return min(default_min_chars, 20)
    # XLS/XLSX entries may be binary placeholders.
    if source.domain == "gov_spreadsheets":
        return min(default_min_chars, 1)
    return default_min_chars


def sample_from_source(
    source: DatasetSource,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
) -> list[dict[str, Any]]:
    if source.source_type != "huggingface":
        raise ValueError(f"Unsupported source_type for sample_from_source: {source.source_type}")
    if quota <= 0:
        return []
    rng = random.Random(seed)
    dataset = load_dataset(source.path, source.name, split=source.split, streaming=True)
    buffered: list[dict[str, Any]] = []
    seen = 0
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    min_chars_eff = effective_min_chars(source, min_chars)

    for example in dataset:
        text = extract_text(example, source.text_fields)
        if not text:
            continue
        text = " ".join(text.split())
        if len(text) < min_chars_eff:
            continue
        item = {
            "domain": source.domain,
            "source_path": source.path,
            "source_name": source.name,
            "split": source.split,
            "doc_id": f"{source.path.replace('/', '_')}::{seed}::{seen}",
            "text": text[:max_chars],
            "metadata": keep_metadata(example, source.metadata_fields),
        }
        seen += 1
        if len(buffered) < quota:
            buffered.append(item)
        else:
            replace_idx = rng.randint(0, seen - 1)
            if replace_idx < quota:
                buffered[replace_idx] = item
        if seen >= max_seen:
            break

    return buffered


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
    text = body.decode("utf-8", errors="ignore").strip()
    return text if text else None


def commoncrawl_index_records(
    crawl: str,
    query: str,
    max_records: int,
    meaningful_only: bool = True,
    max_pages: int = 40,
    page_size: int = 2000,
    timeout_s: int = 45,
) -> list[dict[str, Any]]:
    endpoint = f"https://index.commoncrawl.org/{crawl}-index"
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for page in range(max_pages):
        params: dict[str, Any] = {
            "url": query,
            "output": "json",
            "filter": ["status:200", "mime:text/html"],
            "page": page,
            "pageSize": page_size,
        }
        resp = requests.get(endpoint, params=params, timeout=timeout_s)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        if not lines:
            break

        page_added = 0
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
            if meaningful_only and not is_meaningful_web_url(url):
                continue

            seen_urls.add(url)
            records.append(record)
            page_added += 1
            if len(records) >= max_records:
                return records

        if page_added == 0:
            # No useful records on this page; continue briefly then stop early.
            if page >= 3:
                break
    return records


def fetch_commoncrawl_html(record: dict[str, Any], timeout_s: int = 45) -> str | None:
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
    warc_url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset_i}-{end_i}"}
    resp = requests.get(warc_url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return parse_warc_html(resp.content)


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


def sample_from_commoncrawl(
    source: DatasetSource,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
    commoncrawl_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    crawl = source.name or "CC-MAIN-2024-10"
    query = source.split or "*.com/*"
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    rng = random.Random(seed)
    enforce_host_cap = source.domain != "web_html_wikipedia"
    per_host_limit = 3

    if commoncrawl_cache_dir is not None:
        cache_file = commoncrawl_cache_dir / f"{source.domain}.jsonl"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Common Crawl cache missing for domain '{source.domain}': {cache_file}"
            )
        return sample_from_commoncrawl_cache(
            source=source,
            cache_file=cache_file,
            quota=quota,
            min_chars=min_chars,
            max_chars=max_chars,
            seed=seed,
            sample_pool_multiplier=sample_pool_multiplier,
            sample_pool_min=sample_pool_min,
        )

    records = commoncrawl_index_records(
        crawl=crawl,
        query=query,
        max_records=max_seen * 4,
        meaningful_only=True,
    )
    rng.shuffle(records)

    buffered: list[dict[str, Any]] = []
    seen = 0
    host_usage: dict[str, int] = defaultdict(int)
    for record in records:
        url = str(record.get("url", "")).strip()
        if not is_meaningful_web_url(url):
            continue
        host = urlparse(url).netloc.lower()
        if enforce_host_cap and host and host_usage[host] >= per_host_limit:
            continue

        html = fetch_commoncrawl_html(record)
        if not html:
            continue
        html = " ".join(html.split())
        if len(html) < min_chars:
            continue

        item = {
            "domain": source.domain,
            "source_path": source.path,
            "source_name": source.name,
            "split": source.split,
            "doc_id": f"commoncrawl::{seed}::{seen}",
            "text": html[:max_chars],
            "metadata": {
                "url": url,
                "filename": record.get("filename"),
                "offset": record.get("offset"),
                "length": record.get("length"),
                "mime": record.get("mime"),
                "status": record.get("status"),
                "digest": record.get("digest"),
                "crawl": crawl,
            },
        }
        seen += 1
        if enforce_host_cap and host:
            host_usage[host] += 1
        if len(buffered) < quota:
            buffered.append(item)
        else:
            replace_idx = rng.randint(0, seen - 1)
            if replace_idx < quota:
                buffered[replace_idx] = item
        if seen >= max_seen:
            break
    return buffered


def sample_from_commoncrawl_cache(
    source: DatasetSource,
    cache_file: Path,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    candidates: list[dict[str, Any]] = []

    with cache_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            html = str(row.get("text", "")).strip()
            if not html:
                continue
            html = " ".join(html.split())
            if len(html) < min_chars:
                continue

            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            candidates.append(
                {
                    "text": html[:max_chars],
                    "metadata": metadata,
                }
            )

    if not candidates:
        return []

    # Shuffle cache rows so sampling is not biased by file write order.
    rng.shuffle(candidates)
    if len(candidates) > max_seen:
        candidates = candidates[:max_seen]

    buffered: list[dict[str, Any]] = []
    seen = 0
    for row in candidates:
        item = {
                "domain": source.domain,
                "source_path": source.path,
                "source_name": source.name,
                "split": source.split,
                "doc_id": f"commoncrawl-cache::{seed}::{seen}",
                "text": row["text"],
                "metadata": row["metadata"],
            }
        seen += 1
        if len(buffered) < quota:
            buffered.append(item)
        else:
            replace_idx = rng.randint(0, seen - 1)
            if replace_idx < quota:
                buffered[replace_idx] = item
    return buffered


def sample_documents_for_source(
    source: DatasetSource,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
    commoncrawl_cache_dir: Path | None = None,
    hf_cache_dir: Path | None = None,
    pdf_cache_dir: Path | None = None,
    spreadsheet_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if source.source_type == "huggingface":
        if hf_cache_dir is not None:
            cache_file = hf_cache_file_for_source(hf_cache_dir, source)
            if cache_file.exists():
                return sample_from_hf_cache(
                    source=source,
                    cache_file=cache_file,
                    quota=quota,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    seed=seed,
                    sample_pool_multiplier=sample_pool_multiplier,
                    sample_pool_min=sample_pool_min,
                )
        return sample_from_source(
            source=source,
            quota=quota,
            min_chars=min_chars,
            max_chars=max_chars,
            seed=seed,
            sample_pool_multiplier=sample_pool_multiplier,
            sample_pool_min=sample_pool_min,
        )
    if source.source_type == "commoncrawl":
        return sample_from_commoncrawl(
            source=source,
            quota=quota,
            min_chars=min_chars,
            max_chars=max_chars,
            seed=seed,
            sample_pool_multiplier=sample_pool_multiplier,
            sample_pool_min=sample_pool_min,
            commoncrawl_cache_dir=commoncrawl_cache_dir,
        )
    if source.source_type == "pdf_cache":
        if pdf_cache_dir is None:
            raise ValueError(
                f"Source '{source.domain}' requires --pdf-cache-dir with predownloaded files."
            )
        cache_file = pdf_cache_dir / f"{source.domain}.jsonl"
        if not cache_file.exists():
            raise FileNotFoundError(f"PDF cache missing for domain '{source.domain}': {cache_file}")
        return sample_from_pdf_cache(
            source=source,
            cache_file=cache_file,
            quota=quota,
            min_chars=min_chars,
            max_chars=max_chars,
            seed=seed,
            sample_pool_multiplier=sample_pool_multiplier,
            sample_pool_min=sample_pool_min,
        )
    if source.source_type == "spreadsheet_cache":
        if spreadsheet_cache_dir is None:
            raise ValueError(
                f"Source '{source.domain}' requires --spreadsheet-cache-dir with predownloaded files."
            )
        manifest_file = spreadsheet_cache_dir / "gov_spreadsheets_manifest.jsonl"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Spreadsheet cache manifest missing for domain '{source.domain}': {manifest_file}"
            )
        return sample_from_spreadsheet_cache(
            source=source,
            manifest_file=manifest_file,
            quota=quota,
            min_chars=min_chars,
            max_chars=max_chars,
            seed=seed,
            sample_pool_multiplier=sample_pool_multiplier,
            sample_pool_min=sample_pool_min,
        )
    raise ValueError(f"Unsupported source_type: {source.source_type}")


def sample_from_hf_cache(
    source: DatasetSource,
    cache_file: Path,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    rng = random.Random(seed)
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    buffered: list[dict[str, Any]] = []
    seen = 0
    min_chars_eff = effective_min_chars(source, min_chars)

    with cache_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(row.get("text", "")).strip()
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) < min_chars_eff:
                continue

            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            item = {
                "domain": source.domain,
                "source_path": source.path,
                "source_name": source.name,
                "split": source.split,
                "doc_id": f"hf-cache::{seed}::{seen}",
                "text": text[:max_chars],
                "metadata": metadata,
            }
            seen += 1
            if len(buffered) < quota:
                buffered.append(item)
            else:
                replace_idx = rng.randint(0, seen - 1)
                if replace_idx < quota:
                    buffered[replace_idx] = item
            if seen >= max_seen:
                break
    return buffered


def sample_from_pdf_cache(
    source: DatasetSource,
    cache_file: Path,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    rng = random.Random(seed)
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    candidates: list[dict[str, Any]] = []

    with cache_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pdf_base64 = row.get("pdf_base64")
            if not isinstance(pdf_base64, str) or not pdf_base64.strip():
                continue
            # Keep PDFs as raw bytes payload (base64-encoded), not extracted text.
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = {**metadata, "pdf_base64": pdf_base64, "binary_format": "pdf"}
            candidates.append({"text": "PDF_BINARY", "metadata": metadata})

    if not candidates:
        return []
    rng.shuffle(candidates)
    if len(candidates) > max_seen:
        candidates = candidates[:max_seen]

    buffered: list[dict[str, Any]] = []
    seen = 0
    for row in candidates:
        item = {
            "domain": source.domain,
            "source_path": source.path,
            "source_name": source.name,
            "split": source.split,
            "doc_id": f"pdf-cache::{seed}::{seen}",
            "text": row["text"],
            "metadata": row["metadata"],
        }
        seen += 1
        if len(buffered) < quota:
            buffered.append(item)
        else:
            replace_idx = rng.randint(0, seen - 1)
            if replace_idx < quota:
                buffered[replace_idx] = item
    return buffered


def sample_from_spreadsheet_cache(
    source: DatasetSource,
    manifest_file: Path,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    rng = random.Random(seed)
    max_seen = max(quota * sample_pool_multiplier, sample_pool_min)
    min_chars_eff = effective_min_chars(source, min_chars)
    buffered: list[dict[str, Any]] = []
    seen = 0

    with manifest_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            local_path_str = row.get("local_path")
            if not isinstance(local_path_str, str) or not local_path_str.strip():
                continue
            local_path = Path(local_path_str)
            if not local_path.exists() or not local_path.is_file():
                continue

            ext = local_path.suffix.lower()
            try:
                file_bytes = local_path.read_bytes()
            except OSError:
                continue
            if not file_bytes:
                continue

            metadata: dict[str, Any] = {
                "source_page": row.get("source_page"),
                "url": row.get("file_url"),
                "path": str(local_path),
                "filename": local_path.name,
                "mime": row.get("content_type", ""),
                "size_bytes": row.get("size_bytes", len(file_bytes)),
            }

            if ext == ".csv":
                text = file_bytes.decode("utf-8", errors="ignore")
                text = " ".join(text.split())[:max_chars]
                metadata["binary_format"] = "csv"
            elif ext in {".xls", ".xlsx"}:
                text = "SPREADSHEET_BINARY"
                metadata["binary_format"] = "excel"
                metadata["spreadsheet_base64"] = base64.b64encode(file_bytes).decode("ascii")
            else:
                continue

            if len(text) < min_chars_eff:
                continue

            item = {
                "domain": source.domain,
                "source_path": source.path,
                "source_name": source.name,
                "split": source.split,
                "doc_id": f"spreadsheet-cache::{seed}::{seen}",
                "text": text,
                "metadata": metadata,
            }
            seen += 1
            if len(buffered) < quota:
                buffered.append(item)
            else:
                replace_idx = rng.randint(0, seen - 1)
                if replace_idx < quota:
                    buffered[replace_idx] = item
            if seen >= max_seen:
                break
    return buffered


def sample_from_source_with_retry(
    source: DatasetSource,
    quota: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    sample_pool_multiplier: int,
    sample_pool_min: int,
    commoncrawl_cache_dir: Path | None = None,
    hf_cache_dir: Path | None = None,
    pdf_cache_dir: Path | None = None,
    spreadsheet_cache_dir: Path | None = None,
    retries: int = 2,
) -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return sample_documents_for_source(
                source=source,
                quota=quota,
                min_chars=min_chars,
                max_chars=max_chars,
                seed=seed + attempt,
                sample_pool_multiplier=sample_pool_multiplier,
                sample_pool_min=sample_pool_min,
                commoncrawl_cache_dir=commoncrawl_cache_dir,
                hf_cache_dir=hf_cache_dir,
                pdf_cache_dir=pdf_cache_dir,
                spreadsheet_cache_dir=spreadsheet_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            retryable = "client has been closed" in msg
            if not retryable or attempt >= retries:
                raise
            time.sleep(0.3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return []


def source_key(path: str, name: str | None, split: str) -> str:
    return f"{path}|{name or ''}|{split}"


def infer_programming_language(doc: dict[str, Any]) -> str | None:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    language = metadata.get("language")
    if isinstance(language, str) and language.strip():
        return language.strip()

    path = metadata.get("path")
    if not isinstance(path, str) or "." not in path:
        return None

    ext = Path(path).suffix.lower()
    ext_to_language = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".cc": "C++",
        ".c": "C",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".sh": "Shell",
        ".sql": "SQL",
    }
    return ext_to_language.get(ext)


def infer_document_type(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    domain = doc.get("domain", "unknown")
    domain_defaults = {
        "legal": "legal_case",
        "books": "book_excerpt",
        "patents": "patent_document",
        "news": "news_article",
        "science": "scientific_paper",
        "code": "source_code",
        "emails": "email",
        "finance_transcripts": "earnings_transcript",
        "finance_news": "financial_news",
        "podcasts": "podcast_transcript",
        "encyclopedia": "encyclopedia_entry",
        "math": "math_text",
        "tweets": "tweet",
        "web_html_random": "web_page",
        "web_html_wikipedia": "web_page",
        "web_html_wikipedia_en": "web_page",
        "sec_filings": "sec_filing",
        "gov_reports": "government_report",
        "gov_spreadsheets": "spreadsheet",
    }

    if "cpc_title" in metadata:
        return "patent_document"
    if {"from", "to"} & set(metadata):
        return "email"
    if "court" in metadata:
        return "legal_case"
    if "episode" in metadata:
        return "podcast_transcript"

    return domain_defaults.get(domain, "document")


def build_source_statistics(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        key = source_key(doc["source_path"], doc.get("source_name"), doc["split"])
        grouped[key].append(doc)

    stats: dict[str, dict[str, Any]] = {}
    for key, source_docs in grouped.items():
        lengths = [len(doc["text"]) for doc in source_docs]
        largest_idx = max(range(len(source_docs)), key=lambda i: lengths[i])
        smallest_idx = min(range(len(source_docs)), key=lambda i: lengths[i])

        natural_languages: set[str] = set()
        programming_languages: set[str] = set()
        doc_types: set[str] = set()

        for doc in source_docs:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                for field in ("language", "lang", "iso_language"):
                    value = metadata.get(field)
                    if isinstance(value, str) and value.strip():
                        natural_languages.add(value.strip())
            if doc.get("domain") == "code":
                pl = infer_programming_language(doc)
                if pl:
                    programming_languages.add(pl)
            doc_types.add(infer_document_type(doc))

        source_path, source_name, split = key.split("|", 2)
        stats[key] = {
            "source_path": source_path,
            "source_name": source_name or None,
            "split": split,
            "num_docs": len(source_docs),
            "char_count": {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths), 2),
            },
            "largest_doc": {
                "doc_id": source_docs[largest_idx]["doc_id"],
                "char_count": lengths[largest_idx],
            },
            "smallest_doc": {
                "doc_id": source_docs[smallest_idx]["doc_id"],
                "char_count": lengths[smallest_idx],
            },
            "languages": sorted(natural_languages),
            "programming_languages": sorted(programming_languages),
            "document_types": sorted(doc_types),
        }

    return dict(sorted(stats.items()))


def log_source_progress(
    phase: str,
    completed: int,
    total: int,
    source: DatasetSource,
    ok: bool,
    sampled_count: int | None = None,
    error: str | None = None,
) -> None:
    remaining = max(total - completed, 0)
    status = "ok" if ok else "failed"
    details = f"sampled={sampled_count}" if ok and sampled_count is not None else f"error={error}"
    print(
        (
            f"[{phase}] source {completed}/{total} processed; "
            f"{remaining} remaining | domain={source.domain} path={source.path} "
            f"status={status} {details}"
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.num_docs <= 0:
        raise ValueError("--num-docs must be > 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")
    if args.sample_pool_multiplier <= 0:
        raise ValueError("--sample-pool-multiplier must be > 0")
    if args.sample_pool_min <= 0:
        raise ValueError("--sample-pool-min must be > 0")

    selected_sources = [
        source for source in DEFAULT_SOURCES if not args.sources or source.domain in set(args.sources)
    ]
    if not selected_sources:
        raise ValueError("No matching sources selected. Check --sources values.")

    quotas = weighted_quotas(args.num_docs, [source.weight for source in selected_sources])
    all_docs: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    successful_sources: list[DatasetSource] = []
    resampled_copy_count = 0

    first_round_results: dict[int, list[dict[str, Any]]] = {}
    first_round_success: set[int] = set()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {}
        for idx, source in enumerate(selected_sources):
            quota = quotas[idx]
            if quota == 0:
                continue
            future = executor.submit(
                sample_from_source_with_retry,
                source=source,
                quota=quota,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                seed=args.seed + idx,
                sample_pool_multiplier=args.sample_pool_multiplier,
                sample_pool_min=args.sample_pool_min,
                commoncrawl_cache_dir=args.commoncrawl_cache_dir,
                hf_cache_dir=args.hf_cache_dir,
                pdf_cache_dir=args.pdf_cache_dir,
                spreadsheet_cache_dir=args.spreadsheet_cache_dir,
            )
            future_to_idx[future] = idx

        total_futures = len(future_to_idx)
        completed_futures = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            source = selected_sources[idx]
            completed_futures += 1
            try:
                sampled_docs = future.result()
                first_round_results[idx] = sampled_docs
                first_round_success.add(idx)
                log_source_progress(
                    phase="initial",
                    completed=completed_futures,
                    total=total_futures,
                    source=source,
                    ok=True,
                    sampled_count=len(sampled_docs),
                )
            except Exception as exc:  # noqa: BLE001
                failures[source.path] = str(exc)
                log_source_progress(
                    phase="initial",
                    completed=completed_futures,
                    total=total_futures,
                    source=source,
                    ok=False,
                    error=str(exc),
                )

    for idx in sorted(first_round_results):
        all_docs.extend(first_round_results[idx])
    successful_sources = [selected_sources[idx] for idx in sorted(first_round_success)]

    if len(all_docs) < args.num_docs and successful_sources:
        remaining = args.num_docs - len(all_docs)
        refill_quotas = weighted_quotas(remaining, [source.weight for source in successful_sources])
        refill_candidates = [
            (refill_idx, source, refill_quotas[refill_idx])
            for refill_idx, source in enumerate(successful_sources)
            if refill_quotas[refill_idx] > 0
        ]
        total_refill = len(refill_candidates)
        for pos, (refill_idx, source, quota) in enumerate(refill_candidates, start=1):
            try:
                sampled_docs = sample_from_source_with_retry(
                    source=source,
                    quota=quota,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    seed=args.seed + 10_000 + refill_idx,
                    sample_pool_multiplier=args.sample_pool_multiplier,
                    sample_pool_min=args.sample_pool_min,
                    commoncrawl_cache_dir=args.commoncrawl_cache_dir,
                    hf_cache_dir=args.hf_cache_dir,
                    pdf_cache_dir=args.pdf_cache_dir,
                    spreadsheet_cache_dir=args.spreadsheet_cache_dir,
                )
                all_docs.extend(sampled_docs)
                log_source_progress(
                    phase="refill",
                    completed=pos,
                    total=total_refill,
                    source=source,
                    ok=True,
                    sampled_count=len(sampled_docs),
                )
            except Exception as exc:  # noqa: BLE001
                existing = failures.get(source.path)
                refill_error = f"refill failed: {exc}"
                failures[source.path] = (
                    f"{existing}; {refill_error}" if existing else refill_error
                )
                log_source_progress(
                    phase="refill",
                    completed=pos,
                    total=total_refill,
                    source=source,
                    ok=False,
                    error=refill_error,
                )

    # Guarantee exact requested size: if still short due source failures/filters, repeat sampled docs.
    rng = random.Random(args.seed)
    if len(all_docs) < args.num_docs and all_docs:
        needed = args.num_docs - len(all_docs)
        for _ in range(needed):
            base_doc = rng.choice(all_docs)
            copied = {
                **base_doc,
                "metadata": {
                    **(base_doc.get("metadata", {}) if isinstance(base_doc.get("metadata"), dict) else {}),
                    "_resampled_copy": True,
                },
            }
            all_docs.append(copied)
            resampled_copy_count += 1

    if len(all_docs) > args.num_docs:
        rng.shuffle(all_docs)
        all_docs = all_docs[: args.num_docs]
    elif len(all_docs) < args.num_docs:
        raise RuntimeError(
            "Could not reach requested --num-docs because no source yielded valid documents."
        )

    # Final, stable doc IDs.
    for idx, doc in enumerate(all_docs):
        doc["doc_id"] = f"doc-{idx:06d}"

    domain_counts: dict[str, int] = defaultdict(int)
    for doc in all_docs:
        domain_counts[doc["domain"]] += 1
    source_statistics = build_source_statistics(all_docs)

    output = {
        "meta": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "num_docs": len(all_docs),
            "requested_num_docs": args.num_docs,
            "seed": args.seed,
            "max_workers": args.max_workers,
            "sample_pool_multiplier": args.sample_pool_multiplier,
            "sample_pool_min": args.sample_pool_min,
            "hf_cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
            "commoncrawl_cache_dir": (
                str(args.commoncrawl_cache_dir) if args.commoncrawl_cache_dir else None
            ),
            "pdf_cache_dir": str(args.pdf_cache_dir) if args.pdf_cache_dir else None,
            "spreadsheet_cache_dir": (
                str(args.spreadsheet_cache_dir) if args.spreadsheet_cache_dir else None
            ),
            "domain_counts": dict(sorted(domain_counts.items())),
            "source_statistics": source_statistics,
            "resampled_copy_count": resampled_copy_count,
            "sources": [
                {
                    "domain": s.domain,
                    "path": s.path,
                    "name": s.name,
                    "split": s.split,
                    "weight": s.weight,
                    "source_type": s.source_type,
                }
                for s in selected_sources
            ],
            "load_failures": failures,
        },
        "documents": all_docs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_docs)} documents to {args.output}")
    if failures:
        print(f"Some sources failed to load: {json.dumps(failures, indent=2)}")


if __name__ == "__main__":
    main()
