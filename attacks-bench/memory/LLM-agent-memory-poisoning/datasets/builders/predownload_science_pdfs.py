#!/usr/bin/env python3
"""Predownload a small PDF set from JoaoCoelho/scientific_papers_citation_scores."""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset


DATASET_ID = "JoaoCoelho/scientific_papers_citation_scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for cache/PDFs.")
    parser.add_argument("--num-pdfs", type=int, default=20, help="Number of PDFs to store.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout seconds.")
    parser.add_argument("--http-retries", type=int, default=4, help="HTTP retry attempts per PDF.")
    parser.add_argument(
        "--sample-pool",
        type=int,
        default=3000,
        help="How many dataset rows to scan for PDF URLs before sampling.",
    )
    return parser.parse_args()


def find_pdf_url(example: dict[str, Any]) -> str | None:
    # Canonical source for this dataset: `pdf` column contains the URL.
    value = example.get("pdf")
    if isinstance(value, str):
        value = value.strip()
        if value.lower().startswith(("http://", "https://")):
            return value
    return None


def safe_filename(i: int) -> str:
    return f"science_pdf_{i:04d}.pdf"


def http_get_with_retry(url: str, timeout: int, retries: int) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(0.8 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed with unknown error")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.output_dir / "science_pdfs_files"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "science_pdfs.jsonl"

    print(f"[science_pdfs] scanning {DATASET_ID} for PDF URLs...", flush=True)
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    candidates: list[dict[str, Any]] = []
    seen = 0
    for row in dataset:
        url = find_pdf_url(row)
        if not url:
            continue
        candidates.append(
            {
                "pdf_url": url,
                "title": row.get("title"),
                "doi": row.get("doi"),
                "citation_count": row.get("citation_count"),
            }
        )
        seen += 1
        if seen >= args.sample_pool:
            break
    if not candidates:
        raise RuntimeError("No PDF URLs found in scanned rows.")

    rng.shuffle(candidates)
    candidates = candidates[: max(args.num_pdfs * 4, args.num_pdfs)]

    kept = 0
    attempted = 0
    with cache_path.open("w", encoding="utf-8") as fh:
        for cand in candidates:
            if kept >= args.num_pdfs:
                break
            attempted += 1
            pdf_url = str(cand["pdf_url"])
            try:
                resp = http_get_with_retry(pdf_url, timeout=args.timeout, retries=args.http_retries)
                content_type = resp.headers.get("content-type", "").lower()
                if "pdf" not in content_type and ".pdf" not in pdf_url.lower():
                    continue
                pdf_bytes = resp.content
            except Exception:
                continue

            local_name = safe_filename(kept + 1)
            local_path = pdf_dir / local_name
            local_path.write_bytes(pdf_bytes)
            pdf_base64 = base64.b64encode(pdf_bytes).decode("ascii")

            row = {
                "text": "PDF_BINARY",
                "pdf_base64": pdf_base64,
                "metadata": {
                    "pdf_url": pdf_url,
                    "title": cand.get("title"),
                    "doi": cand.get("doi"),
                    "citation_count": cand.get("citation_count"),
                    "local_pdf_path": str(local_path),
                    "mime": "application/pdf",
                    "source_dataset": DATASET_ID,
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            print(f"[science_pdfs] kept {kept}/{args.num_pdfs}", flush=True)

    print(
        f"[science_pdfs] wrote {kept} PDFs to {pdf_dir} and cache {cache_path} (attempted {attempted})",
        flush=True,
    )


if __name__ == "__main__":
    main()
