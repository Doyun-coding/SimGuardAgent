#!/usr/bin/env python3
"""Predownload Hugging Face sources into local JSONL cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from build_eval_dataset import (  # type: ignore[import-not-found]
    DEFAULT_SOURCES,
    DatasetSource,
    extract_text,
    hf_cache_file_for_source,
    keep_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="HF cache output directory.")
    parser.add_argument(
        "--rows-per-source",
        type=int,
        default=5000,
        help="Maximum valid rows to cache per source.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum chars for cached row text.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum chars stored per cached row text.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="Optional domain filter, e.g. --sources legal news code.",
    )
    return parser.parse_args()


def effective_min_chars(source: DatasetSource, default_min_chars: int) -> int:
    # Tweets are short by nature; lower threshold prevents underfilled caches.
    if source.domain == "tweets":
        return min(default_min_chars, 20)
    return default_min_chars


def main() -> None:
    args = parse_args()
    if args.rows_per_source <= 0:
        raise ValueError("--rows-per-source must be > 0")

    selected = [
        s
        for s in DEFAULT_SOURCES
        if s.source_type == "huggingface" and (not args.sources or s.domain in set(args.sources))
    ]
    if not selected:
        raise ValueError("No matching Hugging Face sources selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in selected:
        print(f"[hf-cache] start domain={source.domain} path={source.path}", flush=True)
        out_file = hf_cache_file_for_source(args.output_dir, source)
        kept = 0
        seen = 0
        min_chars_eff = effective_min_chars(source, args.min_chars)
        print(
            f"[hf-cache] domain={source.domain} min_chars_effective={min_chars_eff}",
            flush=True,
        )
        dataset = load_dataset(source.path, source.name, split=source.split, streaming=True)
        with out_file.open("w", encoding="utf-8") as fh:
            for example in dataset:
                text = extract_text(example, source.text_fields)
                if not text:
                    continue
                text = " ".join(text.split())
                if len(text) < min_chars_eff:
                    continue
                row = {
                    "text": text[: args.max_chars],
                    "metadata": keep_metadata(example, source.metadata_fields),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                seen += 1
                if kept % 500 == 0:
                    print(
                        f"[hf-cache] domain={source.domain} kept={kept}/{args.rows_per_source}",
                        flush=True,
                    )
                if kept >= args.rows_per_source:
                    break
        print(
            f"[hf-cache] done domain={source.domain} wrote={kept} file={out_file}",
            flush=True,
        )


if __name__ == "__main__":
    main()
