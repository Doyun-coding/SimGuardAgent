#!/usr/bin/env python3
"""Annotate dataset documents with language and document format."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from langdetect import DetectorFactory, detect_langs
except Exception:  # noqa: BLE001
    DetectorFactory = None
    detect_langs = None


LANGUAGE_NAMES = {
    "ab": "Abkhaz",
    "en": "English",
    "id": "Indonesian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "th": "Thai",
    "fa": "Persian",
    "he": "Hebrew",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "el": "Greek",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input dataset JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output annotated JSON path.")
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        default=None,
        help="Optional fastText language model (.bin). If absent, uses langdetect fallback.",
    )
    parser.add_argument(
        "--min-lang-chars",
        type=int,
        default=80,
        help="Minimum cleaned text length for language detection.",
    )
    parser.add_argument(
        "--max-lang-chars",
        type=int,
        default=8000,
        help="Max chars sampled from cleaned text for language detection.",
    )
    return parser.parse_args()


def url_or_path_extension(doc: dict[str, Any]) -> str | None:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    candidates = [
        metadata.get("url"),
        metadata.get("file_url"),
        metadata.get("filename"),
        metadata.get("path"),
        metadata.get("local_path"),
    ]
    for raw in candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = raw.strip()
        if value.startswith("http://") or value.startswith("https://"):
            path = urlparse(value).path
        else:
            path = value
        suffix = Path(path).suffix.lower()
        if suffix:
            return suffix
    return None


def detect_document_format(doc: dict[str, Any]) -> tuple[str, float, str]:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    domain = str(doc.get("domain", "")).lower()
    text = str(doc.get("text", "")).lstrip()
    head = text[:5000].lower()
    mime = str(metadata.get("mime", "") or metadata.get("content_type", "")).lower()
    if "pdf" in mime:
        return ("pdf", 0.99, "mime")
    if "html" in mime:
        if looks_like_html_markup(text):
            return ("html", 0.99, "mime")
        return ("text", 0.85, "mime_content_override")
    if "csv" in mime:
        return ("csv", 0.99, "mime")
    if "excel" in mime or "spreadsheet" in mime:
        return ("excel", 0.99, "mime")
    if "json" in mime:
        return ("json", 0.99, "mime")
    if "xml" in mime:
        return ("xml", 0.99, "mime")

    ext = url_or_path_extension(doc)
    ext_map = {
        ".html": "html",
        ".htm": "html",
        ".xhtml": "html",
        ".pdf": "pdf",
        ".csv": "csv",
        ".tsv": "csv",
        ".xls": "excel",
        ".xlsx": "excel",
        ".ods": "excel",
        ".json": "json",
        ".xml": "xml",
        ".txt": "text",
        ".md": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".java": "code",
        ".cpp": "code",
        ".c": "code",
        ".go": "code",
        ".rs": "code",
    }
    if ext in ext_map:
        candidate = ext_map[ext]
        if candidate == "html" and not looks_like_html_markup(text):
            return ("text", 0.8, "extension_content_override")
        return (candidate, 0.95, "extension")

    if looks_like_email(text, metadata):
        return ("email", 0.93, "heuristic")
    if domain == "legal":
        # Legal corpora are overwhelmingly prose/statutory text; avoid false code hits.
        return ("text", 0.88, "domain_override")
    if domain == "tweets":
        return ("tweet", 0.95, "domain_override")
    if domain == "gov_spreadsheets":
        ext = url_or_path_extension(doc)
        if ext == ".csv":
            return ("csv", 0.95, "domain_extension")
        if ext in {".xls", ".xlsx", ".ods"}:
            return ("excel", 0.95, "domain_extension")
        return ("spreadsheet", 0.8, "domain_fallback")

    # Code should be detected before xml/html tag heuristics because many languages
    # legitimately contain angle brackets in includes/generics/comments.
    if looks_like_code(text):
        return ("code", 0.9, "heuristic")

    if head.startswith("%pdf-"):
        return ("pdf", 0.9, "signature")
    if "<html" in head or "<!doctype html" in head or "<body" in head:
        return ("html", 0.88, "signature")
    if head.startswith("{") or head.startswith("["):
        try:
            json.loads(text[:20000])
            return ("json", 0.85, "signature")
        except Exception:  # noqa: BLE001
            pass
    if head.startswith("<?xml") or looks_like_xml(head):
        return ("xml", 0.75, "signature")
    if looks_like_csv(text):
        return ("csv", 0.7, "heuristic")

    domain = str(doc.get("domain", ""))
    if domain == "code":
        return ("code", 0.8, "domain")
    if domain == "emails":
        return ("email", 0.8, "domain")
    return ("text", 0.6, "fallback")


def looks_like_csv(text: str) -> bool:
    sample = "\n".join(text.splitlines()[:12])
    if not sample:
        return False
    delimiters = [",", "\t", ";", "|"]
    for delimiter in delimiters:
        try:
            rows = list(csv.reader(sample.splitlines(), delimiter=delimiter))
        except Exception:  # noqa: BLE001
            continue
        if len(rows) < 3:
            continue
        widths = [len(row) for row in rows if row]
        if len(widths) >= 3 and len(set(widths)) <= 2 and max(widths) >= 3:
            return True
    return False


def looks_like_html_markup(text: str) -> bool:
    head = text[:12000]
    if "<!doctype html" in head.lower() or "<html" in head.lower() or "<body" in head.lower():
        return True
    # Require multiple tags with at least one closing tag to avoid false positives.
    open_tags = re.findall(r"<([a-zA-Z][a-zA-Z0-9:-]*)\b[^>]*>", head)
    close_tags = re.findall(r"</([a-zA-Z][a-zA-Z0-9:-]*)\s*>", head)
    return len(open_tags) >= 4 and len(close_tags) >= 2


def looks_like_code(text: str) -> bool:
    head = text[:8000]
    strong_patterns = [
        r"#\s*(include|import)\b",
        r"@\s*(interface|implementation|end)\b",
        r"\b(function|def|class)\s+[A-Za-z_][A-Za-z0-9_]*\s*[:({]",
        r"\b(public|private|protected)\s+(class|interface|enum)\b",
        r"\b[A-Za-z_][A-Za-z0-9_<>\[\]:*]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{",
        r"/\*.*?\*/",
        r"(?m)^\s*//",
    ]
    weak_patterns = [
        r"\b(return|static|void|int|float|bool|string|namespace|template|typedef|const|let|var)\b",
        r"[{}]",
        r"(==|!=|<=|>=|=>|::|->)",
    ]
    strong = sum(bool(re.search(pattern, head, flags=re.IGNORECASE | re.DOTALL)) for pattern in strong_patterns)
    weak = sum(bool(re.search(pattern, head, flags=re.IGNORECASE | re.DOTALL)) for pattern in weak_patterns)

    semicolons = head.count(";")
    braces = head.count("{") + head.count("}")
    if semicolons >= 8 and braces >= 2:
        strong += 1
    return strong >= 2 or (strong >= 1 and weak >= 2)


def looks_like_xml(head: str) -> bool:
    # Require structured xml-like evidence; a single <NAME> token is not enough.
    open_tags = re.findall(r"<([a-zA-Z_][a-zA-Z0-9:_-]*)\b[^>]*>", head)
    close_tags = re.findall(r"</([a-zA-Z_][a-zA-Z0-9:_-]*)\s*>", head)
    if not open_tags:
        return False
    if close_tags:
        return True
    self_closing = re.findall(r"<([a-zA-Z_][a-zA-Z0-9:_-]*)\b[^>]*/>", head)
    return len(self_closing) >= 2 and len(open_tags) >= 3


def looks_like_email(text: str, metadata: dict[str, Any]) -> bool:
    if {"from", "to", "subject"} & set(metadata):
        return True

    head = text[:12000]
    header_hits = 0
    header_patterns = [
        r"(?im)^\s*date:\s+.+$",
        r"(?im)^\s*from:\s+.+$",
        r"(?im)^\s*to:\s+.+$",
        r"(?im)^\s*subject:\s+.+$",
    ]
    for pattern in header_patterns:
        if re.search(pattern, head):
            header_hits += 1

    if header_hits >= 3:
        return True

    # Common promotional/newsletter email markers.
    email_markers = [
        "unsubscribe",
        "click here",
        "aol members click here",
        "this email message may be a recurring mailing",
    ]
    marker_hits = sum(marker in head.lower() for marker in email_markers)
    if marker_hits >= 2 and header_hits >= 1:
        return True

    return False


def clean_text_for_language(text: str, document_format: str) -> str:
    data = text or ""
    if document_format == "html":
        data = re.sub(r"(?is)<script.*?>.*?</script>", " ", data)
        data = re.sub(r"(?is)<style.*?>.*?</style>", " ", data)
        data = re.sub(r"(?is)<[^>]+>", " ", data)
        data = html.unescape(data)
    elif document_format in {"json", "xml"}:
        data = re.sub(r"[{}\\[\\]<>:=/_-]+", " ", data)
    return " ".join(data.split())


def make_language_detector(model_path: Path | None):
    fasttext_model = None
    if model_path is not None:
        try:
            import fasttext  # type: ignore[import-not-found]

            fasttext_model = fasttext.load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load fastText model. Install fasttext and provide valid --fasttext-model."
            ) from exc

    if DetectorFactory is not None:
        DetectorFactory.seed = 0

    def detect(text: str) -> tuple[str, str, float, str]:
        if not text:
            return ("unknown", "Unknown", 0.0, "none")
        if fasttext_model is not None:
            clean = text.replace("\n", " ")
            try:
                labels, probs = fasttext_model.predict(clean, k=1)
                label = labels[0] if labels else "__label__unknown"
                code = label.replace("__label__", "").lower()
                conf = float(probs[0]) if probs else 0.0
                return (code, LANGUAGE_NAMES.get(code, code), conf, "fasttext")
            except ValueError:
                # fasttext Python wrapper can fail with NumPy 2.x due to copy=False.
                raw = fasttext_model.f.predict(clean, 1, 0.0, "strict")
                if raw:
                    conf, label = raw[0]
                    code = str(label).replace("__label__", "").lower()
                    return (code, LANGUAGE_NAMES.get(code, code), float(conf), "fasttext_lowlevel")
                return ("unknown", "Unknown", 0.0, "fasttext_lowlevel")

        if detect_langs is None:
            return ("unknown", "Unknown", 0.0, "none")
        try:
            guesses = detect_langs(text)
        except Exception:  # noqa: BLE001
            return ("unknown", "Unknown", 0.0, "langdetect")
        if not guesses:
            return ("unknown", "Unknown", 0.0, "langdetect")
        best = guesses[0]
        code = str(best.lang).lower()
        conf = float(best.prob)
        return (code, LANGUAGE_NAMES.get(code, code), conf, "langdetect")

    return detect


def annotate_document(
    doc: dict[str, Any],
    detect_language_fn,
    min_lang_chars: int,
    max_lang_chars: int,
) -> dict[str, Any]:
    doc_format, format_conf, format_method = detect_document_format(doc)
    raw_text = str(doc.get("text", ""))
    cleaned = clean_text_for_language(raw_text, doc_format)

    if len(cleaned) < min_lang_chars:
        lang_code, lang_name, lang_conf, lang_method = ("unknown", "Unknown", 0.0, "length_threshold")
    else:
        probe = cleaned[:max_lang_chars]
        lang_code, lang_name, lang_conf, lang_method = detect_language_fn(probe)

    wiki_lang = wikipedia_language_from_doc(doc)
    if wiki_lang is not None:
        lang_code = wiki_lang
        lang_name = LANGUAGE_NAMES.get(wiki_lang, wiki_lang)
        lang_conf = 0.99
        lang_method = "wikipedia_subdomain"

    annotations = {
        "language_code": lang_code,
        "language": lang_name,
        "language_confidence": round(lang_conf, 4),
        "language_method": lang_method,
        "document_format": doc_format,
        "document_format_confidence": round(format_conf, 4),
        "document_format_method": format_method,
        "version": "v1",
    }
    out = dict(doc)
    out["annotations"] = annotations
    return out


def wikipedia_language_from_doc(doc: dict[str, Any]) -> str | None:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    url = metadata.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return None
    if not host.endswith(".wikipedia.org"):
        return None

    subdomain = host.split(".")[0]
    if not subdomain:
        return None
    # Accept common Wikipedia language codes, including hyphenated variants.
    if re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]+)?", subdomain):
        return subdomain
    return None


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise ValueError("Input file missing documents[]")

    detect_language_fn = make_language_detector(args.fasttext_model)

    annotated_docs = []
    language_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    for idx, doc in enumerate(docs, start=1):
        annotated = annotate_document(
            doc=doc,
            detect_language_fn=detect_language_fn,
            min_lang_chars=args.min_lang_chars,
            max_lang_chars=args.max_lang_chars,
        )
        anns = annotated["annotations"]
        language_counts[str(anns["language_code"])] += 1
        format_counts[str(anns["document_format"])] += 1
        annotated_docs.append(annotated)
        if idx % 200 == 0:
            print(f"[annotate] processed {idx}/{len(docs)}", flush=True)

    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["annotation_summary"] = {
        "num_docs": len(annotated_docs),
        "language_counts": dict(language_counts.most_common()),
        "document_format_counts": dict(format_counts.most_common()),
        "config": {
            "min_lang_chars": args.min_lang_chars,
            "max_lang_chars": args.max_lang_chars,
            "fasttext_model": str(args.fasttext_model) if args.fasttext_model else None,
        },
    }
    data["meta"] = meta
    data["documents"] = annotated_docs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[annotate] wrote {len(annotated_docs)} docs -> {args.output}")


if __name__ == "__main__":
    main()
