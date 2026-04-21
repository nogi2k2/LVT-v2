from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import PyPDF2


def load_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def summarize_text(text: str, max_chars: int = 260) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."


def slugify(text: str, max_len: int = 72) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (cleaned or "item")[:max_len].rstrip("_")


def create_run_dir(output_root: Path, method_name: str, pdf_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / method_name / f"{slugify(pdf_path.stem, 48)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_page_spec(page_spec: str) -> list[int]:
    pages: list[int] = []
    for part in (page_spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start_page = int(start_str)
                end_page = int(end_str)
            except ValueError:
                continue
            step_pages = range(min(start_page, end_page), max(start_page, end_page) + 1)
            pages.extend(step_pages)
            continue
        try:
            pages.append(int(token))
        except ValueError:
            continue
    return sorted(set(pages))


def probe_toc_pages(pdf_path: Path, max_pages: int = 12) -> dict[str, Any]:
    candidates = []
    with open(pdf_path, "rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for page_index in range(min(max_pages, len(reader.pages))):
            text = reader.pages[page_index].extract_text() or ""
            normalized = normalize_whitespace(text)
            lowered = normalized.lower()
            dot_leader_count = len(re.findall(r"\.{4,}", text))
            section_ref_count = len(re.findall(r"\b\d+(?:\.\d+){0,2}\b", normalized))
            looks_like_toc = False
            reasons = []
            if lowered.startswith("contents") or "table of contents" in lowered[:160]:
                looks_like_toc = True
                reasons.append("contents header")
            if dot_leader_count >= 8:
                looks_like_toc = True
                reasons.append(f"dot leaders={dot_leader_count}")
            if section_ref_count >= 10 and dot_leader_count >= 4:
                looks_like_toc = True
                reasons.append(f"section refs={section_ref_count}")
            if looks_like_toc:
                candidates.append(
                    {
                        "page_number": page_index + 1,
                        "reason": ", ".join(reasons),
                        "preview": summarize_text(normalized, 500),
                    }
                )
    page_numbers = [item["page_number"] for item in candidates]
    return {
        "page_numbers": page_numbers,
        "toc_start_page": page_numbers[0] if page_numbers else None,
        "candidates": candidates,
    }


def dedupe_passages(passages: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for passage in passages:
        normalized = normalize_whitespace(passage)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def split_passage(passage: str) -> tuple[str, str]:
    normalized = normalize_whitespace(passage)
    if "Content:" in normalized:
        title_part, content_part = normalized.split("Content:", 1)
        title = title_part.replace("Title:", "").strip() or "Untitled"
        return title, content_part.strip()
    return "Passage", normalized


def summarize_passage_for_terminal(passage: str, max_chars: int = 220) -> str:
    title, content = split_passage(passage)
    preview = content[:max_chars].rstrip()
    if len(content) > max_chars:
        preview += " ..."
    return f"{title}: {preview}"
