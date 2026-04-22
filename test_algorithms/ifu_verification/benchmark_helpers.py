from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import PyPDF2
import threading
import time
import traceback


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


class ProgressTracker:
    """Lightweight progress tracker that writes progress.log, status.json, and failure.json.

    Methods:
    - set_stage(stage, message, **extra)
    - log_block(stage, lines, **extra)
    - record_llm_start(label) -> call_id
    - record_llm_end(call_id, label, seconds, error=None)
    - write_failure(exc)
    - finish(message)
    - close()
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.progress_log_path = run_dir / "progress.log"
        self.status_path = run_dir / "status.json"
        self.failure_path = run_dir / "failure.json"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.state: dict[str, Any] = {
            "state": "starting",
            "stage": "init",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_call_count": 0,
            "last_llm_started_at": None,
            "last_llm_finished_at": None,
            "last_message": "Benchmark created.",
        }
        self._last_logged_state = self.state["state"]
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._write_status()
        self._log_line("[Init] Progress tracker created.")
        self._log_line(f"[State] {self.state['state']} | stage={self.state['stage']}")
        self._heartbeat_thread.start()

    def _timestamp(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _log_line(self, message: str) -> None:
        line = f"[{self._timestamp()}] {message}"
        print(line, flush=True)
        with open(self.progress_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _write_status(self) -> None:
        self.state["updated_at"] = self._timestamp()
        self.status_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_state_if_changed(self) -> None:
        current_state = str(self.state.get("state", "unknown"))
        if current_state != self._last_logged_state:
            self._last_logged_state = current_state
            self._log_line(f"[State] {current_state} | stage={self.state.get('stage', 'unknown')}")

    def set_stage(self, stage: str, message: str, **extra: Any) -> None:
        with self._lock:
            if "state" not in extra and self.state.get("state") == "starting":
                self.state["state"] = "running"
            self.state["stage"] = stage
            self.state["last_message"] = message
            self.state.update(extra)
            self._write_status()
            self._log_state_if_changed()
        self._log_line(f"[{stage}] {message}")

    def log_block(self, stage: str, lines: list[str], **extra: Any) -> None:
        message = lines[0] if lines else ""
        with self._lock:
            if "state" not in extra and self.state.get("state") == "starting":
                self.state["state"] = "running"
            self.state["stage"] = stage
            self.state["last_message"] = message
            self.state.update(extra)
            self._write_status()
            self._log_state_if_changed()
        if not lines:
            self._log_line(f"[{stage}]")
            return
        self._log_line(f"[{stage}] {lines[0]}")
        for line in lines[1:]:
            self._log_line(f"[detail] {line}")

    def record_llm_start(self, label: str) -> int:
        with self._lock:
            if self.state.get("state") == "starting":
                self.state["state"] = "running"
            self.state["llm_call_count"] = int(self.state.get("llm_call_count", 0)) + 1
            call_id = self.state["llm_call_count"]
            self.state["last_llm_started_at"] = self._timestamp()
            self.state["last_message"] = f"LLM call {call_id} started: {label}"
            self._write_status()
            self._log_state_if_changed()
        self._log_line(f"[LLM {call_id:03d} START] {label}")
        return call_id

    def record_llm_end(self, call_id: int, label: str, seconds: float, *, error: str | None = None) -> None:
        with self._lock:
            self.state["last_llm_finished_at"] = self._timestamp()
            if error:
                self.state["last_message"] = f"LLM call {call_id} failed: {label}"
            else:
                self.state["last_message"] = f"LLM call {call_id} finished: {label}"
            self._write_status()
        if error:
            self._log_line(f"[LLM {call_id:03d} ERROR] {label} after {seconds:.2f}s | {error}")
        else:
            self._log_line(f"[LLM {call_id:03d} DONE] {label} in {seconds:.2f}s")

    def write_failure(self, exc: BaseException) -> None:
        payload = {
            "timestamp": self._timestamp(),
            "stage": self.state.get("stage"),
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.failure_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.set_stage("failed", f"Run failed: {exc}", state="failed")

    def finish(self, message: str) -> None:
        with self._lock:
            self.state["state"] = "completed"
            self.state["last_message"] = message
            self._write_status()
            self._log_state_if_changed()
        self._log_line(f"[Done] {message}")
        self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(30):
            with self._lock:
                state = self.state.get("state", "unknown")
                stage = self.state.get("stage", "unknown")
                llm_calls = self.state.get("llm_call_count", 0)
                message = self.state.get("last_message", "")
            self._log_line(f"[Heartbeat] state={state} stage={stage} llm_calls={llm_calls} note={message}")
