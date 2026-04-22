from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from qdrant_client import QdrantClient, models

from test_algorithms.ifu_verification.benchmark_helpers import (
    ProgressTracker,
    create_run_dir,
    dedupe_passages,
    load_requirements,
    slugify,
    summarize_passage_for_terminal,
    summarize_text,
    write_json,
    write_text,
)

DEFAULT_PDF = PROJECT_ROOT / "test_algorithms" / "data" / "test_ifu" / "Trilogy_EV300_IFU.pdf"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "test_algorithms" / "data" / "test_ifu" / "Trilogy_EV300_requirements.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "test_output"
DEFAULT_TOP_K = 8
DEFAULT_RERANK_K = 3
DEFAULT_COSINE_THRESHOLD = 0.55
DEFAULT_CHUNK_WORDS = 140
DEFAULT_CHUNK_OVERLAP = 35
DEFAULT_OLLAMA_TIMEOUT = 120
DEFAULT_LLM_EVIDENCE_K = 3
DEFAULT_LLM_EVIDENCE_CHARS = 320
HUB_ROOT = PROJECT_ROOT / ".cache" / "huggingface" / "hub"

GROUND_TRUTH_RULES: dict[int, dict[str, Any]] = {
    1: {"expected_present": True, "anchor_phrases": ["1.2.1 intended use", "continuous or intermittent positive pressure ventilation", "weighing at least 2.5 kg"]},
    2: {"expected_present": True, "anchor_phrases": ["1.2.2 environments of use", "institutional environments", "home", "non-emergency transport settings"]},
    3: {"expected_present": True, "anchor_phrases": ["1.2.3 contraindications", "using noninvasive ventilation", "avaps-ae therapy mode is contraindicated", "avaps feature is contraindicated"]},
    4: {"expected_present": True, "anchor_phrases": ["1.3 package contents", "package contents may vary based on model", "instructions for use usb flash drive"]},
    5: {"expected_present": True, "anchor_phrases": ["1.4.1 environmental", "do not operate the device in the presence of flammable gases", "do not block the cooling and intake air vents"]},
    6: {"expected_present": True, "anchor_phrases": ["1.4.2 clinical", "before placing a patient on the ventilator, perform a clinical assessment", "only the supervising physician’s orders authorize changes"]},
    7: {"expected_present": True, "anchor_phrases": ["4.7 adding oxygen", "4.7.1 high pressure oxygen", "4.7.2 low flow oxygen", "oxygen blender"]},
    8: {"expected_present": True, "anchor_phrases": ["1.4.7 cleaning and maintenance", "do not immerse the device", "cleaning and disinfection instructions"]},
    9: {"expected_present": True, "anchor_phrases": ["1.5 mri safety information", "mr unsafe", "outside the mri scan room"]},
    10: {"expected_present": True, "anchor_phrases": ["1.6 symbols glossary", "symbols used on this device and its packaging", "symbols on the device label and package label"]},
    11: {"expected_present": True, "anchor_phrases": ["2.3.4 monitored parameters pane", "the monitored parameters pane shows values while delivering therapy", "standby window"]},
    12: {"expected_present": True, "anchor_phrases": ["6.7 alarms and system messages", "high-priority system alarms", "troubleshooting suggestions"]},
    13: {"expected_present": True, "anchor_phrases": ["5.5 starting and stopping therapy", "to start therapy from standby", "to stop therapy and put the device into the standby state"]},
    14: {"expected_present": True, "anchor_phrases": ["use trilogy ev300 only with accessories intended for use with this device", "accessories guide", "patient interfaces, circuits, exhalation ports, and cables"]},
    15: {"expected_present": True, "anchor_phrases": ["11.1 overview", "11.4 detachable battery", "operate on ac (wall outlet) or dc (battery) power", "the detachable battery can power the device"]},
}


@dataclass
class ChunkRecord:
    chunk_id: str
    page_number: int
    ordinal: int
    text: str
    word_count: int


class StageTimer:
    def __init__(self) -> None:
        self._times: dict[str, float] = {}

    def time_call(self, key: str, func, *args, **kwargs):
        started = time.perf_counter()
        value = func(*args, **kwargs)
        self._times[key] = time.perf_counter() - started
        return value

    @property
    def times(self) -> dict[str, float]:
        return dict(self._times)

    def set_time(self, key: str, seconds: float) -> None:
        self._times[key] = seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyMuPDF + Qdrant Trilogy IFU benchmark.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    parser.add_argument("--cascading-chunking", action="store_true", help="Enable cascading heading-based chunking when possible")
    parser.add_argument("--ollama-model", type=str, default="phi3:latest", help="Ollama model to use for final judgement (e.g. phi3:latest)")
    parser.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT, help="Timeout in seconds for each Ollama decision request")
    parser.add_argument("--llm-evidence-k", type=int, default=DEFAULT_LLM_EVIDENCE_K, help="How many reranked passages to send to the LLM")
    parser.add_argument("--llm-evidence-chars", type=int, default=DEFAULT_LLM_EVIDENCE_CHARS, help="Max chars per evidence quote sent to the LLM")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def _configure_quiet_logging() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    for name in [
        "sentence_transformers",
        "transformers",
        "transformers.modeling_utils",
        "qdrant_client",
        "httpx",
        "urllib3",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as tf_logging

        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except Exception:
        pass


def _find_model_snapshot(model_dir_name: str) -> Path:
    model_root = HUB_ROOT / model_dir_name / "snapshots"
    if not model_root.exists():
        raise FileNotFoundError(f"Model snapshot root not found: {model_root}")
    for child in sorted(model_root.iterdir()):
        if child.is_dir():
            return child
    raise FileNotFoundError(f"No snapshot found under: {model_root}")


def _extract_pages_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    import fitz

    doc = fitz.open(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for index in range(len(doc)):
        page = doc.load_page(index)
        text = (page.get_text("text") or "").strip()
        pages.append({"page_number": index + 1, "text": text})
    return pages


def _sliding_chunks(words: list[str], chunk_words: int, overlap: int) -> list[list[str]]:
    if not words:
        return []
    if chunk_words <= 0:
        return [words]
    step = max(1, chunk_words - max(0, overlap))
    output = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            continue
        output.append(chunk)
        if start + chunk_words >= len(words):
            break
    return output


def _chunk_pages(pages: list[dict[str, Any]], chunk_words: int, overlap: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    ordinal = 0
    for page in pages:
        normalized = " ".join(page["text"].split())
        words = normalized.split()
        if not words:
            continue
        for local_index, word_chunk in enumerate(_sliding_chunks(words, chunk_words, overlap), start=1):
            chunk_text = " ".join(word_chunk).strip()
            if not chunk_text:
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=f"p{page['page_number']}_c{local_index}",
                    page_number=page["page_number"],
                    ordinal=ordinal,
                    text=chunk_text,
                    word_count=len(word_chunk),
                )
            )
            ordinal += 1
    return chunks


def _chunk_pages_cascading(pages: list[dict[str, Any]], chunk_words: int, overlap: int) -> list[ChunkRecord]:
    import re

    chunks: list[ChunkRecord] = []
    ordinal = 0
    heading_re = re.compile(r"^\s*(?:#{1,6}\s+.+|(?:\d+(?:\.\d+){0,3})\s+.+)$", flags=re.MULTILINE)
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        # find headings; if none, fallback to sliding chunks per page
        headings = list(heading_re.finditer(text))
        if not headings:
            words = " ".join(text.split()).split()
            for local_index, word_chunk in enumerate(_sliding_chunks(words, chunk_words, overlap), start=1):
                chunk_text = " ".join(word_chunk).strip()
                if not chunk_text:
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"p{page['page_number']}_c{local_index}",
                        page_number=page["page_number"],
                        ordinal=ordinal,
                        text=chunk_text,
                        word_count=len(word_chunk),
                    )
                )
                ordinal += 1
            continue

        # create chunks by heading spans
        spans = []
        for i, m in enumerate(headings):
            start = m.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            spans.append(text[start:end].strip())

        for local_index, span in enumerate(spans, start=1):
            words = " ".join(span.split()).split()
            if not words:
                continue
            # if span is long, further split to sliding chunks
            for sub_index, word_chunk in enumerate(_sliding_chunks(words, chunk_words, overlap), start=1):
                chunk_text = " ".join(word_chunk).strip()
                if not chunk_text:
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"p{page['page_number']}_c{local_index}_{sub_index}",
                        page_number=page["page_number"],
                        ordinal=ordinal,
                        text=chunk_text,
                        word_count=len(word_chunk),
                    )
                )
                ordinal += 1
    return chunks


def _write_markdown_dump(pages: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for page in pages:
        lines.append(f"## Page {page['page_number']}\n\n")
        lines.append(page["text"])
        lines.append("\n\n")
    write_text(path, "".join(lines))


def _normalize_page_text(text: str) -> str:
    return " ".join((text or "").split())


def _find_ground_truth_context(index: int, requirement: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    rule = GROUND_TRUTH_RULES.get(index, {"expected_present": True, "anchor_phrases": []})
    anchors = [item.lower() for item in rule.get("anchor_phrases", [])]
    candidates: list[dict[str, Any]] = []
    for page in pages:
        page_text = _normalize_page_text(page.get("text", ""))
        lowered = page_text.lower()
        if lowered.startswith("contents ") or "table of contents" in lowered[:160]:
            continue
        hits = [anchor for anchor in anchors if anchor in lowered]
        if not hits:
            continue
        score = len(hits)
        if requirement.lower().split("shall include information related to")[-1].strip()[:40] in lowered:
            score += 1
        candidates.append(
            {
                "page_number": page.get("page_number"),
                "matched_anchors": hits,
                "quote": summarize_text(page_text, 420),
                "score": score,
            }
        )
    candidates.sort(key=lambda item: (item["score"], -int(item["page_number"] or 0)), reverse=True)
    matches = candidates[:2]

    return {
        "requirement_index": index,
        "requirement": requirement,
        "expected_present": bool(rule.get("expected_present", True)),
        "anchor_phrases": rule.get("anchor_phrases", []),
        "matches": matches,
        "expected_pages": [item["page_number"] for item in matches],
    }


def _ground_truth_text(records: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Trilogy EV300 Ground Truth Context", ""]
    for record in records:
        lines.append(f"Requirement #{record['requirement_index']:03d}")
        lines.append(record["requirement"])
        lines.append(f"Expected present: {record['expected_present']}")
        lines.append(f"Expected pages: {record.get('expected_pages', [])}")
        lines.append("Reference context:")
        if record.get("matches"):
            for match in record["matches"]:
                lines.append(f"- page={match['page_number']} anchors={', '.join(match.get('matched_anchors', []))}")
                lines.append(f"  quote: {match['quote']}")
        else:
            lines.append("- No deterministic match found")
        lines.append("")
    return "\n".join(lines)


def _compare_results_to_ground_truth(results: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    gt_by_index = {item["requirement_index"]: item for item in ground_truth}
    comparisons = []
    verdict_matches = 0
    page_matches = 0
    for result in results:
        gt = gt_by_index.get(result["index"], {})
        expected_present = bool(gt.get("expected_present", False))
        expected_pages = set(gt.get("expected_pages", []))
        actual_pages = {
            item.get("page_number")
            for item in (result.get("evidence") or [])
            if item.get("page_number") is not None
        }
        verdict_match = bool(result.get("passed")) == expected_present
        page_overlap = sorted(expected_pages.intersection(actual_pages))
        if verdict_match:
            verdict_matches += 1
        if page_overlap:
            page_matches += 1
        comparisons.append(
            {
                "index": result["index"],
                "requirement": result["requirement"],
                "expected_present": expected_present,
                "expected_pages": sorted(expected_pages),
                "actual_passed": bool(result.get("passed")),
                "actual_pages": sorted(actual_pages),
                "verdict_match": verdict_match,
                "page_overlap": page_overlap,
            }
        )
    total = len(results) or 1
    return {
        "verdict_accuracy": verdict_matches / total,
        "page_overlap_accuracy": page_matches / total,
        "comparisons": comparisons,
    }


def _comparison_text(comparison: dict[str, Any]) -> str:
    lines = [
        "Ground Truth Comparison",
        "",
        f"Verdict accuracy: {comparison['verdict_accuracy']:.2%}",
        f"Page overlap accuracy: {comparison['page_overlap_accuracy']:.2%}",
        "",
    ]
    for item in comparison["comparisons"]:
        lines.append(f"Requirement #{item['index']:03d}")
        lines.append(f"- expected_present={item['expected_present']} actual_passed={item['actual_passed']} verdict_match={item['verdict_match']}")
        lines.append(f"- expected_pages={item['expected_pages']} actual_pages={item['actual_pages']} overlap={item['page_overlap']}")
        lines.append(f"- {item['requirement']}")
        lines.append("")
    return "\n".join(lines)


def _load_embedding_models(embed_path: Path, cross_path: Path):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed_model = SentenceTransformer(str(embed_path), local_files_only=True)
    cross_encoder = CrossEncoder(str(cross_path), local_files_only=True)
    return embed_model, cross_encoder


def _embed_texts(model, texts: list[str]):
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return vectors


def _call_ollama(model: str, prompt: str, timeout: int = DEFAULT_OLLAMA_TIMEOUT) -> dict:
    import json
    import subprocess

    # Prefer the local HTTP API because it is more reliable than shell parsing on Windows.
    try:
        import urllib.request

        url = "http://localhost:11434/api/generate"
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 220},
            }
        )
        req = urllib.request.Request(url, data=body.encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
                    return {"output": parsed["response"], "raw_response": parsed}
                return parsed
            except Exception:
                return {"output": text}
    except Exception as http_exc:
        http_error = str(http_exc)

    # Fallback to CLI only if the HTTP path fails outright.
    try:
        proc = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0 and proc.stdout:
            return {"output": proc.stdout.strip()}
        return {"error": proc.stderr.strip() or f"ollama run exited with code {proc.returncode}", "http_error": http_error}
    except Exception as cli_exc:
        return {"error": str(cli_exc), "http_error": http_error}


def _llm_decide(
    requirement: str,
    reranked: list[dict[str, Any]],
    model: str,
    timeout: int,
    evidence_k: int,
    evidence_chars: int,
) -> dict:
    import json, re

    def _parse_llm_json(resp: dict | str) -> dict:
        text = ""
        if isinstance(resp, dict):
            if "output" in resp and isinstance(resp["output"], str):
                text = resp["output"]
            elif "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                c = resp["choices"][0]
                text = c.get("text") or c.get("message") or json.dumps(c)
            else:
                text = json.dumps(resp)
        else:
            text = str(resp)

        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not m:
            return {"verdict": "FAIL", "confidence": 0.0, "reason": "LLM did not return JSON", "raw": text}
        try:
            parsed = json.loads(m.group(1))
            if not isinstance(parsed, dict):
                return {"verdict": "FAIL", "confidence": 0.0, "reason": "LLM returned non-object JSON", "raw": parsed}
            parsed.setdefault("verdict", "FAIL")
            parsed.setdefault("confidence", 0.0)
            parsed.setdefault("reason", "")
            parsed.setdefault("evidence", [])
            return parsed
        except Exception:
            return {"verdict": "FAIL", "confidence": 0.0, "reason": "Could not parse LLM JSON", "raw": m.group(1)}

    # Build prompt with top evidence
    pieces = []
    for item in reranked[: max(1, evidence_k)]:
        txt = summarize_text(item.get("text", ""), max(120, evidence_chars))
        pieces.append(
            f"- chunk_id={item.get('chunk_id')} page={item.get('page_number')} cosine={item.get('cosine_score', 0.0):.3f} cross={item.get('cross_score', 0.0):.3f}\n{txt}"
        )

    prompt_lines = [
        "You are an assistant that judges whether a requirement is satisfied by provided evidence.",
        "Return ONLY one compact JSON object with keys: verdict ('PASS' or 'FAIL'), confidence (0.0-1.0), reason (short), evidence (list).",
        "Use at most 2 evidence items. Each evidence item must include: chunk_id, page_number, quote (<=160 chars), cosine, cross_score.",
        "Mark PASS when the requirement is clearly covered by the evidence, even if wording differs.",
        "Do not restate the full requirement. Keep the answer brief.",
        "",
        "Requirement:",
        requirement,
        "",
        "Top evidence chunks:",
    ]
    prompt_lines.extend(pieces)
    prompt_lines.append("")
    prompt_lines.append("Decision:")
    prompt = "\n".join(prompt_lines)

    resp = _call_ollama(model, prompt, timeout=timeout)
    parsed = _parse_llm_json(resp)
    if parsed.get("verdict") in {"PASS", "FAIL"} and parsed.get("reason") != "LLM did not return JSON":
        return parsed

    retry_prompt = "\n".join(
        [
            "Return only compact JSON: {\"verdict\":\"PASS|FAIL\",\"confidence\":0.0,\"reason\":\"short\",\"evidence\":[{\"chunk_id\":\"...\",\"page_number\":1,\"quote\":\"short\",\"cosine\":0.0,\"cross_score\":0.0}]}",
            f"Requirement: {requirement}",
            f"Evidence chunk: {pieces[0] if pieces else 'none'}",
        ]
    )
    retry_resp = _call_ollama(model, retry_prompt, timeout=max(30, timeout // 2))
    retry_parsed = _parse_llm_json(retry_resp)
    if retry_parsed.get("reason") == "LLM did not return JSON" and isinstance(retry_resp, dict):
        retry_parsed["error"] = retry_resp.get("error") or retry_resp.get("http_error")
    return retry_parsed


def _create_qdrant_client(storage_path: Path) -> QdrantClient:
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def _rebuild_collection(
    client: QdrantClient,
    collection_name: str,
    chunks: list[ChunkRecord],
    vectors,
) -> None:
    existing = {item.name for item in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
    )
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "word_count": chunk.word_count,
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points, wait=True)


def _search_requirement(
    client: QdrantClient,
    collection_name: str,
    query_vector,
    top_k: int,
) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    hits = response.points
    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "ordinal": payload.get("ordinal"),
                "text": payload.get("text", ""),
                "word_count": payload.get("word_count"),
                "cosine_score": float(hit.score),
            }
        )
    return results


def _rerank_requirement(cross_encoder, requirement: str, hits: list[dict[str, Any]], rerank_k: int) -> list[dict[str, Any]]:
    if not hits:
        return []
    pairs = [(requirement, item["text"]) for item in hits]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    reranked = []
    for item, cross_score in zip(hits, scores):
        enriched = dict(item)
        enriched["cross_score"] = float(cross_score)
        reranked.append(enriched)
    reranked.sort(key=lambda item: (item["cross_score"], item["cosine_score"]), reverse=True)
    return reranked[:rerank_k]


def _decide_pass_fail(best_hit: dict[str, Any] | None, threshold: float) -> tuple[bool, str]:
    if not best_hit:
        return False, "no evidence retrieved"
    if best_hit["cosine_score"] >= threshold:
        return True, f"top cosine {best_hit['cosine_score']:.3f} >= {threshold:.2f}"
    return False, f"top cosine {best_hit['cosine_score']:.3f} < {threshold:.2f}"


def _build_terminal_summary(result: dict[str, Any]) -> str:
    evidence = result.get("evidence", [])
    if evidence:
        best = evidence[0]
        preview = summarize_passage_for_terminal(best.get("quote") or best.get("text", ""), 160)
        return (
            f"req={result['index']:03d} status={'PASS' if result['passed'] else 'FAIL'} "
            f"conf={result.get('confidence', 0.0):.2f} page={best.get('page_number')} "
            f"cosine={best.get('cosine', best.get('cosine_score', 0.0)):.3f} "
            f"cross={best.get('cross_score', 0.0):.3f} | {preview}"
        )
    return f"req={result['index']:03d} status=FAIL | no evidence"


def _build_requirement_block(result: dict[str, Any], ground_truth: dict[str, Any] | None = None) -> list[str]:
    evidence = result.get("evidence") or []
    lines = [
        f"Requirement #{result['index']:03d}",
        "=" * 88,
        f"Requirement Text:",
        result.get("requirement", ""),
        "",
        f"LLM Verdict: {'PASS' if result.get('passed') else 'FAIL'}",
        f"Confidence: {float(result.get('confidence', 0.0)):.2f}",
        f"Decision Reason: {result.get('decision_reason', '')}",
        f"Top Cosine: {float(result.get('top_cosine_score', 0.0)):.3f}",
        f"Top Cross: {float(result.get('top_cross_score', 0.0)):.3f}",
    ]
    if ground_truth:
        lines.extend(
            [
                "",
                f"Ground Truth Expected Present: {ground_truth.get('expected_present')}",
                f"Ground Truth Expected Pages: {ground_truth.get('expected_pages', [])}",
            ]
        )
    lines.append("")
    lines.append("Evidence:")
    if evidence:
        for item in evidence:
            cid = item.get("chunk_id")
            page_number = item.get("page_number")
            cosine_score = float(item.get("cosine", item.get("cosine_score", 0.0)))
            cross_score = float(item.get("cross_score", 0.0))
            quote = item.get("quote") or summarize_text(item.get("text", ""), 220)
            lines.append(f"- chunk={cid} | page={page_number} | cosine={cosine_score:.3f} | cross={cross_score:.3f}")
            lines.append(f"  quote:")
            lines.append(f"  {quote}")
            lines.append("")
    else:
        lines.append("- No evidence available")
        lines.append("")
    return lines


def _write_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "PyMuPDF + Qdrant Trilogy IFU Benchmark",
        "",
        f"Run timestamp: {summary['run_timestamp']}",
        f"PDF: {summary['pdf']}",
        f"Requirements: {summary['requirements_path']}",
        f"Embedding model: {summary['models']['embedding']}",
        f"Cross encoder: {summary['models']['cross_encoder']}",
        f"Qdrant path: {summary['artifacts']['qdrant_path']}",
        f"Markdown dump: {summary['artifacts']['markdown_dump']}",
        f"Chunks indexed: {summary['stats']['chunks_indexed']}",
        f"Requirements checked: {summary['stats']['requirements_checked']}",
        f"Pass count: {summary['stats']['pass_count']}",
        f"Fail count: {summary['stats']['fail_count']}",
        "",
        "[Stage Timings]",
    ]
    for key, value in summary["stage_timings"].items():
        lines.append(f"- {key}: {value:.2f}s")
    lines.append("")
    lines.append("[Requirement Results]")
    for item in summary["results"]:
        lines.append(
            f"- #{item['index']:03d} {'PASS' if item['passed'] else 'FAIL'} | "
            f"conf={item.get('confidence', 0.0):.2f} | score={item['top_cosine_score']:.3f} | {item['decision_reason']} | "
            f"{item['requirement']}"
        )
    return "\n".join(lines)


def main() -> None:
    _configure_quiet_logging()
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    requirements_path = Path(args.requirements).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = create_run_dir(output_root, "pymupdf_qdrant_trilogy", pdf_path)
    progress = ProgressTracker(run_dir)
    timer = StageTimer()
    client = None

    try:
        progress.set_stage("startup", "Benchmark starting.")
        print("=== PyMuPDF + Qdrant Trilogy Benchmark ===", flush=True)

        embed_snapshot = _find_model_snapshot("models--sentence-transformers--all-MiniLM-L6-v2")
        cross_snapshot = _find_model_snapshot("models--cross-encoder--ms-marco-MiniLM-L-6-v2")
        progress.set_stage("models", "Resolved local embedding model snapshots.")

        pages = timer.time_call("parse_pdf_seconds", _extract_pages_with_pymupdf, pdf_path)
        markdown_dump = run_dir / f"{pdf_path.stem}_pymupdf.md"
        _write_markdown_dump(pages, markdown_dump)
        progress.set_stage("parse_complete", f"Parsed {len(pages)} pages in {timer.times['parse_pdf_seconds']:.2f}s.")

        chunk_fn = _chunk_pages_cascading if args.cascading_chunking else _chunk_pages
        chunks = timer.time_call("chunk_pages_seconds", chunk_fn, pages, args.chunk_words, args.chunk_overlap)
        progress.set_stage("chunk_complete", f"Built {len(chunks)} page-aware chunks in {timer.times['chunk_pages_seconds']:.2f}s.")

        embed_model, cross_encoder = timer.time_call("load_models_seconds", _load_embedding_models, embed_snapshot, cross_snapshot)
        progress.set_stage("models_ready", f"Loaded embedding and reranker models in {timer.times['load_models_seconds']:.2f}s.")

        chunk_vectors = timer.time_call("embed_chunks_seconds", _embed_texts, embed_model, [chunk.text for chunk in chunks])
        progress.set_stage("embed_complete", f"Embedded {len(chunks)} chunks in {timer.times['embed_chunks_seconds']:.2f}s.")

        qdrant_path = run_dir / "qdrant_store"
        collection_name = f"trilogy_{slugify(pdf_path.stem, 32)}"
        client = timer.time_call("qdrant_init_seconds", _create_qdrant_client, qdrant_path)
        progress.set_stage("qdrant_ready", f"Local Qdrant initialized in {timer.times['qdrant_init_seconds']:.2f}s.")

        timer.time_call("qdrant_index_seconds", _rebuild_collection, client, collection_name, chunks, chunk_vectors)
        progress.set_stage("index_complete", f"Indexed {len(chunks)} chunks into Qdrant in {timer.times['qdrant_index_seconds']:.2f}s.")

        requirements = load_requirements(requirements_path)
        requirement_vectors = timer.time_call("embed_requirements_seconds", _embed_texts, embed_model, requirements)
        progress.set_stage("verify_start", f"Loaded {len(requirements)} requirements; starting retrieval.")

        ground_truth_records = [_find_ground_truth_context(index, requirement, pages) for index, requirement in enumerate(requirements, start=1)]
        write_json(run_dir / "ground_truth.json", ground_truth_records)
        write_text(run_dir / "ground_truth.txt", _ground_truth_text(ground_truth_records))

        results = []
        verify_started = time.perf_counter()
        detailed_log_lines = []
        detailed_log_path = run_dir / "requirement_details.log"
        detailed_log_path.write_text("", encoding="utf-8")
        for index, requirement in enumerate(requirements, start=1):
            progress.set_stage("requirement", f"Requirement {index}/{len(requirements)} | {summarize_text(requirement, 120)}")
            hits = _search_requirement(client, collection_name, requirement_vectors[index - 1], args.top_k)
            reranked = _rerank_requirement(cross_encoder, requirement, hits, args.rerank_k)
            # Use Ollama LLM to make final PASS/FAIL judgement with pinpoint evidence
            llm_resp = None
            llm_label = f"req={index:03d} {summarize_text(requirement, 80)}"
            llm_call_id = progress.record_llm_start(llm_label)
            llm_started = time.perf_counter()
            try:
                if reranked:
                    llm_resp = _llm_decide(
                        requirement,
                        reranked,
                        args.ollama_model,
                        args.ollama_timeout,
                        args.llm_evidence_k,
                        args.llm_evidence_chars,
                    )
                else:
                    llm_resp = {"verdict": "FAIL", "confidence": 0.0, "reason": "no evidence retrieved", "evidence": []}
                progress.record_llm_end(llm_call_id, llm_label, time.perf_counter() - llm_started)
            except Exception as exc:
                llm_resp = {"verdict": "FAIL", "confidence": 0.0, "reason": f"LLM error: {exc}", "evidence": []}
                progress.record_llm_end(llm_call_id, llm_label, time.perf_counter() - llm_started, error=str(exc))

            verdict = str(llm_resp.get("verdict", "FAIL")).upper()
            passed = verdict == "PASS"
            confidence = float(llm_resp.get("confidence", 0.0) or 0.0)
            reason = llm_resp.get("reason", "")
            # allow LLM-provided evidence if present, otherwise use reranked
            evidence = llm_resp.get("evidence") or reranked
            top_cos = float(reranked[0]["cosine_score"]) if reranked else 0.0
            top_cross = float(reranked[0]["cross_score"]) if reranked else 0.0
            result = {
                "index": index,
                "requirement": requirement,
                "passed": passed,
                "confidence": confidence,
                "decision_reason": reason,
                "top_cosine_score": top_cos,
                "top_cross_score": top_cross,
                "evidence": evidence,
                "llm_raw": llm_resp,
            }
            results.append(result)
            terminal_line = _build_terminal_summary(result)
            progress.set_stage("requirement_result", terminal_line)
            ground_truth = ground_truth_records[index - 1]
            block_lines = _build_requirement_block(result, ground_truth)
            if isinstance(llm_resp, dict) and llm_resp.get("raw"):
                block_lines.extend(["LLM Raw:", summarize_text(str(llm_resp.get("raw")), 600), ""])
            if isinstance(llm_resp, dict) and llm_resp.get("error"):
                block_lines.extend([f"LLM Error: {llm_resp.get('error')}", ""])
            progress.log_block("requirement_detail", block_lines)
            block_text = "\n".join(block_lines)
            detailed_log_lines.append(block_text)
            with open(detailed_log_path, "a", encoding="utf-8") as handle:
                handle.write(block_text + "\n\n")
        timer.set_time("verify_requirements_seconds", time.perf_counter() - verify_started)
        progress.set_stage("verify_complete", f"Verified {len(requirements)} requirements in {timer.times['verify_requirements_seconds']:.2f}s.")

        comparison = _compare_results_to_ground_truth(results, ground_truth_records)
        write_json(run_dir / "ground_truth_comparison.json", comparison)
        write_text(run_dir / "ground_truth_comparison.txt", _comparison_text(comparison))

        summary = {
            "method": "PyMuPDF + local Qdrant",
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "pdf": str(pdf_path),
            "requirements_path": str(requirements_path),
            "models": {
                "embedding": str(embed_snapshot),
                "cross_encoder": str(cross_snapshot),
                "ollama_model": str(args.ollama_model),
            },
            "artifacts": {
                "run_dir": str(run_dir),
                "qdrant_path": str(qdrant_path),
                "markdown_dump": str(markdown_dump),
                "progress_log": str(run_dir / "progress.log"),
                "status": str(run_dir / "status.json"),
                "ground_truth": str(run_dir / "ground_truth.json"),
                "ground_truth_comparison": str(run_dir / "ground_truth_comparison.json"),
                "requirement_details": str(detailed_log_path),
            },
            "stats": {
                "pages_parsed": len(pages),
                "chunks_indexed": len(chunks),
                "requirements_checked": len(requirements),
                "pass_count": sum(1 for item in results if item["passed"]),
                "fail_count": sum(1 for item in results if not item["passed"]),
                "ground_truth_verdict_accuracy": comparison["verdict_accuracy"],
                "ground_truth_page_overlap_accuracy": comparison["page_overlap_accuracy"],
            },
            "stage_timings": timer.times,
            "ground_truth_comparison": comparison,
            "results": results,
        }
        write_json(run_dir / "summary.json", summary)
        write_text(run_dir / "summary.txt", _write_summary_text(summary))
        write_text(run_dir / "requirement_details.log", "\n\n".join(detailed_log_lines))
        progress.finish(f"Saved benchmark outputs to {run_dir}")
        print(f"\nSaved benchmark outputs to: {run_dir}", flush=True)
    except Exception as exc:
        progress.write_failure(exc)
        raise
    finally:
        try:
            client.close()
        except Exception:
            pass
        progress.close()


if __name__ == "__main__":
    main()
