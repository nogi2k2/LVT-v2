from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "Trilogy_EV300_IFU.pdf"
DEFAULT_REQUIREMENTS_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "Trilogy_EV300_requirements.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_output" / "ifu_benchmark"
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
_SERVER_ROOT = PROJECT_ROOT / "UltraRAG" / "servers"
_MCP_READY = False


def _ensure_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports() -> None:
    ultrarag_src = PROJECT_ROOT / "UltraRAG" / "src"
    if not ultrarag_src.exists():
        raise ModuleNotFoundError("UltraRAG/src not found in project root.")
    _ensure_on_sys_path(ultrarag_src)
    _ensure_on_sys_path(PROJECT_ROOT)


_bootstrap_imports()

from test_algorithms.ifu_verification.benchmark_helpers import (
    create_run_dir,
    dedupe_passages,
    load_requirements,
    summarize_passage_for_terminal,
    summarize_text,
    write_json,
    write_text,
)
from test_algorithms.ifu_verification.benchmark_helpers import ProgressTracker

_ultrarag_api = importlib.import_module("ultrarag.api")
ToolCall = _ultrarag_api.ToolCall
initialize = _ultrarag_api.initialize

litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch UltraRAG MCP benchmark for the Trilogy EV300 IFU.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the Trilogy IFU PDF.")
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS_PATH), help="Text file containing one requirement per line.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama chat model used for final verification.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model for UltraRAG retriever.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many passages to retrieve per requirement.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size used during UltraRAG chunking.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap used during UltraRAG chunking.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for benchmark outputs.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild UltraRAG raw corpus, chunks, embeddings, and index.")
    parser.add_argument("--backend", default="sentence_transformers", choices=["sentence_transformers", "bm25"], help="UltraRAG retriever backend.")
    return parser.parse_args()


def _normalize_ollama_model(model: str) -> str:
    return f"ollama/{model}" if not model.startswith("ollama/") else model


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_dir(output_root: Path, pdf_path: Path) -> Path:
    return output_root / "ultrarag_mcp_artifacts" / _file_sha256(pdf_path)[:16]


def _resolve_local_hf_snapshot(model_name: str) -> str:
    if not model_name or os.path.exists(model_name):
        return model_name
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def _ensure_mcp_initialized() -> None:
    global _MCP_READY
    if _MCP_READY:
        return
    initialize(["corpus", "retriever"], server_root=str(_SERVER_ROOT), log_level="info")
    _MCP_READY = True


def _extract_json_block(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"verdict": text}


def _time_call(func, *args, **kwargs):
    started = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - started


def _verify_with_ollama(query: str, passages: list[str], model: str) -> tuple[dict[str, Any], float, list[str]]:
    prepared_passages = dedupe_passages(passages)
    prompt = f"""
You are verifying whether an IFU satisfies a requirement.

Requirement:
{query}

Retrieved passages:
{json.dumps(prepared_passages, ensure_ascii=False, indent=2)}

Answer using only the retrieved passages.
Return JSON only:
{{
  "status": "PASS" | "FAIL" | "UNKNOWN",
  "verdict": "short verdict sentence",
  "supporting_chunks": ["verbatim supporting text copied from the passages"],
  "missing": ["missing detail if any"],
  "notes": "very short explanation"
}}
"""
    # perform the LLM call; caller may have a ProgressTracker in scope and set stages
    started = time.time()
    response = litellm.completion(
        model=_normalize_ollama_model(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=1800,
    )
    llm_seconds = time.time() - started
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json_block(content)
    if "notes" not in parsed:
        parsed["notes"] = ""
    return parsed, llm_seconds, prepared_passages


def _ingest_ifu(pdf_path: Path, output_root: Path, *, embed_model: str, force_rebuild: bool, backend: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    _ensure_mcp_initialized()
    resolved_embed_model = _resolve_local_hf_snapshot(embed_model)
    work_dir = _artifact_dir(output_root, pdf_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_corpus_path = work_dir / "raw_corpus.jsonl"
    chunk_path = work_dir / "chunks.jsonl"
    embedding_path = work_dir / "embeddings.npy"
    index_path = work_dir / "ultrarag.index"
    stage_timings: dict[str, float] = {}
    cache_hits: dict[str, bool] = {}

    if force_rebuild or not raw_corpus_path.exists():
        _, stage_timings["build_text_corpus_seconds"] = _time_call(
            ToolCall.corpus.build_text_corpus,
            parse_file_path=str(pdf_path),
            text_corpus_save_path=str(raw_corpus_path),
        )
        cache_hits["build_text_corpus"] = False
    else:
        stage_timings["build_text_corpus_seconds"] = 0.0
        cache_hits["build_text_corpus"] = True

    if force_rebuild or not chunk_path.exists():
        _, stage_timings["chunk_documents_seconds"] = _time_call(
            ToolCall.corpus.chunk_documents,
            raw_chunk_path=str(raw_corpus_path),
            chunk_backend_configs={
                "sentence": {"chunk_overlap": chunk_overlap, "min_sentences_per_chunk": 1},
                "recursive": {"min_characters_per_chunk": 80},
            },
            chunk_backend="sentence",
            tokenizer_or_token_counter="character",
            chunk_size=chunk_size,
            chunk_path=str(chunk_path),
            use_title=True,
        )
        cache_hits["chunk_documents"] = False
    else:
        stage_timings["chunk_documents_seconds"] = 0.0
        cache_hits["chunk_documents"] = True

    backend_configs = {
        "sentence_transformers": {
            "trust_remote_code": True,
            "sentence_transformers_encode": {
                "normalize_embeddings": True,
                "encode_chunk_size": 128,
                "q_prompt_name": None,
                "psg_prompt_name": None,
                "q_task": None,
                "psg_task": None,
            },
        },
        "bm25": {"lang": "en", "save_path": str(work_dir / "bm25")},
    }

    _, stage_timings["retriever_init_seconds"] = _time_call(
        ToolCall.retriever.retriever_init,
        model_name_or_path=resolved_embed_model,
        backend_configs=backend_configs,
        batch_size=8,
        corpus_path=str(chunk_path),
        gpu_ids=None,
        is_multimodal=False,
        backend=backend,
        index_backend="faiss",
        index_backend_configs={
            "faiss": {
                "index_use_gpu": False,
                "index_path": str(index_path),
                "index_chunk_size": 10000,
            }
        },
        is_demo=False,
        collection_name="",
    )

    if backend != "bm25":
        if force_rebuild or not embedding_path.exists():
            _, stage_timings["retriever_embed_seconds"] = _time_call(
                ToolCall.retriever.retriever_embed,
                embedding_path=str(embedding_path),
                overwrite=True,
                is_multimodal=False,
            )
            cache_hits["retriever_embed"] = False
        else:
            stage_timings["retriever_embed_seconds"] = 0.0
            cache_hits["retriever_embed"] = True

        if force_rebuild or not index_path.exists():
            _, stage_timings["retriever_index_seconds"] = _time_call(
                ToolCall.retriever.retriever_index,
                embedding_path=str(embedding_path),
                overwrite=True,
                collection_name="",
                corpus_path=str(chunk_path),
            )
            cache_hits["retriever_index"] = False
        else:
            stage_timings["retriever_index_seconds"] = 0.0
            cache_hits["retriever_index"] = True
    else:
        stage_timings["retriever_embed_seconds"] = 0.0
        stage_timings["retriever_index_seconds"] = 0.0
        cache_hits["retriever_embed"] = True
        cache_hits["retriever_index"] = True

    return {
        "work_dir": work_dir,
        "resolved_embed_model": resolved_embed_model,
        "stage_timings": stage_timings,
        "cache_hits": cache_hits,
        "chunk_path": str(chunk_path),
        "backend": backend,
    }


def _retrieve(query: str, top_k: int) -> tuple[list[str], float]:
    result, retrieval_seconds = _time_call(
        ToolCall.retriever.retriever_search,
        query_list=[query],
        top_k=top_k,
        query_instruction="Represent this IFU verification requirement for retrieving the narrowest supporting passages with exact evidence.",
        collection_name="",
    )
    passages = result.get("ret_psg", []) if isinstance(result, dict) else []
    if not passages:
        return [], retrieval_seconds
    return dedupe_passages(passages[0]), retrieval_seconds


def _build_text_report(summary: dict[str, Any]) -> str:
    lines = [
        "UltraRAG Trilogy IFU Benchmark (MCP)",
        "=" * 96,
        f"Run timestamp: {summary['run_timestamp']}",
        f"PDF: {summary['pdf_path']}",
        f"Requirements file: {summary['requirements_path']}",
        f"LLM model: {summary['model']}",
        f"Retriever backend: {summary['backend']}",
        f"Embedding model: {summary['embed_model']}",
        f"Resolved embedding path: {summary['resolved_embed_model']}",
        f"Requirement count: {summary['requirement_count']}",
        f"Total runtime (s): {summary['total_runtime_seconds']:.2f}",
        "",
        "[Ingestion Stages]",
    ]
    for key, value in summary['setup']['stage_timings'].items():
        cache_label = summary['setup']['cache_hits'].get(key.replace('_seconds', ''), False)
        lines.append(f"- {key}: {value:.2f}s | cached={cache_label}")
    lines.extend([
        "",
        "[Per-Requirement Summary]",
        "Idx | Status   | Total(s) | Retrieve(s) | Verify(s) | Chunks | Chars | Requirement",
        "-" * 96,
    ])
    for result in summary['results']:
        lines.append(
            f"{result['index']:>3} | {result['status']:<8} | {result['total_seconds']:<8.2f} | {result['retrieval_seconds']:<11.2f} | "
            f"{result['verification_seconds']:<9.2f} | {result['retrieved_chunk_count']:<6} | {result['retrieved_char_count']:<5} | {result['requirement']}"
        )
    for result in summary['results']:
        lines.extend([
            "",
            f"[Requirement {result['index']:02d}] {result['requirement']}",
            f"Status: {result['status']}",
            f"Verdict: {result['verdict']}",
            f"Timing breakdown (s): retrieve={result['retrieval_seconds']:.2f}, verify={result['verification_seconds']:.2f}, total={result['total_seconds']:.2f}",
            f"Retrieved chunks: {result['retrieved_chunk_count']} unique chunk(s)",
            f"Retrieved text size: {result['retrieved_char_count']} characters",
            "Retrieved passage previews:",
        ])
        previews = result.get('retrieved_passage_previews', [])
        if previews:
            for preview in previews:
                lines.append(f"- {preview}")
        else:
            lines.append("- No passages retrieved.")
        evidence = result.get('supporting_chunks', [])
        lines.append("Evidence:")
        if evidence:
            for item in evidence:
                lines.append(f"- {item}")
        else:
            lines.append("- No supporting chunks returned.")
        missing = result.get('missing', [])
        if missing:
            lines.append("Missing:")
            for item in missing:
                lines.append(f"- {item}")
        notes = result.get('notes', '')
        if notes:
            lines.append(f"Notes: {notes}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    requirements_path = Path(args.requirements).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"IFU PDF not found: {pdf_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    requirements = load_requirements(requirements_path)
    run_dir = create_run_dir(output_root, "ultrarag_trilogy_mcp", pdf_path)
    progress = ProgressTracker(run_dir)
    write_json(
        run_dir / "run_config.json",
        {
            "pdf_path": str(pdf_path),
            "requirements_path": str(requirements_path),
            "model": args.model,
            "embed_model": args.embed_model,
            "backend": args.backend,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "force_rebuild": args.force_rebuild,
            "requirement_count": len(requirements),
        },
    )

    print("=== UltraRAG Trilogy Benchmark (MCP) ===")
    print(f"PDF: {pdf_path}")
    print(f"Requirements: {requirements_path}")
    print(f"LLM model: {args.model}")
    print(f"Embedding model: {args.embed_model}")
    print(f"Output dir: {run_dir}")

    try:
        overall_started = time.time()
        progress.set_stage("ingest", "Starting UltraRAG ingestion and index build.")
        ingest_info = _ingest_ifu(
            pdf_path,
            output_root,
            embed_model=args.embed_model,
            force_rebuild=args.force_rebuild,
            backend=args.backend,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        progress.set_stage("ingest_complete", "Ingestion complete.", **{"stage_timings": ingest_info.get("stage_timings", {})})

    print(f"Resolved embedding model path: {ingest_info['resolved_embed_model']}")
    for key, value in ingest_info['stage_timings'].items():
        print(f"{key}: {value:.2f}s")

        results = []
        for index, requirement in enumerate(requirements, start=1):
            progress.set_stage("requirement", f"Running requirement {index}/{len(requirements)}.", requirement_index=index, current_requirement=requirement)
            print(f"\n[{index:02d}/{len(requirements):02d}] Running requirement")
            print(f"Requirement: {requirement}")
            try:
                progress.set_stage("retrieval", f"Retrieving top-{args.top_k} passages for requirement {index}.")
                passages, retrieval_seconds = _retrieve(requirement, args.top_k)

                # trace LLM verification call
                call_id = progress.record_llm_start("requirement_verification")
                verdict, verification_seconds, prepared_passages = _verify_with_ollama(requirement, passages, args.model)
                progress.record_llm_end(call_id, "requirement_verification", verification_seconds)

                result = {
                    "index": index,
                    "requirement": requirement,
                    "status": verdict.get('status', 'UNKNOWN'),
                    "verdict": verdict.get('verdict', ''),
                    "supporting_chunks": verdict.get('supporting_chunks', []),
                    "missing": verdict.get('missing', []),
                    "notes": verdict.get('notes', ''),
                    "retrieval_seconds": retrieval_seconds,
                    "verification_seconds": verification_seconds,
                    "total_seconds": retrieval_seconds + verification_seconds,
                    "retrieved_chunk_count": len(prepared_passages),
                    "retrieved_char_count": sum(len(item) for item in prepared_passages),
                    "retrieved_passages": prepared_passages,
                    "retrieved_passage_previews": [summarize_passage_for_terminal(item) for item in prepared_passages],
                }
            except Exception as exc:
                progress.set_stage("requirement_error", f"Requirement {index} failed: {exc}")
                result = {
                    "index": index,
                    "requirement": requirement,
                    "status": "ERROR",
                    "verdict": f"Error: {exc}",
                    "supporting_chunks": [],
                    "missing": [],
                    "notes": "",
                    "retrieval_seconds": 0.0,
                    "verification_seconds": 0.0,
                    "total_seconds": 0.0,
                    "retrieved_chunk_count": 0,
                    "retrieved_char_count": 0,
                    "retrieved_passages": [],
                    "retrieved_passage_previews": [],
                }
            results.append(result)
            write_json(run_dir / f"requirement_{index:02d}.json", result)
            write_json(run_dir / "partial_results.json", {"results": results})
            print(
                f"Result: {result['status']} | total={result['total_seconds']:.2f}s | retrieve={result['retrieval_seconds']:.2f}s | "
                f"verify={result['verification_seconds']:.2f}s | chunks={result['retrieved_chunk_count']} | chars={result['retrieved_char_count']}"
            )
            print(f"Verdict: {result['verdict']}")
            evidence = result.get('supporting_chunks', [])
            if evidence:
                print(f"Evidence: {evidence[0]}")

        summary = {
            "method": "UltraRAG MCP",
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "pdf_path": str(pdf_path),
            "requirements_path": str(requirements_path),
            "model": args.model,
            "backend": args.backend,
            "embed_model": args.embed_model,
            "resolved_embed_model": ingest_info['resolved_embed_model'],
            "requirement_count": len(requirements),
            "total_runtime_seconds": time.time() - overall_started,
            "setup": {
                "stage_timings": ingest_info['stage_timings'],
                "cache_hits": ingest_info['cache_hits'],
                "chunk_path": ingest_info['chunk_path'],
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "top_k": args.top_k,
                "artifact_dir": str(ingest_info['work_dir']),
            },
            "results": results,
        }

        write_json(run_dir / "run_summary.json", summary)
        write_text(run_dir / "run_summary.txt", _build_text_report(summary))
        progress.finish(f"Saved UltraRAG benchmark logs to {run_dir}")
        print(f"\nSaved UltraRAG benchmark logs to: {run_dir}")
        return 0
    except Exception as exc:
        # record failure and re-raise so exit code is non-zero
        try:
            progress.write_failure(exc)
        except Exception:
            pass
        raise
    finally:
        try:
            progress.close()
        except Exception:
            pass
    


if __name__ == "__main__":
    raise SystemExit(main())
