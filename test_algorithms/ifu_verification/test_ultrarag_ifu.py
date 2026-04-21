from __future__ import annotations

import asyncio
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
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "DMVC-IFU.pdf"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_output" / "ultrarag_ifu_pure"
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K = 5
DEFAULT_OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
_SERVER_ROOT = PROJECT_ROOT / "UltraRAG" / "servers"
_ULTRARAG_RUNTIME: dict[str, Any] | None = None


def _ensure_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports() -> None:
    ultrarag_src = PROJECT_ROOT / "UltraRAG" / "src"
    corpus_src = PROJECT_ROOT / "UltraRAG" / "servers" / "corpus" / "src"
    retriever_src = PROJECT_ROOT / "UltraRAG" / "servers" / "retriever" / "src"
    if not ultrarag_src.exists():
        raise ModuleNotFoundError("UltraRAG/src not found in project root.")
    _ensure_on_sys_path(ultrarag_src)
    _ensure_on_sys_path(corpus_src)
    _ensure_on_sys_path(retriever_src)
    _ensure_on_sys_path(PROJECT_ROOT)


_bootstrap_imports()


litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive IFU verification using UltraRAG tools without the UltraRAG UI."
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the IFU PDF.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama chat model used for final verification.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model passed to UltraRAG retriever.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many retrieved passages to use.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for UltraRAG experiment artifacts.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the parsed corpus, chunks, and vector index.")
    parser.add_argument("--backend", default="sentence_transformers", choices=["sentence_transformers", "bm25"], help="UltraRAG retriever backend.")
    return parser.parse_args()


def _normalize_ollama_base(base_url: str) -> str:
    cleaned = (base_url or "").rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_dir(output_root: Path, pdf_path: Path) -> Path:
    return output_root / _file_sha256(pdf_path)[:16]


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


def _run_async(awaitable):
    return asyncio.run(awaitable)


def _get_ultrarag_runtime() -> dict[str, Any]:
    global _ULTRARAG_RUNTIME
    if _ULTRARAG_RUNTIME is not None:
        return _ULTRARAG_RUNTIME

    corpus_module = importlib.import_module("corpus")
    retriever_module = importlib.import_module("retriever")
    retriever_app = getattr(retriever_module, "app")
    retriever_instance = getattr(retriever_module, "Retriever")(retriever_app)

    _ULTRARAG_RUNTIME = {
        "corpus_module": corpus_module,
        "retriever": retriever_instance,
    }
    return _ULTRARAG_RUNTIME


def _extract_json_block(text: str) -> dict:
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


def _preflight_check() -> None:
    required_modules = {
        "pandas": "pip install pandas",
        "chonkie": "pip install chonkie tiktoken pandas",
        "tiktoken": "pip install tiktoken",
        "sentence_transformers": "pip install sentence-transformers torch",
    }
    missing = []
    for module_name, install_hint in required_modules.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append((module_name, install_hint))

    if missing:
        details = "; ".join(f"{name} -> {hint}" for name, hint in missing)
        raise RuntimeError(
            "Missing dependencies for the pure UltraRAG pipeline: " + details
        )


def _dedupe_passages(passages: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for passage in passages:
        text = " ".join((passage or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _split_passage(passage: str) -> tuple[str, str]:
    text = " ".join((passage or "").split())
    if "Content:" in text:
        title_part, content_part = text.split("Content:", 1)
        title = title_part.replace("Title:", "").strip() or "Untitled"
        content = content_part.strip()
        return title, content
    return "Passage", text


def _summarize_passage_for_terminal(passage: str, max_chars: int = 220) -> str:
    title, content = _split_passage(passage)
    preview = content[:max_chars].rstrip()
    if len(content) > max_chars:
        preview += " ..."
    return f"{title}: {preview}"


def _verify_with_ollama(query: str, passages: list[str], model: str) -> dict:
    prepared_passages = _dedupe_passages(passages)
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
    response = litellm.completion(
        model=f"ollama/{model}" if not model.startswith("ollama/") else model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=1800,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json_block(content)
    if "notes" not in parsed:
        parsed["notes"] = ""
    return parsed


def _run_stage(label: str, func, *args, **kwargs):
    print(f"[Stage] {label}: START", flush=True)
    started = time.time()
    result = func(*args, **kwargs)
    print(f"[Stage] {label}: DONE in {time.time() - started:.2f}s", flush=True)
    return result


def _ingest_ifu(pdf_path: Path, output_root: Path, *, embed_model: str, force_rebuild: bool, backend: str) -> dict:
    runtime = _get_ultrarag_runtime()
    corpus_module = runtime["corpus_module"]
    retriever = runtime["retriever"]
    resolved_embed_model = _resolve_local_hf_snapshot(embed_model)
    work_dir = _artifact_dir(output_root, pdf_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_corpus_path = work_dir / "raw_corpus.jsonl"
    chunk_path = work_dir / "chunks.jsonl"
    embedding_path = work_dir / "embeddings.npy"
    index_path = work_dir / "ultrarag.index"
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if force_rebuild or not raw_corpus_path.exists():
        _run_stage(
            "build_text_corpus",
            _run_async,
            corpus_module.build_text_corpus(
                parse_file_path=str(pdf_path),
                text_corpus_save_path=str(raw_corpus_path),
            ),
        )
    else:
        print(f"[Stage] build_text_corpus: SKIP existing {raw_corpus_path}", flush=True)

    if force_rebuild or not chunk_path.exists():
        _run_stage(
            "chunk_documents",
            _run_async,
            corpus_module.chunk_documents(
                raw_chunk_path=str(raw_corpus_path),
                chunk_backend_configs={
                    "sentence": {"chunk_overlap": 80, "min_sentences_per_chunk": 1},
                    "recursive": {"min_characters_per_chunk": 80},
                },
                chunk_backend="sentence",
                tokenizer_or_token_counter="character",
                chunk_size=1200,
                chunk_path=str(chunk_path),
                use_title=True,
            ),
        )
    else:
        print(f"[Stage] chunk_documents: SKIP existing {chunk_path}", flush=True)

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

    _run_stage(
        "retriever_init",
        _run_async,
        retriever.retriever_init(
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
        ),
    )

    if backend != "bm25":
        if force_rebuild or not embedding_path.exists():
            _run_stage(
                "retriever_embed",
                _run_async,
                retriever.retriever_embed(
                    embedding_path=str(embedding_path),
                    overwrite=True,
                    is_multimodal=False,
                ),
            )
        else:
            print(f"[Stage] retriever_embed: SKIP existing {embedding_path}", flush=True)
        if force_rebuild or not index_path.exists():
            _run_stage(
                "retriever_index",
                _run_async,
                retriever.retriever_index(
                    embedding_path=str(embedding_path),
                    overwrite=True,
                    collection_name="",
                    corpus_path=str(chunk_path),
                ),
            )
        else:
            print(f"[Stage] retriever_index: SKIP existing {index_path}", flush=True)

    return {
        "work_dir": work_dir,
        "chunk_path": chunk_path,
        "logs_dir": logs_dir,
        "backend": backend,
        "retriever": retriever,
        "resolved_embed_model": resolved_embed_model,
    }


def _retrieve(query: str, top_k: int, retriever: Any) -> list[str]:
    result = _run_async(
        retriever.retriever_search(
            query_list=[query],
            top_k=top_k,
            query_instruction="Represent this IFU verification requirement for retrieving relevant supporting passages: ",
            collection_name="",
        )
    )
    passages = result.get("ret_psg", [])
    if not passages:
        return []
    return _dedupe_passages(passages[0])


def _write_log(logs_dir: Path, query: str, passages: list[str], verdict: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{timestamp}_query.txt"
    lines = [
        "UltraRAG IFU Experiment",
        "=" * 80,
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Query: {query}",
        "",
        "Retrieved passages:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            lines.append(f"  {index}. {passage}")
    else:
        lines.append("  - No passages retrieved.")

    lines.extend([
        "",
        "Verification result:",
        json.dumps(verdict, ensure_ascii=False, indent=2),
        "",
    ])
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def main() -> int:
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"IFU not found: {pdf_path}")

    print("=== UltraRAG IFU Setup ===")
    print(f"PDF: {pdf_path}")
    print(f"Retriever backend: {args.backend}")
    print(f"Embedding model: {args.embed_model}")
    print(f"LLM model: {args.model}")
    print(f"Output root: {output_root}")

    _preflight_check()

    ingest_start = time.time()
    artifacts = _ingest_ifu(
        pdf_path,
        output_root,
        embed_model=args.embed_model,
        force_rebuild=args.force_rebuild,
        backend=args.backend,
    )
    print(f"Resolved embedding model path: {artifacts['resolved_embed_model']}")
    print(f"Ingestion/index setup complete in {time.time() - ingest_start:.2f}s")
    print("Type a requirement/query. Use 'quit' or 'exit' to stop.")

    while True:
        query = input("\nRequirement> ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        t0 = time.time()
        passages = _run_stage(
            "retriever_search",
            _retrieve,
            query,
            args.top_k,
            artifacts["retriever"],
        )
        retrieval_time = time.time() - t0
        verdict = _run_stage(
            "ollama_verify",
            _verify_with_ollama,
            query,
            passages,
            args.model,
        )
        log_path = _write_log(artifacts["logs_dir"], query, passages, verdict)

        print("\n[Retrieved passages]")
        if passages:
            for index, passage in enumerate(passages, start=1):
                print(f"{index}. {_summarize_passage_for_terminal(passage)}")
        else:
            print("- No passages retrieved.")

        print("\n[Verification]")
        print(f"Status: {verdict.get('status', 'UNKNOWN')}")
        print(f"Verdict: {verdict.get('verdict', '-')}")
        evidence = verdict.get("supporting_chunks", [])
        if evidence:
            print("[Evidence]")
            for item in evidence:
                print(f"- {item}")
        missing = verdict.get("missing", [])
        if missing:
            print("[Missing]")
            for item in missing:
                print(f"- {item}")
        print(f"Retrieval time (s): {retrieval_time:.2f}")
        print(f"Saved log to: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
