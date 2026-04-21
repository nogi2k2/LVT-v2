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
import numpy as np


THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
WORK_ROOT = PROJECT_ROOT.parent
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "DMVC-IFU.pdf"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_output" / "ultrarag_ifu_hybrid"
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_TOP_K = 5


def _ensure_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports() -> None:
    parserlib_src = WORK_ROOT / "Parser-Lib" / "src"
    retrieval_src = WORK_ROOT / "retrieval_engine" / "Retrieval-Engine" / "src"
    ultrarag_retriever_src = PROJECT_ROOT / "UltraRAG" / "servers" / "retriever" / "src"

    missing = [
        str(path)
        for path in [parserlib_src, retrieval_src, ultrarag_retriever_src]
        if not path.exists()
    ]
    if missing:
        raise ModuleNotFoundError(f"Missing required source paths: {missing}")

    _ensure_on_sys_path(parserlib_src)
    _ensure_on_sys_path(retrieval_src)
    _ensure_on_sys_path(ultrarag_retriever_src)
    _ensure_on_sys_path(PROJECT_ROOT)


_bootstrap_imports()


litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive IFU verification using Parser-Lib + custom chunking + UltraRAG FAISS search backend."
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the IFU PDF.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama chat model used for final verification.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many retrieved chunks to use.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for hybrid experiment artifacts.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild parsed markdown, chunks, embeddings, and FAISS index.")
    parser.add_argument("--artifacts-path", default=os.getenv("DOCLING_ARTIFACTS_PATH", ""), help="Optional local Docling artifacts path for Parser-Lib.")
    return parser.parse_args()


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_dir(output_root: Path, pdf_path: Path) -> Path:
    return output_root / _file_sha256(pdf_path)[:16]


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


def _verify_with_ollama(query: str, passages: list[str], model: str) -> dict:
    prompt = f"""
You are verifying whether an IFU satisfies a requirement.

Requirement:
{query}

Retrieved chunks:
{json.dumps(passages, ensure_ascii=False, indent=2)}

Answer using only the retrieved chunks.
Return JSON only:
{{
  "status": "PASS" | "FAIL" | "UNKNOWN",
  "verdict": "short verdict sentence",
  "supporting_chunks": ["verbatim supporting text copied from the chunks"],
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
    return _extract_json_block(content)


def _save_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(file_path: Path) -> list[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_rows(parsed_md_dir: Path) -> list[dict]:
    retrieval_chunker = importlib.import_module("retrieval_engine.ingestion.chunker")
    CascadingChunker = getattr(retrieval_chunker, "CascadingChunker")
    chunker = CascadingChunker()
    rows: list[dict] = []
    chunk_id = 0
    for md_path in sorted(parsed_md_dir.glob("*.md")):
        markdown_text = md_path.read_text(encoding="utf-8")
        chunks = chunker.chunk_document(markdown_text, md_path.name)
        for chunk in chunks:
            section = chunk.get("section", "General")
            body = chunk.get("text", "").strip()
            if not body:
                continue
            contents = f"Source: {md_path.name}\nSection: {section}\n\n{body}"
            rows.append(
                {
                    "id": chunk_id,
                    "doc_id": md_path.stem,
                    "title": md_path.stem,
                    "section": section,
                    "source": md_path.name,
                    "contents": contents,
                }
            )
            chunk_id += 1
    return rows


def _ingest_ifu(pdf_path: Path, output_root: Path, artifacts_path: str, force_rebuild: bool) -> dict:
    parser_lib = importlib.import_module("parser_lib")
    retrieval_embeddings = importlib.import_module("retrieval_engine.storage.embeddings")
    faiss_backend_module = importlib.import_module("index_backends.faiss_backend")
    DocumentParser = getattr(parser_lib, "DocumentParser")
    DualEmbeddingModel = getattr(retrieval_embeddings, "DualEmbeddingModel")
    FaissIndexBackend = getattr(faiss_backend_module, "FaissIndexBackend")

    work_dir = _artifact_dir(output_root, pdf_path)
    parsed_md_dir = work_dir / "parsed_md"
    corpus_path = work_dir / "chunks.jsonl"
    embedding_path = work_dir / "dense_embeddings.npy"
    index_path = work_dir / "hybrid.index"
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if force_rebuild and work_dir.exists():
        pass

    parser = DocumentParser(
        output_dir=str(parsed_md_dir),
        artifacts_path=artifacts_path or None,
    )
    if force_rebuild or not any(parsed_md_dir.glob("*.md")):
        parser.parse(str(pdf_path))

    if force_rebuild or not corpus_path.exists():
        rows = _build_rows(parsed_md_dir)
        _save_jsonl(rows, corpus_path)
    else:
        rows = _load_jsonl(corpus_path)

    if not rows:
        raise RuntimeError("No chunk rows were produced from the parsed markdown.")

    contents = [row["contents"] for row in rows]
    embedder = DualEmbeddingModel()
    if force_rebuild or not embedding_path.exists():
        vectors = embedder.encode_batch(contents, batch_size=16)
        dense = np.array([item["dense"] for item in vectors], dtype=np.float32)
        np.save(embedding_path, dense)
    else:
        dense = np.load(embedding_path)

    backend = FaissIndexBackend(
        contents=contents,
        config={
            "index_use_gpu": False,
            "index_path": str(index_path),
            "index_chunk_size": 10000,
        },
        logger=type("Logger", (), {"info": staticmethod(print), "warning": staticmethod(print), "error": staticmethod(print)})(),
        device_num=1,
    )

    if force_rebuild or not index_path.exists():
        ids = np.arange(len(contents)).astype(np.int64)
        backend.build_index(embeddings=dense, ids=ids, overwrite=True)
    else:
        backend.load_index()

    return {
        "work_dir": work_dir,
        "corpus_path": corpus_path,
        "embedder": embedder,
        "backend": backend,
        "logs_dir": logs_dir,
    }


def _retrieve(query: str, top_k: int, *, embedder: Any, backend: Any) -> list[str]:
    query_text = f"Represent this IFU verification requirement for retrieving relevant supporting chunks: {query}"
    vector = embedder.encode_batch([query_text], batch_size=1)[0]["dense"]
    results = backend.search(np.array([vector], dtype=np.float32), top_k)
    if not results:
        return []
    return results[0]


def _write_log(logs_dir: Path, query: str, passages: list[str], verdict: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{timestamp}_query.txt"
    lines = [
        "UltraRAG Hybrid IFU Experiment",
        "=" * 80,
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Query: {query}",
        "",
        "Retrieved chunks:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            lines.append(f"  {index}. {passage}")
    else:
        lines.append("  - No chunks retrieved.")
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

    print("=== UltraRAG Hybrid IFU Setup ===")
    print(f"PDF: {pdf_path}")
    print(f"LLM model: {args.model}")
    print(f"Output root: {output_root}")

    ingest_start = time.time()
    runtime = _ingest_ifu(
        pdf_path,
        output_root,
        artifacts_path=args.artifacts_path,
        force_rebuild=args.force_rebuild,
    )
    print(f"Ingestion/index setup complete in {time.time() - ingest_start:.2f}s")
    print("Type a requirement/query. Use 'quit' or 'exit' to stop.")

    while True:
        query = input("\nRequirement> ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        t0 = time.time()
        passages = _retrieve(
            query,
            args.top_k,
            embedder=runtime["embedder"],
            backend=runtime["backend"],
        )
        retrieval_time = time.time() - t0
        verdict = _verify_with_ollama(query, passages, args.model)
        log_path = _write_log(runtime["logs_dir"], query, passages, verdict)

        print("\n[Retrieved chunks]")
        if passages:
            for index, passage in enumerate(passages, start=1):
                print(f"{index}. {passage[:700]}")
        else:
            print("- No chunks retrieved.")

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
