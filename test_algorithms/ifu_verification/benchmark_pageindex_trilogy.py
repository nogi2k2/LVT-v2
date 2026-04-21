from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
import PyPDF2

THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "Trilogy_EV300_IFU.pdf"
DEFAULT_REQUIREMENTS_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "Trilogy_EV300_requirements.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_output" / "ifu_benchmark"
DEFAULT_WORKSPACE = PROJECT_ROOT / "test_output" / "pageindex_trilogy_workspace"
DEFAULT_MODEL = "ollama/qwen2.5:14b"
DEFAULT_LLM_TIMEOUT = float(os.getenv("PAGEINDEX_LLM_TIMEOUT", os.getenv("LITELLM_REQUEST_TIMEOUT", "1800")))
DEFAULT_TOC_SCAN_PAGES = 8


def _ensure_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports() -> None:
    _ensure_on_sys_path(PROJECT_ROOT)
    for candidate in (PROJECT_ROOT, PROJECT_ROOT / "PageIndex"):
        if (candidate / "pageindex" / "__init__.py").exists():
            _ensure_on_sys_path(candidate)
            return
    raise ModuleNotFoundError("PageIndex package could not be located in the workspace.")


_bootstrap_imports()

from test_algorithms.ifu_verification.benchmark_helpers import (
    create_run_dir,
    load_requirements,
    parse_page_spec,
    probe_toc_pages,
    summarize_text,
    write_json,
    write_text,
)

PageIndexClient = importlib.import_module("pageindex").PageIndexClient
page_index = importlib.import_module("pageindex.page_index").page_index
extract_json = importlib.import_module("pageindex.utils").extract_json

litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PageIndex benchmark for the Trilogy EV300 IFU.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the Trilogy IFU PDF.")
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS_PATH), help="Text file containing one requirement per line.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM/Ollama model name.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="PageIndex workspace directory.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for benchmark outputs.")
    parser.add_argument("--reindex", action="store_true", help="Force rebuilding the PageIndex cache for this PDF.")
    parser.add_argument("--toc-scan-pages", type=int, default=DEFAULT_TOC_SCAN_PAGES, help="How many initial PDF pages PageIndex should scan for TOC detection.")
    return parser.parse_args()


def _json_completion(model: str, prompt: str) -> dict[str, Any]:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=DEFAULT_LLM_TIMEOUT,
    )
    content = response.choices[0].message.content or "{}"
    parsed = extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-JSON content: {content}")
    return parsed


def _flatten_structure(nodes: list[dict[str, Any]], depth: int = 0) -> list[dict[str, Any]]:
    flattened = []
    for node in nodes or []:
        flattened.append(
            {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "summary": summarize_text(node.get("summary", ""), 320),
                "depth": depth,
            }
        )
        flattened.extend(_flatten_structure(node.get("nodes", []), depth + 1))
    return flattened


def _extract_text_blobs(value: Any, max_items: int = 10, min_length: int = 50) -> list[str]:
    blobs: list[str] = []

    def _walk(node: Any) -> None:
        if len(blobs) >= max_items:
            return
        if isinstance(node, dict):
            for item in node.values():
                _walk(item)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            text = summarize_text(node, 700)
            if len(text) >= min_length and text not in blobs:
                blobs.append(text)

    _walk(value)
    return blobs


def _read_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages = []
    with open(pdf_path, "rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append({"page": page_number, "content": page.extract_text() or ""})
    return pages


def _find_cached_doc_id(client: Any, pdf_path: Path) -> str | None:
    resolved_pdf = str(pdf_path.resolve())
    for doc_id, doc in client.documents.items():
        doc_path = doc.get("path")
        if doc_path:
            try:
                if str(Path(doc_path).resolve()) == resolved_pdf:
                    return doc_id
            except OSError:
                pass
        if doc.get("doc_name") == pdf_path.name:
            return doc_id
    return None


def _index_pdf_with_toc(client: Any, pdf_path: Path, toc_scan_pages: int) -> dict[str, Any]:
    pages = _read_pdf_pages(pdf_path)
    started = time.time()
    result = page_index(
        doc=str(pdf_path),
        model=client.model,
        toc_check_page_num=toc_scan_pages,
        if_add_node_summary="yes",
        if_add_node_text="yes",
        if_add_node_id="yes",
        if_add_doc_description="yes",
    )
    index_seconds = time.time() - started

    doc_id = str(uuid.uuid4())
    client.documents[doc_id] = {
        "id": doc_id,
        "type": "pdf",
        "path": str(pdf_path),
        "doc_name": result.get("doc_name", pdf_path.name),
        "doc_description": result.get("doc_description", ""),
        "page_count": len(pages),
        "structure": result["structure"],
        "pages": pages,
    }
    if client.workspace:
        client._save_doc(doc_id)

    return {
        "doc_id": doc_id,
        "cached": False,
        "index_seconds": index_seconds,
        "page_count": len(pages),
        "doc_description": result.get("doc_description", ""),
    }


def _get_or_index_doc(client: Any, pdf_path: Path, force_reindex: bool, toc_scan_pages: int) -> dict[str, Any]:
    cached_doc_id = None if force_reindex else _find_cached_doc_id(client, pdf_path)
    if cached_doc_id:
        doc = client.documents.get(cached_doc_id, {})
        return {
            "doc_id": cached_doc_id,
            "cached": True,
            "index_seconds": 0.0,
            "page_count": doc.get("page_count", 0),
            "doc_description": doc.get("doc_description", ""),
        }
    return _index_pdf_with_toc(client, pdf_path, toc_scan_pages)


def _select_relevant_pages(client: Any, doc_id: str, requirement: str, model: str) -> tuple[dict[str, Any], float]:
    structure = json.loads(client.get_document_structure(doc_id))
    flat_nodes = _flatten_structure(structure)
    prompt = f"""
You are helping verify whether an IFU satisfies a requirement.

Requirement:
{requirement}

Document structure (JSON list of sections):
{json.dumps(flat_nodes, ensure_ascii=False)}

Pick the 1 to 3 tightest page ranges that are most relevant for verifying the requirement.
Only use page numbers from start_index/end_index.

Return JSON only:
{{
  "pages": "12-14,18",
  "reason": "why these pages matter",
  "sections": ["section title 1", "section title 2"]
}}
"""
    started = time.time()
    selection = _json_completion(model, prompt)
    selection_seconds = time.time() - started
    pages = str(selection.get("pages", "")).strip()
    if not pages:
        raise ValueError("Page selection returned no pages.")
    selection["pages"] = pages
    return selection, selection_seconds


def _verify_requirement(client: Any, doc_id: str, requirement: str, model: str) -> dict[str, Any]:
    selection, selection_seconds = _select_relevant_pages(client, doc_id, requirement, model)

    fetch_started = time.time()
    page_payload = json.loads(client.get_page_content(doc_id, selection["pages"]))
    page_fetch_seconds = time.time() - fetch_started

    prompt = f"""
You are verifying whether an IFU satisfies a requirement.

Requirement:
{requirement}

Relevant page ranges selected from the PageIndex tree:
{selection['pages']}

Section rationale:
{selection.get('reason', '')}

Page content JSON:
{json.dumps(page_payload, ensure_ascii=False)}

Answer using only the provided page content.
Return JSON only:
{{
  "status": "PASS" | "FAIL" | "UNKNOWN",
  "verdict": "short verdict sentence",
  "supporting_chunks": ["verbatim supporting text copied from the page content"],
  "missing": ["missing detail if any"],
  "pages_used": "same or narrower pages string"
}}
"""
    verify_started = time.time()
    verdict = _json_completion(model, prompt)
    verification_seconds = time.time() - verify_started

    selected_pages = selection["pages"]
    pages_used = str(verdict.get("pages_used", selected_pages)).strip() or selected_pages
    return {
        "status": verdict.get("status", "UNKNOWN"),
        "verdict": verdict.get("verdict", ""),
        "supporting_chunks": verdict.get("supporting_chunks", []),
        "missing": verdict.get("missing", []),
        "selected_pages": selected_pages,
        "pages_used": pages_used,
        "selected_page_count": len(parse_page_spec(selected_pages)),
        "pages_used_count": len(parse_page_spec(pages_used)),
        "selection_reason": selection.get("reason", ""),
        "sections": selection.get("sections", []),
        "page_text_blobs": _extract_text_blobs(page_payload),
        "timings": {
            "selection_seconds": selection_seconds,
            "page_fetch_seconds": page_fetch_seconds,
            "verification_seconds": verification_seconds,
            "total_seconds": selection_seconds + page_fetch_seconds + verification_seconds,
        },
    }


def _build_text_report(summary: dict[str, Any]) -> str:
    lines = [
        "PageIndex Trilogy IFU Benchmark",
        "=" * 96,
        f"Run timestamp: {summary['run_timestamp']}",
        f"PDF: {summary['pdf_path']}",
        f"Requirements file: {summary['requirements_path']}",
        f"Model: {summary['model']}",
        f"Requirement count: {summary['requirement_count']}",
        f"Total runtime (s): {summary['total_runtime_seconds']:.2f}",
        "",
        "[TOC Probe]",
        f"TOC candidate pages: {summary['toc_probe'].get('page_numbers', [])}",
        f"Detected TOC start page: {summary['toc_probe'].get('toc_start_page')}",
    ]
    for candidate in summary['toc_probe'].get('candidates', []):
        lines.append(f"- Page {candidate['page_number']}: {candidate['reason']} | {candidate['preview']}")

    lines.extend([
        "",
        "[Setup]",
        f"Client init time (s): {summary['setup']['client_init_seconds']:.2f}",
        f"Index created this run: {summary['setup']['index_created']}",
        f"Index creation time (s): {summary['setup']['index_seconds']:.2f}",
        f"Document page count: {summary['setup']['page_count']}",
        f"Top-level sections: {', '.join(summary['setup']['top_sections']) if summary['setup']['top_sections'] else '-'}",
        "",
        "[Per-Requirement Summary]",
        "Idx | Status   | Total(s) | Select(s) | Fetch(s) | Verify(s) | Pages | Requirement",
        "-" * 96,
    ])

    for result in summary['results']:
        timings = result['timings']
        lines.append(
            f"{result['index']:>3} | {result['status']:<8} | {timings['total_seconds']:<8.2f} | {timings['selection_seconds']:<9.2f} | "
            f"{timings['page_fetch_seconds']:<8.2f} | {timings['verification_seconds']:<9.2f} | {result['selected_pages']:<9} | {result['requirement']}"
        )

    for result in summary['results']:
        lines.extend([
            "",
            f"[Requirement {result['index']:02d}] {result['requirement']}",
            f"Status: {result['status']}",
            f"Verdict: {result['verdict']}",
            f"Selected pages: {result['selected_pages']} ({result['selected_page_count']} page(s))",
            f"Pages used: {result['pages_used']} ({result['pages_used_count']} page(s))",
            f"Sections: {', '.join(result['sections']) if result['sections'] else '-'}",
            f"Selection reason: {result['selection_reason'] or '-'}",
            f"Timing breakdown (s): select={result['timings']['selection_seconds']:.2f}, fetch={result['timings']['page_fetch_seconds']:.2f}, verify={result['timings']['verification_seconds']:.2f}, total={result['timings']['total_seconds']:.2f}",
            "Evidence:",
        ])
        evidence = result.get('supporting_chunks', [])
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
        blobs = result.get('page_text_blobs', [])
        if blobs:
            lines.append("Retrieved text previews:")
            for blob in blobs:
                lines.append(f"- {blob}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    requirements_path = Path(args.requirements).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"IFU PDF not found: {pdf_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    requirements = load_requirements(requirements_path)
    run_dir = create_run_dir(output_root, "pageindex_trilogy", pdf_path)

    print("=== PageIndex Trilogy Benchmark ===")
    print(f"PDF: {pdf_path}")
    print(f"Requirements: {requirements_path}")
    print(f"Model: {args.model}")
    print(f"Workspace: {workspace}")
    print(f"Output dir: {run_dir}")

    toc_probe = probe_toc_pages(pdf_path, max_pages=max(args.toc_scan_pages, 10))
    print(f"TOC candidate pages: {toc_probe.get('page_numbers', [])}")
    print(f"Detected TOC start page: {toc_probe.get('toc_start_page')}")

    overall_started = time.time()
    client_started = time.time()
    client = PageIndexClient(model=args.model, retrieve_model=args.model, workspace=str(workspace))
    client_init_seconds = time.time() - client_started

    index_info = _get_or_index_doc(client, pdf_path, force_reindex=args.reindex, toc_scan_pages=args.toc_scan_pages)
    doc_id = index_info['doc_id']
    structure = json.loads(client.get_document_structure(doc_id))
    top_sections = [node.get('title', '') for node in structure[:10] if node.get('title')]

    print(f"Client init time: {client_init_seconds:.2f}s")
    print(f"Index created this run: {not index_info['cached']}")
    print(f"Index creation time: {index_info['index_seconds']:.2f}s")
    print(f"Top-level sections: {', '.join(top_sections[:8]) if top_sections else '-'}")

    results = []
    for index, requirement in enumerate(requirements, start=1):
        print(f"\n[{index:02d}/{len(requirements):02d}] Running requirement")
        print(f"Requirement: {requirement}")
        try:
            result = _verify_requirement(client, doc_id, requirement, args.model)
        except Exception as exc:
            result = {
                "status": "ERROR",
                "verdict": f"Error: {exc}",
                "supporting_chunks": [],
                "missing": [],
                "selected_pages": "-",
                "pages_used": "-",
                "selected_page_count": 0,
                "pages_used_count": 0,
                "selection_reason": "",
                "sections": [],
                "page_text_blobs": [],
                "timings": {
                    "selection_seconds": 0.0,
                    "page_fetch_seconds": 0.0,
                    "verification_seconds": 0.0,
                    "total_seconds": 0.0,
                },
            }
        result['index'] = index
        result['requirement'] = requirement
        results.append(result)
        timings = result['timings']
        print(
            f"Result: {result['status']} | total={timings['total_seconds']:.2f}s | "
            f"select={timings['selection_seconds']:.2f}s | fetch={timings['page_fetch_seconds']:.2f}s | "
            f"verify={timings['verification_seconds']:.2f}s | pages={result['selected_pages']}"
        )
        print(f"Verdict: {result['verdict']}")
        evidence = result.get('supporting_chunks', [])
        if evidence:
            print(f"Evidence: {evidence[0]}")

    summary = {
        "method": "PageIndex",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "pdf_path": str(pdf_path),
        "requirements_path": str(requirements_path),
        "model": args.model,
        "requirement_count": len(requirements),
        "total_runtime_seconds": time.time() - overall_started,
        "toc_probe": toc_probe,
        "setup": {
            "client_init_seconds": client_init_seconds,
            "index_created": not index_info['cached'],
            "index_seconds": index_info['index_seconds'],
            "page_count": index_info['page_count'],
            "doc_description": index_info.get('doc_description', ''),
            "top_sections": top_sections,
            "workspace": str(workspace),
            "toc_scan_pages": args.toc_scan_pages,
        },
        "results": results,
    }

    write_json(run_dir / "run_summary.json", summary)
    write_text(run_dir / "run_summary.txt", _build_text_report(summary))
    print(f"\nSaved PageIndex benchmark logs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
