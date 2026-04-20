import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import litellm
import PyPDF2


THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "DMVC-IFU.pdf"
DEFAULT_WORKSPACE = PROJECT_ROOT / "test_output" / "pageindex_ifu_workspace"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_output" / "ifu_experiment_logs"
DEFAULT_MODEL = "ollama/qwen2.5:14b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_LLM_TIMEOUT = float(os.getenv("PAGEINDEX_LLM_TIMEOUT", os.getenv("LITELLM_REQUEST_TIMEOUT", "1800")))
DEFAULT_SKIP_TOC_SCAN = False


def _ensure_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports() -> None:
    _ensure_on_sys_path(PROJECT_ROOT)

    pageindex_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT / "PageIndex",
    ]
    for candidate in pageindex_roots:
        if (candidate / "pageindex" / "__init__.py").exists():
            _ensure_on_sys_path(candidate)
            return

    raise ModuleNotFoundError(
        "Could not locate the PageIndex package. Keep either 'pageindex/' in the "
        "project root, or 'PageIndex/pageindex/' from the cloned repo."
    )


_bootstrap_imports()

from ifu_verifier.controller import IFUController
from ifu_verifier.ifu_verifier_window import extract_keywords
from pageindex import PageIndexClient
from pageindex.page_index import page_index
from pageindex.utils import extract_json, print_tree


os.environ.setdefault("OLLAMA_API_BASE", DEFAULT_OLLAMA_BASE)
litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive IFU experiment: compare the current LVT baseline against a local PageIndex + Ollama workflow."
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the IFU PDF.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM/Ollama model name.")
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Directory used by PageIndexClient to cache document indexes.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the comparison text logs will be written.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuilding the cached PageIndex document entry.",
    )
    return parser.parse_args()


def _flatten_structure(nodes, depth=0):
    flat_nodes = []
    for node in nodes or []:
        flat_nodes.append(
            {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "summary": (node.get("summary") or "")[:350],
                "depth": depth,
            }
        )
        flat_nodes.extend(_flatten_structure(node.get("nodes", []), depth + 1))
    return flat_nodes


def _json_completion(model: str, prompt: str) -> dict:
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


def _slugify(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return (cleaned or "requirement")[:max_len].rstrip("_")


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _extract_text_blobs(value, *, max_items: int = 12, min_length: int = 40) -> list[str]:
    blobs = []

    def _walk(node):
        if len(blobs) >= max_items:
            return
        if isinstance(node, dict):
            for item in node.values():
                _walk(item)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            text = " ".join(node.split())
            if len(text) >= min_length and text not in blobs:
                blobs.append(text)

    _walk(value)
    return blobs


def _write_comparison_log(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{_slugify(payload['requirement'])}.txt"
    log_path = output_dir / filename

    baseline = payload["baseline"]
    pageindex = payload["pageindex"]
    lines = [
        "IFU Verification Experiment Log",
        "=" * 90,
        f"Timestamp: {payload['timestamp']}",
        f"PDF: {payload['pdf_path']}",
        f"Requirement: {payload['requirement']}",
        f"Baseline keywords: {', '.join(payload['baseline_keywords']) if payload['baseline_keywords'] else '-'}",
        f"Model: {payload['model']}",
        "",
        "[Baseline Method]",
        f"Verdict: {baseline['verdict']}",
        f"Score: {baseline['score']}",
        f"Reason: {baseline['reason']}",
        f"Time (s): {baseline['time_seconds']:.2f}",
        f"Keyword hits found: {baseline['hit_count']}",
        "",
        "Baseline supporting snippets:",
    ]

    if baseline["hits"]:
        for index, hit in enumerate(baseline["hits"], start=1):
            lines.extend([
                f"  {index}. keyword={hit.get('keyword', '-')}",
                f"     page={hit.get('page', '-')}, match_type={hit.get('match_type', '-')}, section={hit.get('section', '-')}",
                f"     snippet={hit.get('snippet', '-')}",
            ])
    else:
        lines.append("  - No baseline supporting snippets found.")

    lines.extend([
        "",
        "[PageIndex + Ollama Method]",
        f"Status: {pageindex['status']}",
        f"Verdict: {pageindex['verdict']}",
        f"Time (s): {pageindex['time_seconds']:.2f}",
        f"Selected pages: {pageindex['selected_pages']}",
        f"Pages used: {pageindex['pages_used']}",
        f"Selection reason: {pageindex['selection_reason']}",
        f"Sections: {', '.join(pageindex['sections']) if pageindex['sections'] else '-'}",
        "",
        "PageIndex evidence bullets:",
    ])

    evidence = pageindex.get("evidence", [])
    if evidence:
        for item in evidence:
            lines.append(f"  - {item}")
    else:
        lines.append("  - No evidence bullets returned.")

    lines.extend([
        "",
        "PageIndex supporting text chunks:",
    ])
    chunks = pageindex.get("supporting_chunks", [])
    if chunks:
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"  {index}. {chunk}")
    else:
        lines.append("  - No supporting chunks returned.")

    page_text_blobs = pageindex.get("page_text_blobs", [])
    lines.extend([
        "",
        "PageIndex raw retrieved text blobs:",
    ])
    if page_text_blobs:
        for index, blob in enumerate(page_text_blobs, start=1):
            lines.append(f"  {index}. {blob}")
    else:
        lines.append("  - No raw retrieved text blobs extracted.")

    missing = pageindex.get("missing", [])
    if missing:
        lines.extend([
            "",
            "PageIndex missing / gaps:",
        ])
        for item in missing:
            lines.append(f"  - {item}")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _find_cached_doc_id(client: PageIndexClient, pdf_path: Path) -> str | None:
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


def _read_pdf_pages(pdf_path: Path) -> list[dict]:
    pages = []
    with open(pdf_path, 'rb') as handle:
        pdf_reader = PyPDF2.PdfReader(handle)
        for page_number, page in enumerate(pdf_reader.pages, 1):
            pages.append({'page': page_number, 'content': page.extract_text() or ''})
    return pages


def _build_page_fallback_structure(pdf_path: Path, pages: list[dict]) -> dict:
    structure = []
    for page in pages:
        text = " ".join((page.get('content') or '').split())
        summary = text[:280] if text else f"Page {page['page']} content extracted from the IFU."
        structure.append(
            {
                'title': f"Page {page['page']}",
                'node_id': str(page['page']).zfill(4),
                'start_index': page['page'],
                'end_index': page['page'],
                'summary': summary,
                'text': page.get('content') or '',
                'nodes': [],
            }
        )
    return {
        'doc_name': pdf_path.name,
        'doc_description': 'Fallback page-wise structure generated because PageIndex no-TOC parsing failed.',
        'structure': structure,
    }


def _index_pdf_skip_toc(client: PageIndexClient, pdf_path: Path) -> str:
    print("Indexing PDF with PageIndex via direct no-TOC mode. This skips TOC detection and uses the fallback path immediately.")
    pages = _read_pdf_pages(pdf_path)
    try:
        result = page_index(
            doc=str(pdf_path),
            model=client.model,
            toc_check_page_num=0,
            if_add_node_summary='yes',
            if_add_node_text='yes',
            if_add_node_id='yes',
            if_add_doc_description='yes',
        )
    except Exception as exc:
        print(f"Warning: direct no-TOC PageIndex parsing failed: {exc}")
        print("Falling back to a simple page-wise structure so the IFU experiment can continue.")
        result = _build_page_fallback_structure(pdf_path, pages)

    doc_id = str(uuid.uuid4())
    client.documents[doc_id] = {
        'id': doc_id,
        'type': 'pdf',
        'path': str(pdf_path),
        'doc_name': result.get('doc_name', ''),
        'doc_description': result.get('doc_description', ''),
        'page_count': len(pages),
        'structure': result['structure'],
        'pages': pages,
    }
    if client.workspace:
        client._save_doc(doc_id)
    print(f"Indexing complete. Document ID: {doc_id}")
    return doc_id


def _get_or_index_doc(client: PageIndexClient, pdf_path: Path, force_reindex: bool) -> str:
    cached_doc_id = None if force_reindex else _find_cached_doc_id(client, pdf_path)
    if cached_doc_id:
        print(f"Loaded cached PageIndex document: {cached_doc_id}")
        return cached_doc_id

    if DEFAULT_SKIP_TOC_SCAN:
        return _index_pdf_skip_toc(client, pdf_path)

    print("Indexing PDF with PageIndex via local Ollama. This is the expensive one-time step.")
    return client.index(str(pdf_path), mode="pdf")


def _select_relevant_pages(
    client: PageIndexClient,
    doc_id: str,
    requirement: str,
    model: str,
) -> dict:
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
    selection = _json_completion(model, prompt)
    pages = str(selection.get("pages", "")).strip()
    if not pages:
        raise ValueError("PageIndex page selection returned no pages.")
    selection["pages"] = pages
    return selection


def _verify_with_pageindex(
    client: PageIndexClient,
    doc_id: str,
    requirement: str,
    model: str,
) -> dict:
    selection = _select_relevant_pages(client, doc_id, requirement, model)
    page_payload = json.loads(client.get_page_content(doc_id, selection["pages"]))

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
  "evidence": ["short bullet evidence 1", "short bullet evidence 2"],
    "supporting_chunks": ["verbatim supporting text copied from the page content", "second verbatim supporting text if needed"],
  "missing": ["missing detail if any"],
  "pages_used": "same or narrower pages string"
}}
"""
    verdict = _json_completion(model, prompt)
    verdict["selected_pages"] = selection["pages"]
    verdict["selection_reason"] = selection.get("reason", "")
    verdict["sections"] = selection.get("sections", [])
    verdict["page_text_blobs"] = _extract_text_blobs(page_payload)
    return verdict


def _run_single_test(
    pdf_path: Path,
    requirement: str,
    client: PageIndexClient,
    doc_id: str,
    model: str,
    output_dir: Path,
) -> None:
    print("\nRunning verification")
    baseline_keywords = extract_keywords(requirement)
    print(f"Auto-derived baseline keywords: {', '.join(baseline_keywords) if baseline_keywords else '-'}")

    t0 = time.time()
    try:
        baseline_result = IFUController.verify_prd(str(pdf_path), baseline_keywords, [], progress_cb=None)
        baseline_verdict = baseline_result.get("verdict", "ERROR")
        baseline_hits = len(baseline_result.get("hits", []))
        baseline_score = baseline_result.get("score", 0.0)
        baseline_reason = baseline_result.get("reason", "")
    except Exception as exc:
        baseline_result = {"hits": [], "score": 0.0, "reason": f"Error: {exc}"}
        baseline_verdict = f"Error: {exc}"
        baseline_hits = 0
        baseline_score = 0.0
        baseline_reason = f"Error: {exc}"
    baseline_time = time.time() - t0

    t1 = time.time()
    try:
        pageindex_result = _verify_with_pageindex(client, doc_id, requirement, model)
        pageindex_status = pageindex_result.get("status", "UNKNOWN")
        pageindex_verdict = pageindex_result.get("verdict", "No verdict returned.")
    except Exception as exc:
        pageindex_result = {"status": "ERROR", "verdict": f"Error: {exc}"}
        pageindex_status = "ERROR"
        pageindex_verdict = pageindex_result["verdict"]
    pageindex_time = time.time() - t1

    print("\n" + "=" * 100)
    print(f"{'Method':<18} | {'Time (s)':<10} | {'Metric / Status':<20} | Verdict Summary")
    print("-" * 100)
    print(f"{'LVT Baseline':<18} | {baseline_time:<10.2f} | {f'Hits: {baseline_hits}':<20} | {baseline_verdict}")
    print(f"{'PageIndex+Ollama':<18} | {pageindex_time:<10.2f} | {pageindex_status:<20} | {pageindex_verdict[:55]}")
    print("=" * 100)

    print("\n[PageIndex Selection]")
    print(f"Pages: {pageindex_result.get('selected_pages', '-')}")
    print(f"Reason: {pageindex_result.get('selection_reason', '-')}")

    evidence = pageindex_result.get("evidence", [])
    if evidence:
        print("\n[Evidence]")
        for item in evidence:
            print(f"- {item}")

    missing = pageindex_result.get("missing", [])
    if missing:
        print("\n[Missing]")
        for item in missing:
            print(f"- {item}")

    print(f"\n[Full PageIndex Verdict]\n{pageindex_verdict.strip()}")

    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pdf_path": str(pdf_path),
        "requirement": requirement,
        "baseline_keywords": baseline_keywords,
        "model": model,
        "baseline": {
            "verdict": baseline_verdict,
            "score": baseline_score,
            "reason": baseline_reason,
            "time_seconds": baseline_time,
            "hit_count": baseline_hits,
            "hits": baseline_result.get("hits", []),
        },
        "pageindex": {
            "status": pageindex_status,
            "verdict": pageindex_verdict,
            "time_seconds": pageindex_time,
            "selected_pages": pageindex_result.get("selected_pages", "-"),
            "pages_used": pageindex_result.get("pages_used", "-"),
            "selection_reason": pageindex_result.get("selection_reason", "-"),
            "sections": pageindex_result.get("sections", []),
            "evidence": pageindex_result.get("evidence", []),
            "supporting_chunks": pageindex_result.get("supporting_chunks", []),
            "page_text_blobs": pageindex_result.get("page_text_blobs", []),
            "missing": pageindex_result.get("missing", []),
            "raw_result": _stringify(pageindex_result),
        },
    }
    log_path = _write_comparison_log(output_dir, log_payload)
    print(f"\nSaved comparison log to: {log_path}")


def run_ifu_tests() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"IFU PDF not found: {pdf_path}")

    print("=== Phase 1: PageIndex Setup ===")
    print(f"PDF: {pdf_path}")
    print(f"Model: {args.model}")
    print(f"Workspace: {workspace}")
    print(f"Output dir: {output_dir}")
    print(f"LLM timeout (s): {DEFAULT_LLM_TIMEOUT}")

    start_idx = time.time()
    client = PageIndexClient(
        model=args.model,
        retrieve_model=args.model,
        workspace=str(workspace),
    )
    doc_id = _get_or_index_doc(client, pdf_path, force_reindex=args.reindex)
    print(f"Index ready in {time.time() - start_idx:.2f}s")

    try:
        structure = json.loads(client.get_document_structure(doc_id))
        print("\nTop-level PageIndex tree:")
        print_tree(structure[:8])
    except Exception as exc:
        print(f"Warning: failed to print tree preview: {exc}")

    print("\n=== Phase 2: Interactive Testing ===")
    print("Enter one IFU requirement at a time.")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        requirement = input("\nEnter IFU requirement: ").strip()
        if requirement.lower() in {"q", "quit", "exit"}:
            break
        if not requirement:
            continue

        _run_single_test(pdf_path, requirement, client, doc_id, args.model, output_dir)


if __name__ == "__main__":
    run_ifu_tests()