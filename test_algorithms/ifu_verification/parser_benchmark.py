from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

# ensure project root is on sys.path so we can import local modules when running as script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_algorithms.ifu_verification.parser_adapters import parse_with_parserlib, parse_with_pymupdf
import json as _json

def write_json(path, payload):
    path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark Parser-Lib vs PyMuPDF on an IFU PDF")
    p.add_argument("--pdf", default=str(Path("test_algorithms") / "data" / "test_ifu" / "Trilogy_EV300_IFU.pdf"))
    p.add_argument("--output-root", default=str(Path("test_output") / "parser_bench"))
    return p.parse_args()


def _run_benchmark(pdf_path: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    # Parser-Lib
    pl_out = output_root / "parserlib"
    pl_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        generated = parse_with_parserlib(pdf_path, pl_out)
        status = "ok"
    except Exception as exc:
        generated = []
        status = f"error: {exc}"
    t_pl = time.time() - t0
    results.append({"parser": "parserlib", "status": status, "time_seconds": t_pl, "generated": [str(p) for p in generated]})

    # PyMuPDF fallback
    pm_out = output_root / "pymupdf"
    pm_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        generated = parse_with_pymupdf(pdf_path, pm_out)
        status = "ok"
    except Exception as exc:
        generated = []
        status = f"error: {exc}"
    t_pm = time.time() - t0
    results.append({"parser": "pymupdf", "status": status, "time_seconds": t_pm, "generated": [str(p) for p in generated]})

    summary = {
        "pdf": str(pdf_path),
        "timestamp": time.time(),
        "results": results,
    }
    write_json(output_root / "summary.json", summary)
    lines = [f"Parser benchmark for {pdf_path}", ""]
    for r in results:
        lines.append(f"{r['parser']}: {r['status']} in {r['time_seconds']:.2f}s -> {len(r['generated'])} files")
    write_text(output_root / "summary.txt", "\n".join(lines))
    return summary


def main():
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    summary = _run_benchmark(pdf_path, output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
