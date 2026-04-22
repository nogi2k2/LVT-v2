from __future__ import annotations

import sys
import shutil
import time
from pathlib import Path
from typing import List

from .parser_adapters import parse_with_pymupdf


def run_retrieval_ingest_with_pymupdf(pdf_path: Path, project_name: str = "trilogy_ev300") -> dict:
    """Parse `pdf_path` with PyMuPDF, then call the Retrieval-Engine ingestion pipeline while
    monkeypatching its `DocumentParser` to return the generated markdown files.
    """
    repo_root = Path.cwd().resolve()
    retrieval_src = repo_root / "potential-solution" / "Retrieval-Engine" / "src"
    if str(retrieval_src) not in sys.path:
        sys.path.insert(0, str(retrieval_src))

    # parse to a local temp folder
    temp_out = Path("test_output") / "parser_bench" / "pymupdf_for_retrieval"
    temp_out.mkdir(parents=True, exist_ok=True)
    generated = parse_with_pymupdf(pdf_path, temp_out)

    try:
        import retrieval_engine.tools as re_tools
        import retrieval_engine.config as re_config
    except Exception as exc:
        return {"status": "error", "reason": f"Could not import retrieval_engine: {exc}"}

    # Create project parsed dir where Retrieval-Engine expects parsed files
    collection = project_name.lower().replace(" ", "_")
    target_parsed = re_config.PARSED_MD_DIR / collection
    if target_parsed.exists():
        shutil.rmtree(target_parsed)
    target_parsed.mkdir(parents=True, exist_ok=True)

    # copy generated md files into target parsed dir
    copied = []
    for p in generated:
        src = Path(p)
        dest = target_parsed / src.name
        shutil.copy2(src, dest)
        copied.append(str(dest))

    # monkeypatch DocumentParser used by retrieval_engine.tools so it doesn't attempt to re-parse
    class DummyParser:
        def __init__(self, output_dir: str, artifacts_path: str = None):
            self.output_dir = output_dir

        def parse(self, target_path: str) -> List[Path]:
            # return list of markdown files in the target_parsed dir
            return [p for p in target_parsed.glob("*.md")]

    re_tools.DocumentParser = DummyParser

    # call ingestion
    try:
        started = time.time()
        result = re_tools.ingest_documents(project_name, str(pdf_path))
        result["ingest_seconds"] = time.time() - started
        result["copied_md"] = copied
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default=str(Path("test_algorithms") / "data" / "test_ifu" / "Trilogy_EV300_IFU.pdf"))
    p.add_argument("--project", default="trilogy_ev300")
    args = p.parse_args()
    pdf = Path(args.pdf).expanduser().resolve()
    out = run_retrieval_ingest_with_pymupdf(pdf, args.project)
    print(out)
