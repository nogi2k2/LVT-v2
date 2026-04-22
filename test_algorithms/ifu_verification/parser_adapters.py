from __future__ import annotations

import sys
from pathlib import Path
from typing import List
import time

from pathlib import os

# Prefer repository root from current working directory when running scripts
POTENTIAL = Path.cwd().resolve() / "potential-solution"


def _ensure_parserlib_on_path() -> None:
    parserlib_src = POTENTIAL / "Parser-Lib" / "src"
    if str(parserlib_src) not in sys.path:
        sys.path.insert(0, str(parserlib_src))


def parse_with_parserlib(pdf_path: Path, out_dir: Path) -> List[Path]:
    """Use the local `Parser-Lib` implementation to parse the PDF into markdown.

    Returns list of generated file paths.
    """
    _ensure_parserlib_on_path()
    try:
        from parser_lib import DocumentParser
    except Exception as exc:
        raise ImportError("parser_lib not available on sys.path") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    parser = DocumentParser(str(out_dir))
    started = time.time()
    generated = parser.parse(str(pdf_path))
    elapsed = time.time() - started
    return generated


def parse_with_pymupdf(pdf_path: Path, out_dir: Path) -> List[Path]:
    """Fallback parser using PyMuPDF (fitz) to extract text into a simple markdown file per document.

    This is intentionally simple: it extracts page text and writes headings for pages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise ImportError("PyMuPDF (fitz) is required for the pymupdf adapter") from exc

    doc = fitz.open(str(pdf_path))
    md_lines = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text")
        md_lines.append(f"## Page {pno+1}\n\n")
        md_lines.append(text)
        md_lines.append("\n\n")

    out_path = out_dir / f"{pdf_path.stem}_pymupdf.md"
    out_path.write_text("".join(md_lines), encoding="utf-8")
    return [out_path]
