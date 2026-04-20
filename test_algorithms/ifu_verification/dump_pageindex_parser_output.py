import argparse
import os
import sys
from pathlib import Path

import litellm


THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_PDF_PATH = TEST_ALGORITHMS_DIR / "data" / "test_ifu" / "DMVC-IFU.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_output" / "ifu_parser_output"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "DMVC-IFU_pageindex_parse_dump.md"
DEFAULT_MODEL = "ollama/qwen2.5:14b"


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

from pageindex.page_index import page_list_to_group_text
from pageindex.utils import get_page_tokens


litellm.drop_params = True


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Dump the raw PageIndex parser output for the IFU into a Markdown file."
	)
	parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to the IFU PDF.")
	parser.add_argument(
		"--output",
		default=str(DEFAULT_OUTPUT_PATH),
		help="Path to the Markdown report that will be written.",
	)
	parser.add_argument(
		"--pdf-parser",
		default="PyPDF2",
		choices=["PyPDF2", "PyMuPDF"],
		help="PDF text extractor to use. PageIndex currently defaults to PyPDF2.",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL,
		help="Model name used only for token counting while reproducing PageIndex grouping.",
	)
	return parser.parse_args()


def _normalize_text(text: str) -> str:
	cleaned = (text or "").replace("\x00", "")
	return cleaned.strip()


def _build_page_contents(page_list: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
	page_contents = []
	token_lengths = []
	for page_number, (page_text, token_length) in enumerate(page_list, start=1):
		normalized = _normalize_text(page_text)
		tagged = f"<physical_index_{page_number}>\n{normalized}\n<physical_index_{page_number}>\n\n"
		page_contents.append(tagged)
		token_lengths.append(token_length)
	return page_contents, token_lengths


def _render_markdown(pdf_path: Path, parser_name: str, model: str, page_list: list[tuple[str, int]], grouped_texts: list[str]) -> str:
	lines: list[str] = []
	lines.append("# PageIndex Parser Output Dump")
	lines.append("")
	lines.append(f"- Source PDF: `{pdf_path}`")
	lines.append(f"- Parser used: `{parser_name}`")
	lines.append(f"- Grouping model for token counts: `{model}`")
	lines.append(f"- Page count: `{len(page_list)}`")
	lines.append(f"- Prompt chunk count: `{len(grouped_texts)}`")
	lines.append("")
	lines.append("## What this shows")
	lines.append("")
	lines.append("This report dumps the raw page text and the exact tagged prompt chunks passed into PageIndex's no-TOC extraction flow before the LLM tries to build a hierarchy.")
	lines.append("")
	lines.append("## Per-page extraction")
	lines.append("")

	for page_number, (page_text, token_length) in enumerate(page_list, start=1):
		lines.append(f"### Page {page_number}")
		lines.append("")
		lines.append(f"- Token count: `{token_length}`")
		lines.append("")
		lines.append("```text")
		lines.append(_normalize_text(page_text) or "[No text extracted]")
		lines.append("```")
		lines.append("")

	lines.append("## No-TOC grouped prompt chunks")
	lines.append("")
	for chunk_index, chunk_text in enumerate(grouped_texts, start=1):
		lines.append(f"### Chunk {chunk_index}")
		lines.append("")
		lines.append("```text")
		lines.append(chunk_text.strip() or "[Empty chunk]")
		lines.append("```")
		lines.append("")

	return "\n".join(lines).rstrip() + "\n"


def main() -> int:
	args = _parse_args()
	pdf_path = Path(args.pdf).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve()

	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")

	page_list = get_page_tokens(str(pdf_path), model=args.model, pdf_parser=args.pdf_parser)
	page_contents, token_lengths = _build_page_contents(page_list)
	grouped_texts = page_list_to_group_text(page_contents, token_lengths)

	report = _render_markdown(
		pdf_path=pdf_path,
		parser_name=args.pdf_parser,
		model=args.model,
		page_list=page_list,
		grouped_texts=grouped_texts,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(report, encoding="utf-8")

	print(f"Wrote parser dump to: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
