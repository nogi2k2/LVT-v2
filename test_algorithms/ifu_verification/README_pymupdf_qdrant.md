# PyMuPDF + Qdrant IFU benchmark

Runs a local retrieval benchmark for the Trilogy IFU using:
- `PyMuPDF` for parsing
- `sentence-transformers/all-MiniLM-L6-v2` from the local HuggingFace cache for embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` from the local HuggingFace cache for reranking
- local persistent `QdrantClient(path=...)` storage inside the run directory

## Run

```powershell
C:\Users\320308180\Desktop\Work\LVT_v2\test_algos\Scripts\python.exe test_algorithms\ifu_verification\benchmark_pymupdf_qdrant_trilogy.py
```

## Outputs

Each run creates a timestamped folder under `test_output/pymupdf_qdrant_trilogy/` containing:
- `progress.log` - clean timestamped stage log
- `status.json` - last known run state
- `summary.json` - machine-readable results
- `summary.txt` - human-readable summary
- `requirement_details.log` - detailed requirement evidence
- `qdrant_store/` - persistent local vector store
- `*_pymupdf.md` - parsed markdown dump

## Notes

- The script is tuned for quiet logs; model progress bars and noisy library logs are suppressed.
- Pass/fail currently uses the top retrieved cosine score against a configurable threshold (`--cosine-threshold`, default `0.55`).
- Reranking is used to improve evidence ordering, but the pass/fail decision remains deterministic and local.
