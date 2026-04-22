from pathlib import Path
import sys, os, json
# ensure repo root
ROOT = Path(__file__).resolve().parents[3]
# load parser_adapters module directly (avoid package import issues across venvs)
from importlib.machinery import SourceFileLoader
# prefer adapter next to this script
adapter_path = Path(__file__).resolve().parent / 'parser_adapters.py'
if not adapter_path.exists():
    adapter_path = ROOT / 'test_algorithms' / 'ifu_verification' / 'parser_adapters.py'
parser_adapters = SourceFileLoader('parser_adapters', str(adapter_path)).load_module()
parse_with_pymupdf = parser_adapters.parse_with_pymupdf

# Settings
PDF = Path('test_algorithms') / 'data' / 'test_ifu' / 'Trilogy_EV300_IFU.pdf'
REQUIREMENTS = Path('test_algorithms') / 'data' / 'test_ifu' / 'Trilogy_EV300_requirements.txt'
OUT_DIR = Path('test_output') / 'retrieval_bench'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / 'verification_log.txt'
JSON_FILE = OUT_DIR / 'verification_summary.json'

# find local sentence-transformers snapshot
HUB = Path(r"C:\Users\320308180\Desktop\Work\LVT_v2\.cache\huggingface\hub")
ST_MODEL_DIR = None
for p in (HUB).iterdir():
    if p.is_dir() and 'models--sentence-transformers--all-MiniLM-L6-v2' in p.name:
        snap = p / 'snapshots'
        if snap.exists():
            # pick first snapshot
            for s in snap.iterdir():
                if s.is_dir():
                    ST_MODEL_DIR = str(s)
                    break
        break
if ST_MODEL_DIR is None:
    raise SystemExit('Sentence-transformers model snapshot not found in hub')

# parse pdf with pymupdf (produces markdown)
md_files = parse_with_pymupdf(PDF, OUT_DIR)
# read the generated md (first one)
md_path = Path(md_files[0])
md_text = md_path.read_text(encoding='utf-8')

# simple chunking (by paragraph)
paras = [p.strip() for p in md_text.split('\n\n') if p.strip()]
chunks = []
max_words = 150
for i, p in enumerate(paras):
    words = p.split()
    if len(words) <= max_words:
        chunks.append({'id': f'c{i}', 'text': p})
    else:
        # split long paragraph
        for j in range(0, len(words), max_words):
            part = ' '.join(words[j:j+max_words])
            chunks.append({'id': f'c{i}_{j}', 'text': part})

# embed with sentence-transformers (local model)
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer(ST_MODEL_DIR)
texts = [c['text'] for c in chunks]
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

# build in-memory index (numpy) to avoid FAISS/numpy binary compatibility issues
import numpy as _np
norms = _np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / (norms + 1e-12)


# load requirements
reqs = [r.strip() for r in REQUIREMENTS.read_text(encoding='utf-8').splitlines() if r.strip()]

# embed requirements
req_embs = model.encode(reqs, convert_to_numpy=True, show_progress_bar=False)
req_norms = _np.linalg.norm(req_embs, axis=1, keepdims=True)
req_embs = req_embs / (req_norms + 1e-12)

# verification
results = []
with LOG_FILE.open('w', encoding='utf-8') as logf:
    for i, req in enumerate(reqs):
        q = req_embs[i:i+1]
        # compute cosine similarities via dot product with all chunks
        qv = q[0]
        sims_all = (_np.dot(embs, qv)).tolist() if embs.size else []
        # get top 5
        ranked = sorted(enumerate(sims_all), key=lambda x: x[1], reverse=True)[:5]
        ids = [r[0] for r in ranked]
        sims = [r[1] for r in ranked]
        evidence = []
        for sim, idx in zip(sims, ids):
            if idx < 0 or idx >= len(chunks):
                continue
            evidence.append({'score': float(sim), 'text': chunks[idx]['text']})
        passed = len(evidence) > 0 and evidence[0]['score'] >= 0.65
        out = {'requirement': req, 'pass': passed, 'top_score': evidence[0]['score'] if evidence else 0.0, 'evidence': evidence}
        results.append(out)
        # log
        logf.write(f"REQUIREMENT: {req}\nPASS: {passed} TOP_SCORE: {out['top_score']:.3f}\nEVIDENCE:\n")
        for e in evidence:
            snippet = e['text'][:200].replace('\n', ' ')
            logf.write(f" - {e['score']:.3f}: {snippet}\n")
        logf.write('\n')

# write json summary
JSON_FILE.write_text(json.dumps({'pdf': str(PDF), 'model_dir': ST_MODEL_DIR, 'results': results}, indent=2), encoding='utf-8')
print('Verification complete. Log:', LOG_FILE)
print('Summary JSON:', JSON_FILE)
