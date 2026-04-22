from pathlib import Path
import os
import json
import sys

# ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# set cache to the path you provided
os.environ['HF_HOME'] = r"C:\Users\320308180\Desktop\Work\LVT_v2\.cache\ocr"
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ['HF_HOME']
os.environ['XDG_CACHE_HOME'] = r"C:\Users\320308180\Desktop\Work\LVT_v2\.cache\ocr"

from pathlib import Path
from test_algorithms.ifu_verification import parser_benchmark

pdf = Path('test_algorithms') / 'data' / 'test_ifu' / 'Trilogy_EV300_IFU.pdf'
out = Path('test_output') / 'parser_bench'
print('Using HF_HOME:', os.environ.get('HF_HOME'))
print('Running benchmark for', pdf)
res = parser_benchmark._run_benchmark(pdf, out)
print('\nSummary:\n')
print(json.dumps(res, indent=2))
