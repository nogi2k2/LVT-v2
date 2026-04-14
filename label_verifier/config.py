import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.paths import get_icon_dir, get_base_dir

def get_default_icon_dir() -> str:
    return get_icon_dir()

def get_output_dir() -> str:
    return os.path.normpath(os.path.join(get_base_dir(), 'output'))