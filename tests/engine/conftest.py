"""Rende importabili `engine` e `contracts` lanciando `pytest tests/engine` dalla root.

NOTA: questa cartella non deve contenere `__init__.py`, altrimenti il package
di test `engine` oscurerebbe il vero package `engine` della root.
"""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
