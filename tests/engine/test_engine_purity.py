"""Guardrail di recinto: `/engine` deve restare puro, deterministico e PAPER_ONLY.

Analisi statica (AST) dei sorgenti: nessun import di rete, I/O, asincronia,
orologio di sistema, casualita o firma di transazioni.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[2] / "engine"
ENGINE_FILES = sorted(ENGINE_DIR.glob("*.py"))

FORBIDDEN_MODULES = {
    # rete
    "aiohttp", "http", "httpx", "requests", "socket", "ssl", "urllib", "urllib3", "websocket", "websockets",
    # I/O, processi, asincronia
    "asyncio", "io", "os", "pathlib", "shutil", "subprocess", "sys", "threading",
    # non determinismo
    "random", "secrets", "time", "uuid",
    # firma on-chain / ordini reali
    "eth_account", "py_clob_client", "web3",
}
FORBIDDEN_CALLS = {"open", "input", "print", "exec", "eval", "__import__"}
FORBIDDEN_ATTRIBUTES = {"now", "utcnow", "today"}  # datetime.now() & co: l'orologio va iniettato


def test_engine_modules_exist():
    names = {path.name for path in ENGINE_FILES}
    assert {"__init__.py", "book_math.py", "combinatorial_arb.py", "news_latency.py", "risk_manager.py"} <= names


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda path: path.name)
def test_engine_module_is_pure(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in FORBIDDEN_MODULES, f"{path.name}: import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in FORBIDDEN_MODULES, f"{path.name}: from {node.module} import ..."
            assert root not in {"collector", "paper_trader", "infra"}, f"{path.name}: dipendenza fuori recinto"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}: chiamata a {node.func.id}()"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in FORBIDDEN_ATTRIBUTES, f"{path.name}: uso di .{node.func.attr}()"
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncWith, ast.AsyncFor)):
            pytest.fail(f"{path.name}: codice asincrono non ammesso in /engine")
