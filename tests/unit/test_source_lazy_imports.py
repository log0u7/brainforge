"""Static guard: heavy dependencies are never imported at module level.

AGENTS.md: heavy dependencies (torch, transformers, trl, peft, bitsandbytes,
unsloth) are imported lazily inside functions; the base package and the unit
test suite must run without the ``training`` extra. Runtime sys.modules
checks cannot separate lazy imports exercised by tests from module-level
ones, so this guard parses the AST of every source file and rejects any
top-level heavy import. Imports guarded by ``if TYPE_CHECKING:`` are fine:
they never execute at runtime.
"""

import ast
from pathlib import Path

HEAVY_MODULES = {"torch", "transformers", "trl", "peft", "bitsandbytes", "unsloth"}
SRC_ROOT = Path(__file__).parents[2] / "src" / "brainforge"


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _heavy_root(module_name: str) -> str | None:
    root = module_name.split(".")[0]
    return root if root in HEAVY_MODULES else None


def _scan_node(node: ast.stmt, offenders: list[str], path: Path) -> None:
    if isinstance(node, ast.If) and _is_type_checking(node.test):
        return  # TYPE_CHECKING imports never execute at runtime
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = _heavy_root(alias.name)
            if root:
                offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
    elif isinstance(node, ast.ImportFrom) and node.module:
        root = _heavy_root(node.module)
        if root:
            imported = ", ".join(alias.name for alias in node.names)
            offenders.append(f"{path.name}:{node.lineno} from {node.module} import {imported}")
    elif isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)):
        # Module-level containers still execute at import time: recurse into
        # every body they can run, but never into function/class definitions
        # (lazy imports inside those are the rule, not the violation).
        for body in (
            node.body,
            getattr(node, "orelse", []),
            *[handler.body for handler in getattr(node, "handlers", [])],
            getattr(node, "finalbody", []),
        ):
            for child in body:
                _scan_node(child, offenders, path)


def test_no_module_level_heavy_imports_in_src():
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            _scan_node(node, offenders, path)
    assert not offenders, f"module-level heavy imports: {offenders}"


def _offenders_for(source: str, path: Path) -> list[str]:
    offenders: list[str] = []
    tree = ast.parse(source)
    for node in tree.body:
        _scan_node(node, offenders, path)
    return offenders


def test_guard_flags_module_level_try_import(tmp_path):
    """A try/except around a module-level import is still a module-level import."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
        "\n"
        "def f():\n"
        "    return torch\n"
    )
    offenders = _offenders_for(offender.read_text(encoding="utf-8"), offender)
    assert any("torch" in entry for entry in offenders)


def test_guard_allows_function_level_import(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text("def f():\n    import torch\n    return torch\n")
    assert not _offenders_for(clean.read_text(encoding="utf-8"), clean)


def test_guard_flags_module_level_conditional_import(tmp_path):
    offender = tmp_path / "offender.py"
    offender.write_text(
        "import sys\nif sys.version_info >= (3, 12):\n    import trl\n\ndef f():\n    return trl\n"
    )
    offenders = _offenders_for(offender.read_text(encoding="utf-8"), offender)
    assert any("trl" in entry for entry in offenders)
