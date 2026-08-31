"""Compatibility and dependency guards for the split config loader."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from kiro_crew.config import loader, resolution, sections

_SECTION_IMPORTED_REEXPORTS = {
    "EFFORT_LEVELS",
    "_CONNECT_TIMEOUT_CEILING",
    "_MAX_RECOVERY_CEILING",
    "_MINT_TIMEOUT_CEILING",
    "_MINT_TIMEOUT_FLOOR",
    "_RECOVER_BACKOFF_CEILING",
    "_STT_CATALOG",
    "_resolve_stt_model",
    "resolve_selected_backend",
}


def _defined_names(module: ModuleType) -> set[str]:
    """Names defined by *module*, excluding its private logger binding."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    names.discard("logger")
    return names


def test_loader_reexports_every_extracted_definition_by_identity() -> None:
    """Historical loader imports remain aliases, not parallel implementations."""
    mismatches = [
        f"{module.__name__}.{name}"
        for module in (sections, resolution)
        for name in sorted(
            _defined_names(module) | (_SECTION_IMPORTED_REEXPORTS if module is sections else set())
        )
        if getattr(loader, name, None) is not getattr(module, name)
    ]
    assert mismatches == []


def test_extracted_modules_do_not_import_the_loader() -> None:
    """The facade owns orchestration; extracted modules cannot depend back on it."""
    code = (
        "import sys\n"
        "import kiro_crew.config.sections\n"
        "import kiro_crew.config.resolution\n"
        "forbidden = (\n"
        "    'kiro_crew.config.loader',\n"
        "    'kiro_crew.config.schema',\n"
        "    'kiro_crew.config.validation',\n"
        ")\n"
        "print(','.join(name for name in forbidden if name in sys.modules))\n"
    )
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        env=env,
    )
    assert result.stdout.strip() == ""
