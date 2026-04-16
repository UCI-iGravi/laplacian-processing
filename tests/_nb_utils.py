"""Helpers for extracting code cells from .ipynb files for parity tests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_all_code(notebook_name: str) -> str:
    """Concatenate all code-cell source for a notebook."""
    path = ROOT / notebook_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        parts.append("".join(src) if isinstance(src, list) else src)
    return "\n\n".join(parts)


def get_cell_source(notebook_name: str, cell_id: str) -> str:
    path = ROOT / notebook_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            src = cell["source"]
            return "".join(src) if isinstance(src, list) else src
    raise KeyError(f"cell {cell_id} not found in {notebook_name}")
