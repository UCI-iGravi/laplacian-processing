"""One-shot: extract function bodies from notebooks and compare duplicates.

Run: python tests/_diff_dup.py
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "clustering.ipynb",
    "clustering v2 fixed Laplacian.ipynb",
    "clustering v2 moving Laplacian.ipynb",
    "clustering v3 fixed Laplacian for paper images.ipynb",
]


def extract_functions(nb_name):
    path = ROOT / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    funcs = defaultdict(list)  # name -> list of source strings (one per occurrence)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        # Strip IPython cell magics / line magics so ast.parse succeeds
        stripped = "\n".join(
            "" if line.lstrip().startswith(("%", "!", "?")) else line
            for line in src.split("\n")
        )
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        src = stripped
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seg = ast.get_source_segment(src, node)
                if seg is not None:
                    funcs[node.name].append(seg)
    return funcs


def main():
    all_funcs = {}  # nb -> {name: [source, ...]}
    for nb in NOTEBOOKS:
        all_funcs[nb] = extract_functions(nb)

    # Map name -> list of (nb, occurrence_idx, source)
    by_name = defaultdict(list)
    for nb, funcs in all_funcs.items():
        for name, srcs in funcs.items():
            for i, s in enumerate(srcs):
                by_name[name].append((nb, i, s))

    dups = {n: occs for n, occs in by_name.items() if len(occs) > 1}
    print(f"Found {len(dups)} duplicated function names\n")

    for name, occs in sorted(dups.items()):
        sources = set(s for _, _, s in occs)
        status = "IDENTICAL" if len(sources) == 1 else f"DIVERGED ({len(sources)} variants)"
        print(f"{name}: {len(occs)} copies, {status}")
        for nb, i, s in occs:
            print(f"  - {nb} [occ {i}] ({len(s)} chars)")
        if len(sources) > 1:
            print("  !! sources differ — needs per-notebook extraction or distinct names")
        print()


if __name__ == "__main__":
    main()
