"""Show cell_id + all top-level defs for each notebook, so we can plan edits."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXTRACTED = {
    "square_to_circle_correspondences", "circle_to_circle_correspondences",
    "show", "find_index",
    "jacobian_determinant_central", "jacobian_determinant_finite", "compute_jacobian_det",
    "upscale_dvf_preserve", "upscale_dvf_linear_y",
    "assign_unique_correspondences", "generate_additional_correspondences_from_upscaled_dvf",
    "get_point_info", "get_expanding_coordinates", "get_contracting_coordinates",
    "get_mapped_point", "get_displacement",
}


def main():
    nb_name = sys.argv[1]
    path = ROOT / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        stripped = "\n".join(
            "" if line.lstrip().startswith(("%", "!", "?")) else line
            for line in src.split("\n")
        )
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        top_defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        top_classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        extractable_here = [n for n in top_defs if n in EXTRACTED]
        other_here = [n for n in top_defs if n not in EXTRACTED]
        if extractable_here or other_here or top_classes:
            print(f"Cell {i} id={cell.get('id')}: "
                  f"extractable={extractable_here}, other_defs={other_here}, classes={top_classes}, "
                  f"total_lines={len(src.splitlines())}")


if __name__ == "__main__":
    main()
