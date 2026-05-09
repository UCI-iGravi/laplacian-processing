"""Guards against regression of the module extraction.

The original parity test compared byte-identical notebook-vs-module sources.
Once the notebooks were rewritten to import from the modules, that test
became impossible by design. This replacement asserts the *negative* property:
no extracted function is defined inline in any notebook. Reintroducing an
inline `def` (by copy/paste or by restoring from archive) would cause drift
between the notebook and the module, so this test catches that class of bug.
"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_FUNCTIONS = {
    "square_to_circle_correspondences", "circle_to_circle_correspondences",
    "show", "find_index", "jacobian_plot",
    "jacobian_determinant_central", "jacobian_determinant_finite", "compute_jacobian_det",
    "upscale_dvf_preserve", "upscale_dvf_linear_y",
    "assign_unique_correspondences", "generate_additional_correspondences_from_upscaled_dvf",
    "get_point_info", "get_expanding_coordinates", "get_contracting_coordinates",
    "get_mapped_point", "get_displacement",
    "interpolate_coordinates", "upscale_grid_values", "upscale_grid_with_priors",
    "boundary_points_on_grid", "export_interpolated_grid_to_csv",
    "plot_point_correspondences",
    "print_coord_list", "print_interpolated_coords_grid", "print_fixed_at_moving",
    "load_png", "show_png", "show_2_pngs",
    "create_blobby_circular_shapes", "create_ellipse_correspondences",
    "create_crossing_lines_correspondences", "create_double_helix_waves",
}

NOTEBOOKS = [
    "clustering.ipynb",
    "clustering v2 fixed Laplacian.ipynb",
    "clustering v2 moving Laplacian.ipynb",
    "clustering v3 fixed Laplacian for paper images.ipynb",
]


def _top_level_defs(nb_name: str) -> set[str]:
    path = ROOT / nb_name
    if not path.exists():
        pytest.skip(f"notebook missing: {nb_name}")
    nb = json.loads(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for cell in nb["cells"]:
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
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                found.add(node.name)
    return found


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_no_inline_extracted_defs(nb_name):
    defs = _top_level_defs(nb_name)
    regressions = defs & EXTRACTED_FUNCTIONS
    assert not regressions, (
        f"{nb_name} has inline defs for extracted functions: {sorted(regressions)}. "
        f"These should be imported from modules/ or cluster_modules/ instead."
    )
