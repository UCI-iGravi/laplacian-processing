# CLAUDE.md

Guidance for Claude working in this repo.

## What this project is

Research code exploring how to correct **negative Jacobian determinants** in Laplacian-based deformable image registration by comparing correspondences between **moving-space** and **fixed-space** Laplacian solves. See [README.md](README.md) for the full picture.

Work is notebook-driven; `modules/` and `cluster_modules/` are the library layer the notebooks consume.

## Notebook authority

When making changes, default to these "current" notebooks unless the user says otherwise:
- **Most robust / paper-ready:** `clustering v3 fixed Laplacian for paper images.ipynb`
- **Clustering (fixed/moving variants):** `clustering v2 fixed Laplacian.ipynb`, `clustering v2 moving Laplacian.ipynb`
- **Earlier clustering scaffold:** `clustering.ipynb`

Older versions have been moved to `archive/`.

## Key conventions

- **Namespace packages:** `modules/` and `cluster_modules/` intentionally have no `__init__.py`. Don't add one.
- **Import form:** from the repo root, use `from modules import laplacian, jacobian`, `from modules.data import Data`, `from cluster_modules import cluster`. Do **not** use `import cluster` (would need the module on sys.path).
- **Notebooks:** edit with the `NotebookEdit` tool (the plain `Edit` tool rejects `.ipynb`). Cell IDs are stable — use them to target edits.
- **Point format:** correspondence points are `(z, y, x)` with z=0 for 2D slices. The `Data` class expects `mpoints`, `fpoints`, and a `(H, W)` resolution tuple.
- **Jacobian:** computed with SimpleITK in `modules/jacobian.py` → `sitk_jacobian_determinant(deformation)`.
- **No data in git:** `.npy`, `.nii.gz`, and CCF template volumes are gitignored. Don't assume their paths resolve — if a notebook fails due to missing data, say so rather than fabricating paths.

## Known cleanup debt

- Watch for stale `import cluster` (should be `from cluster_modules import cluster`) when pulling older notebooks from `archive/`.

## When asked to run things

Notebooks depend on unpublished data (`data/*.npy`, `CCF_DATA/`, registration outputs). Do not claim a notebook "runs" without actually executing it end-to-end. If data is missing, report that explicitly.
