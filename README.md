# Laplacian Processing

Research code for correcting negative Jacobian determinants in deformable image registration by leveraging correspondences between Laplacian operations in **moving** and **fixed** space.

## Motivation

Laplacian-based registration interpolates a smooth displacement field from a sparse set of point correspondences. Numerical error in the solve can produce regions with negative Jacobian determinants (folds), which are non-physical. This repo explores correcting those folds by:

1. Identifying the correspondences that cause negative Jdets (by intersection, orientation, or magnitude).
2. Comparing Laplacian solutions computed in moving space vs. fixed space, and using their correspondence to filter / augment the displacement field.
3. Clustering correspondences and applying the Laplacian per-cluster to avoid conflicting displacements.

## Repository layout

```
modules/              Core Laplacian + Jacobian library
  laplacian.py          moving-space Laplacian solver (createA, sliceToSlice3DLaplacian)
  laplacian_fixed.py    fixed-space variant
  data.py / data_fixed.py  Data class wrapping points + deformation + jdet field
  jacobian.py           SimpleITK-based Jacobian determinant
  correspondences.py    correspondence helpers
cluster_modules/      Clustering + plotting utilities for correspondence sets
  cluster.py            clustering methods (y_direction, quadrant, displacement_direction, helix_phase)
  data_utils.py         image I/O, point-to-image rasterization
  reg_utils.py          registration helpers (forward/inverse transforms)
data/                 Correspondence CSVs (sparse/interpolated)
registration/         Standalone 3D Laplacian registration script
images/               Figures used in notebooks / paper
archive/              Deprecated predecessors to current notebooks
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `clustering v3 fixed Laplacian for paper images.ipynb` | **Most comprehensive.** Fixed-space + moving-space Laplacian, augmentation, grid interpolation, ANTs/elastix comparison, paper figures. |
| `clustering v2 fixed Laplacian.ipynb` | Fixed-space Laplacian + Laplacian upscaling. |
| `clustering v2 moving Laplacian.ipynb` | Moving-space variant. |
| `clustering.ipynb` | Earlier clustering scaffold (double-helix synthetic correspondences). |

## Setup

```bash
pip install -r requirements.txt
```

Notebooks assume the repo root is the working directory so that `from modules import ...` and `from cluster_modules import ...` resolve. No `__init__.py` is needed — these are Python 3 namespace packages.

Runtime data (`.npy`, `.nii.gz` deformation fields, template volumes) are not committed. Paths in the notebooks reference `data/` and `CCF_DATA/`; supply your own or point to an existing copy.

## Standalone registration

`registration/laplacian3DRegistration.py` is a standalone script for applying the moving-space Laplacian registration to 3D volumes. It requires Elastix binaries (https://github.com/SuperElastix/elastix/releases) placed alongside the script, plus the parameter files (`001_parameters_Rigid.txt`, `002_parameters_BSpline.txt`).
