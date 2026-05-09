"""Functional smoke tests for extracted functions. Uses small synthetic inputs
so tests run in under a second without any external data files."""
import numpy as np
import pytest


def test_square_to_circle_shapes():
    from cluster_modules.synthetic import square_to_circle_correspondences
    sq, cr = square_to_circle_correspondences(n_points=40, resolution=(128, 128), padding=10)
    assert sq.shape == (40, 2)
    assert cr.shape == (40, 2)
    # Both should land inside the padded image area
    assert sq.min() >= 0 and sq.max() <= 128
    assert cr.min() >= 0 and cr.max() <= 128


def test_circle_to_circle_shapes():
    from cluster_modules.synthetic import circle_to_circle_correspondences
    inner, outer = circle_to_circle_correspondences(n_points=30, resolution=(64, 64), padding=5)
    assert inner.shape == (30, 2)
    assert outer.shape == (30, 2)
    # Inner radius should be smaller in image coords
    inner_r = np.linalg.norm(inner - np.array([32, 32]), axis=1).mean()
    outer_r = np.linalg.norm(outer - np.array([32, 32]), axis=1).mean()
    assert inner_r < outer_r


def test_find_index_hit_and_miss():
    from cluster_modules.viz import find_index
    arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert find_index(arr, [3, 4, 5]) == 1
    assert find_index(arr, [9, 9, 9]) == -1


def _zero_dvf(h=8, w=8):
    return np.zeros((3, 1, h, w), dtype=float)


def _linear_y_dvf(h=8, w=8, slope=0.1):
    dvf = np.zeros((3, 1, h, w), dtype=float)
    for y in range(h):
        dvf[1, 0, y, :] = slope * y
    return dvf


def test_jacobian_determinant_central_zero_field():
    from modules.jacobian import jacobian_determinant_central
    dvf = _zero_dvf()
    jdet = jacobian_determinant_central(dvf, pt=(3, 3))
    assert jdet.shape == (8, 8)
    # Zero displacement → jacobian det = 1
    assert np.allclose(jdet, 1.0)


def test_jacobian_determinant_finite_zero_field():
    from modules.jacobian import jacobian_determinant_finite
    dvf = _zero_dvf()
    jdet = jacobian_determinant_finite(dvf)
    assert jdet.shape == (8, 8)
    assert np.allclose(jdet, 1.0)


def test_compute_jacobian_det_zero_field():
    from modules.jacobian import compute_jacobian_det
    dvf = _zero_dvf()
    jdet = compute_jacobian_det(dvf)
    assert jdet.shape == (8, 8)
    assert np.allclose(jdet, 1.0)


def test_compute_jacobian_det_stretch():
    """A dy = slope*y field means 1 + ddy/dy = 1 + slope; det should be that."""
    from modules.jacobian import compute_jacobian_det
    slope = 0.2
    dvf = _linear_y_dvf(slope=slope)
    jdet = compute_jacobian_det(dvf)
    # Interior should be (1+slope) * 1 - 0 = 1 + slope
    assert np.allclose(jdet[3:5, 3:5], 1 + slope)


def test_upscale_dvf_preserve_preserves_originals():
    from modules.upscaling import upscale_dvf_preserve
    h, w, scale = 4, 4, 3
    dvf = np.random.RandomState(0).rand(3, 1, h, w)
    up = upscale_dvf_preserve(dvf, scale=scale)
    assert up.shape == (3, 1, h * scale, w * scale)
    for y in range(h):
        for x in range(w):
            assert np.allclose(up[:, 0, y * scale, x * scale], dvf[:, 0, y, x])


def test_upscale_dvf_linear_y_endpoints():
    from modules.upscaling import upscale_dvf_linear_y
    h, w, scale = 4, 3, 2
    dvf = np.zeros((3, 1, h, w))
    dvf[1, 0, :, :] = np.arange(h).reshape(-1, 1)  # y-displacement = y
    up = upscale_dvf_linear_y(dvf, scale=scale)
    assert up.shape == (3, 1, h * scale, w * scale)
    # First original row should be preserved
    assert np.allclose(up[1, 0, 0, 0], 0)
    # Last original row should be preserved
    assert np.allclose(up[1, 0, (h - 1) * scale, 0], h - 1)


def test_assign_unique_correspondences_identity():
    from modules.upscaling import assign_unique_correspondences
    fixed = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    disp = np.zeros_like(fixed)
    moving = fixed.copy()
    a = assign_unique_correspondences(fixed, disp, moving)
    assert np.array_equal(a, np.array([0, 1, 2]))


def test_get_mapped_point_and_get_displacement():
    from modules.inspect_utils import get_mapped_point, get_displacement
    dvf = np.zeros((3, 1, 8, 8))
    dvf[1, 0, 3, 4] = 1.5  # dy
    dvf[2, 0, 3, 4] = -0.5  # dx
    mapped = get_mapped_point((3, 4), dvf)
    assert np.allclose(mapped, [3 + 1.5, 4 - 0.5])
    disp = get_displacement((3, 4), dvf)
    assert np.allclose(disp, [0, 1.5, -0.5])


def test_get_expanding_and_contracting_coordinates(capsys):
    from modules.inspect_utils import get_expanding_coordinates, get_contracting_coordinates
    jdet = np.array([[0.5, 1.5], [2.0, 0.2]])
    exp = get_expanding_coordinates(jdet, threshold=1, verbose=False)
    con = get_contracting_coordinates(jdet, threshold=1, verbose=False)
    assert exp.shape[1] == 2
    assert set(map(tuple, exp)) == {(0, 1), (1, 0)}
    assert set(map(tuple, con)) == {(0, 0), (1, 1)}


def test_generate_additional_correspondences_smoke():
    from modules.upscaling import generate_additional_correspondences_from_upscaled_dvf
    scale = 3
    h, w = 4, 4
    dvf_up = np.zeros((3, 1, h * scale, w * scale))
    jac = np.ones((h * scale, w * scale)) * 2.0  # all > threshold
    pairs = generate_additional_correspondences_from_upscaled_dvf(
        dvf_up, jac, scale=scale, jacobian_thresh=1.0
    )
    assert isinstance(pairs, list)
