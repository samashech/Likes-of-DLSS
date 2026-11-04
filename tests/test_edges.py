"""Check the vectorised power method against numpy.linalg.eig."""
import numpy as np

from tsr.edges import power_method, structure_tensor


def test_power_method_matches_eig():
    rng = np.random.default_rng(1)
    img = rng.random((24, 24))
    a, b, c = structure_tensor(img)
    l1, l2, (vx, vy), _ = power_method(a, b, c, iters=500, tol=1e-14)
    J = np.stack([np.stack([a, b], -1), np.stack([b, c], -1)], -2)
    ev = np.linalg.eig(J)[0]
    assert np.allclose(l1, ev.max(-1), rtol=1e-6)
    assert np.allclose(l2, ev.min(-1), rtol=1e-6, atol=1e-12)
    # v1 is an eigenvector: J v = l1 v
    assert np.allclose(a * vx + b * vy, l1 * vx, atol=1e-8)


def test_vertical_edge_points_across():
    img = np.zeros((20, 20))
    img[:, 10:] = 1.0
    l1, l2, (vx, vy), _ = power_method(*structure_tensor(img))
    assert abs(vx[10, 10]) > 0.99 and l2[10, 10] < 1e-6 * l1[10, 10]
