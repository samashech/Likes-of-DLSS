"""Check our interpolation against scipy and against each other."""
import numpy as np
import pytest
from scipy.interpolate import lagrange

from tsr.interp import lagrange_weights, newton_backward, newton_forward, sample

rng = np.random.default_rng(0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # syllabus lists scipy lagrange
def test_lagrange_matches_scipy():
    v = rng.random(6)
    p = np.array([0.3, 2.5, 4.9])
    ours = lagrange_weights(p, 6) @ v
    assert np.allclose(ours, lagrange(np.arange(6), v)(p))


def test_newton_forms_equal_lagrange():
    # the interpolating polynomial is unique, so all three must agree
    v = rng.random((10, 4))
    p = rng.random(10) * 3
    lag = np.sum(lagrange_weights(p, 4) * v, -1)
    assert np.allclose(newton_forward(v, p), lag)
    assert np.allclose(newton_backward(v, p), lag)


def test_cubic_reproduced_exactly():
    # 4-point interpolation is exact for any cubic surface
    x, y = np.meshgrid(np.arange(12.0), np.arange(12.0))
    img = 0.01 * x ** 3 - 0.2 * x * y + 0.05 * y ** 2
    ux, uy = rng.uniform(3, 8, 50), rng.uniform(3, 8, 50)
    exact = 0.01 * ux ** 3 - 0.2 * ux * uy + 0.05 * uy ** 2
    for method in ("lagrange", "newton"):
        assert np.allclose(sample(img, ux, uy, 4, method), exact)


def test_matrix_form_equals_pointwise():
    from tsr.interp import sample_grid
    img = rng.random((16, 20))
    ux, uy = np.linspace(-1, 21, 37), np.linspace(-2, 17, 29)
    UX, UY = np.meshgrid(ux, uy)
    for m in (1, 2, 4, 8):
        assert np.allclose(sample_grid(img, ux, uy, m), sample(img, UX, UY, m))
