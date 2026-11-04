"""Check our quadrature against scipy, using the lab's own test integral:
integral of 1/(1+x^2) on [0, 1] = arctan(1) = pi/4."""
import numpy as np
from scipy.integrate import simpson, trapezoid

from tsr.quadrature import integrate_1d, pixel_average


def f(x):
    return 1.0 / (1.0 + x ** 2)


def test_matches_scipy():
    x = np.linspace(0, 1, 11)
    assert np.isclose(integrate_1d(f, 0, 1, 10, "simpson"), simpson(f(x), x=x))
    assert np.isclose(integrate_1d(f, 0, 1, 10, "trapezoid"), trapezoid(f(x), x=x))


def test_simpson_beats_trapezoid():
    exact = np.pi / 4
    assert abs(integrate_1d(f, 0, 1, 10, "simpson") - exact) < 1e-6
    assert abs(integrate_1d(f, 0, 1, 10, "trapezoid") - exact) > 1e-4


def test_pixel_average_of_plane_is_exact():
    # Simpson is exact for polynomials up to degree 3, so a plane is exact
    plane = lambda x, y: 2 * x + 3 * y
    got = pixel_average(plane, np.array([0.0, 4.0]), np.array([0.0, 0.0]), 4.0)
    assert np.allclose(got, [2 * 2 + 3 * 2, 2 * 6 + 3 * 2])
