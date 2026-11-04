"""Check our root-finders against scipy on a syllabus-style polynomial."""
import numpy as np
from scipy.optimize import bisect, newton as sp_newton, root_scalar

from tsr.rootfind import Counted, bisection, newton, regula_falsi, secant

p = np.poly1d([1, 0, -1, -2])          # x^3 - x - 2, root near 1.5214
dp = p.deriv()
ROOT = bisect(p, 1, 2, xtol=1e-14)


def test_all_methods_find_root():
    assert abs(bisection(p, 1, 2, 1e-10)[0] - ROOT) < 1e-9
    assert abs(regula_falsi(p, 1, 2, 1e-12)[0] - ROOT) < 1e-9
    assert abs(secant(p, 1, 2, 1e-12)[0] - ROOT) < 1e-9
    assert abs(newton(p, dp, 1.5, 1e-12)[0] - ROOT) < 1e-9


def test_matches_scipy():
    assert np.isclose(newton(p, dp, 1.5, 1e-12)[0], sp_newton(p, 1.5, fprime=dp))
    assert np.isclose(secant(p, 1, 2, 1e-12)[0], root_scalar(p, x0=1, x1=2, method="secant").root)


def test_newton_fewer_iterations_than_bisection():
    assert len(newton(p, dp, 1.5, 1e-10)[1]) < len(bisection(p, 1, 2, 1e-10)[1])


def test_counted_memoises():
    f = Counted(lambda x: x * x - 2)
    f(1.0), f(1.0), f(2.0)
    assert f.evals == 2
