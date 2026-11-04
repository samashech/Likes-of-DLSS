"""Gauss-Newton registration recovers a known sub-pixel shift."""
import numpy as np
from scipy.optimize import fsolve

from tsr.motion import Registration, gauss_newton


def test_recovers_shift_with_perfect_reference():
    scene = lambda x, y: 0.5 + 0.25 * np.sin(x / 7.0) * np.cos(y / 5.0) + 0.2 * np.sin((x + y) / 11.0)
    s, n = 4, 32
    q = np.arange(n * s) + 0.5                       # reference grid, origin 0, camera 0
    R = scene(*np.meshgrid(q, q))
    true_d, jit = np.array([0.37, -0.21]), np.array([0.1, -0.3])
    lx = true_d[0] + s * (np.arange(n) + 0.5 + jit[0])
    ly = true_d[1] + s * (np.arange(n) + 0.5 + jit[1])
    low = scene(*np.meshgrid(lx, ly))
    reg = Registration(R, np.zeros(2, int), np.zeros(2), low, jit, s)
    d, hist = gauss_newton(reg, np.zeros(2), robust=False)
    assert np.linalg.norm(d - true_d) < 0.01 and len(hist) < 10
    assert np.allclose(fsolve(reg.F, np.zeros(2)), d, atol=1e-3)
