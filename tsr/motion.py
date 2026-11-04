"""Motion estimation by solving a nonlinear system (Experiment 2e).

Unknown: the camera shift d = (dx, dy) since the previous frame.

If the camera is now at cam_prev + d, low-res sample (i, j) shows the world
point  cam_prev + d + scale*(j + 0.5 + jitter).  Reading the high-res
history H there predicts what the sample should be. The mismatch

    r(d) = predicted(d) - observed          (one entry per low-res pixel)

is smallest at the true shift, so we solve the 2 nonlinear equations

    F(d) = J(d)^T r(d) = 0,     J = dr/dd = gradient of H at the samples

i.e. the gradient of E(d) = 1/2 |r(d)|^2 is zero. Gauss-Newton linearises
r around the current d and solves the 2x2 normal equations
(J^T J) delta = -J^T r with linalg.solve, repeating until |delta| is tiny.

The reference image R is what the upscaler displayed for the previous
frame (history blended with the single-frame Lagrange upscale where
evidence is thin), kept on the history grid. Raw history does not work
early on: each sample fills a flat patch, so the gradient (and J) is zero
inside patches. Blurring the history does not work either: a blurred
prediction compared against sharp samples moves the minimum (measured bias
up to 0.23 px), while the sharp reference converges to 0.002 px.
"""
import numpy as np
from scipy.optimize import fsolve

from .interp import sample_grid

EPS = 0.25  # finite-difference step for the gradient of H, in high-res px


class Registration:
    """Builds r(d) and J(d) for one frame against the reference image H."""

    def __init__(self, H, origin, cam_prev, low, jitter, scale, margin=6):
        self.H, self.low = H, low
        h, w = low.shape
        # world position of each low-res sample when d = 0, in history grid
        # coordinates (pixel centres at integers, as sample_grid expects)
        self.bx = cam_prev[0] + scale * (np.arange(w) + 0.5 + jitter[0]) - origin[0] - 0.5
        self.by = cam_prev[1] + scale * (np.arange(h) + 0.5 + jitter[1]) - origin[1] - 0.5
        # Use every sample that lands inside the history. (A confidence mask
        # does not work: Halton jitter puts each new sample in the gaps
        # between old ones, exactly where confidence is lowest.)
        hh, ww = H.shape
        inside = (((self.by > margin) & (self.by < hh - 1 - margin))[:, None]
                  & ((self.bx > margin) & (self.bx < ww - 1 - margin))[None, :])
        self.mask = inside
        self.n_evals = 0

    def predict(self, d):
        return sample_grid(self.H, self.bx + d[0], self.by + d[1], m=4)

    def residual(self, d):
        self.n_evals += 1
        return (self.predict(d) - self.low)[self.mask]

    def jacobian(self, d):
        """Columns: d r / d dx and d r / d dy, by central differences."""
        x, y = self.bx + d[0], self.by + d[1]
        gx = (sample_grid(self.H, x + EPS, y, m=4) - sample_grid(self.H, x - EPS, y, m=4)) / (2 * EPS)
        gy = (sample_grid(self.H, x, y + EPS, m=4) - sample_grid(self.H, x, y - EPS, m=4)) / (2 * EPS)
        return np.stack([gx[self.mask], gy[self.mask]], -1)

    def F(self, d):
        """The 2 equations: gradient of 1/2 |r|^2 = J^T r."""
        return self.jacobian(d).T @ self.residual(d)


def gauss_newton(reg, d0, tol=1e-4, max_iter=20, robust=True):
    """Returns d and the list of iterates.

    With robust=True each step is a weighted least-squares solve with Huber
    weights (iteratively reweighted least squares): residuals beyond
    c = 1.345 * robust spread get weight c/|r|. Aliased regions, whose
    moire moves differently from the camera, and changed regions (the light
    switch) are outliers; this stops them pulling the estimate.
    """
    d = np.asarray(d0, float)
    hist = [d.copy()]
    for _ in range(max_iter):
        J, r = reg.jacobian(d), reg.residual(d)
        if robust:
            c = 1.345 * 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            w = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
        else:
            w = np.ones_like(r)
        JtJ = (J * w[:, None]).T @ J
        JtJ += 1e-9 * np.trace(JtJ) * np.eye(2)   # tiny damping: never singular
        delta = np.linalg.solve(JtJ, -(J * w[:, None]).T @ r)
        d = d + delta
        hist.append(d.copy())
        if np.linalg.norm(delta) < tol:
            break
    return d, hist


def estimate(R, origin, cam_prev, low, jitter, scale, d0, solver="gauss-newton", robust=True):
    """Motion estimate against reference R; returns d and the iterates."""
    reg = Registration(R, origin, cam_prev, low, jitter, scale)
    if solver == "fsolve":
        d = fsolve(reg.F, d0, xtol=1e-6)
        return d, [np.asarray(d0, float), d]
    return gauss_newton(reg, d0, robust=robust)
