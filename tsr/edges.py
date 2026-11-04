"""Edge analysis with the structure tensor and the power method (Experiment 8).

The structure tensor at a pixel is the 2x2 symmetric matrix

    J = blur( [Ix*Ix  Ix*Iy]
              [Ix*Iy  Iy*Iy] )

built from the image gradient (Ix, Iy). Its eigen-decomposition describes
the local structure:
  * dominant eigenvector v1  -> direction ACROSS the edge (the gradient)
  * second eigenvector   v2  -> direction ALONG the edge
  * eigenvalues l1 >= l2     -> flat: both ~0;  edge: l1 >> l2;  texture/corner: both big

Every pixel has its own 2x2 matrix, so we run the power method on all of
them at once with numpy (one vectorised iteration updates every pixel).
"""
import numpy as np


def gradient(img):
    """Central differences, f'(x) ~ (f(x+1) - f(x-1)) / 2: the derivative of
    the 3-point Newton/Lagrange interpolant at its middle node. Edges repeat."""
    p = np.pad(img, 1, mode="edge")
    ix = (p[1:-1, 2:] - p[1:-1, :-2]) / 2
    iy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
    return ix, iy


def blur(img, sigma):
    """Separable Gaussian blur (edges repeat)."""
    r = max(1, int(3 * sigma))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    p = np.pad(img, r, mode="edge")
    p = np.apply_along_axis(lambda row: np.convolve(row, k, "valid"), 1, p)
    return np.apply_along_axis(lambda col: np.convolve(col, k, "valid"), 0, p)


def structure_tensor(img, rho=1.0):
    """Components (a, b, c) of J = [[a, b], [b, c]] at every pixel."""
    ix, iy = gradient(img)
    return blur(ix * ix, rho), blur(ix * iy, rho), blur(iy * iy, rho)


def power_method(a, b, c, iters=20, tol=1e-10, history=False):
    """Dominant eigenpair of every 2x2 matrix [[a, b], [b, c]] at once.

    Repeats v <- J v / |J v|. The component along v2 shrinks by l2/l1 each
    step, so convergence is fast on clear edges (l2 << l1) and slow where the
    two eigenvalues are close (isotropic texture). l1 is the Rayleigh
    quotient v^T J v; since J is 2x2, l2 = trace(J) - l1.

    Returns l1, l2, (vx, vy) and the number of iterations used; with
    history=True also the l1 estimate after every iteration.
    """
    vx = np.ones_like(a)
    vy = np.full_like(a, 0.5)           # not aligned with an axis, so never orthogonal to v1 everywhere
    n = np.hypot(vx, vy)
    vx, vy = vx / n, vy / n
    lam = np.zeros_like(a)
    trace = [] if history else None
    for k in range(1, iters + 1):
        wx, wy = a * vx + b * vy, b * vx + c * vy
        n = np.maximum(np.hypot(wx, wy), 1e-30)
        vx, vy = wx / n, wy / n
        new = vx * (a * vx + b * vy) + vy * (b * vx + c * vy)
        if history:
            trace.append(new.copy())
        done = np.max(np.abs(new - lam)) < tol * max(np.max(np.abs(new)), 1e-30)
        lam = new
        if done:
            break
    out = (lam, a + c - lam, (vx, vy), k)
    return out + (trace,) if history else out


def edge_strength(img, rho=1.0, iters=20):
    """sqrt(l1): local contrast in brightness per pixel (0 on flat areas)."""
    l1, _, _, _ = power_method(*structure_tensor(img, rho), iters=iters)
    return np.sqrt(np.maximum(l1, 0))
