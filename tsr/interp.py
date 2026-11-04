"""Polynomial image interpolation (Experiments 3 and 4), written from scratch.

To read an image at a non-integer position u (in pixel units), take the m
nearest pixels on each axis, fit the unique polynomial of degree m-1 through
them, and evaluate it at u. Doing x then y ("separable") extends it to 2D.

Three ways to get that same polynomial:
  * Lagrange form       -> sum of values times basis weights   (Exp 3)
  * Newton forward diff -> built from the start of the stencil (Exp 4)
  * Newton backward diff-> built from the end of the stencil   (Exp 4)
The interpolating polynomial is unique, so all three give the same answer up
to rounding. We use Lagrange weights in the pipeline because the weights
depend only on the fractional offset, not on the pixel values.
"""
from math import factorial

import numpy as np


def _stencil(u, m):
    """First stencil index and evaluation point for an m-point stencil.

    Nodes are at i0, i0+1, ..., i0+m-1, centred on u; p is u relative to i0.
    """
    f = np.floor(u)
    i0 = f.astype(int) - (m // 2 - 1)
    return i0, u - i0


def lagrange_weights(p, m):
    """Lagrange basis L_k(p) = prod_{j != k} (p - j) / (k - j), nodes 0..m-1.

    p has any shape; returns shape p.shape + (m,). Weights always sum to 1.
    """
    p = np.asarray(p, float)[..., None]
    k = np.arange(m)
    w = np.ones(p.shape[:-1] + (m,))
    for j in range(m):
        term = (p - j) / np.where(k == j, 1, k - j)
        w *= np.where(k == j, 1.0, term)
    return w


def newton_forward(v, p):
    """Newton forward-difference formula, nodes 0..m-1 (v[..., k] at node k).

    P(p) = sum_k C(p, k) * Delta^k v_0,  C(p, k) = p(p-1)...(p-k+1) / k!
    """
    d = np.array(v, float)
    m = d.shape[-1]
    out = d[..., 0].copy()
    coef = np.ones_like(p, dtype=float)
    for k in range(1, m):
        d = np.diff(d, axis=-1)          # next row of the difference table
        coef = coef * (p - (k - 1))
        out += coef / factorial(k) * d[..., 0]
    return out


def newton_backward(v, p):
    """Newton backward-difference formula from the last node n = m-1.

    With s = p - n:  P = sum_k s(s+1)...(s+k-1) / k! * Nabla^k v_n
    """
    d = np.array(v, float)
    m = d.shape[-1]
    s = p - (m - 1)
    out = d[..., -1].copy()
    coef = np.ones_like(p, dtype=float)
    for k in range(1, m):
        d = np.diff(d, axis=-1)
        coef = coef * (s + (k - 1))
        out += coef / factorial(k) * d[..., -1]
    return out


def _gather(img, iy, ix):
    """img[iy, ix] with indices clamped to the border (edge pixels repeat)."""
    h, w = img.shape
    return img[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]


def sample(img, ux, uy, m=4, method="lagrange"):
    """Read img at real positions (ux, uy) using m-point interpolation per axis.

    Coordinates are in pixel units with pixel centres at integers. m=1 is
    nearest neighbour, m=2 bilinear, m=4 bicubic-Lagrange.
    """
    ux, uy = np.broadcast_arrays(np.asarray(ux, float), np.asarray(uy, float))
    if m == 1:
        return _gather(img, np.rint(uy).astype(int), np.rint(ux).astype(int))
    ix0, px = _stencil(ux, m)
    iy0, py = _stencil(uy, m)

    if method == "lagrange":
        wx, wy = lagrange_weights(px, m), lagrange_weights(py, m)
        out = np.zeros(ux.shape)
        for a in range(m):
            for b in range(m):
                out += wy[..., a] * wx[..., b] * _gather(img, iy0 + a, ix0 + b)
        return out

    # Newton: interpolate each of the m rows along x, then the results along y.
    # Forward formula when p sits in the first half of the stencil, backward in
    # the second half (the textbook rule: start the table from the nearer end).
    newton = lambda v, p: np.where(p < m / 2, newton_forward(v, p), newton_backward(v, p))
    rows = np.stack([newton(np.stack([_gather(img, iy0 + a, ix0 + b) for b in range(m)], -1), px)
                     for a in range(m)], -1)
    return newton(rows, py)


def upscale(low, scale, jitter=(0.0, 0.0), m=4, method="lagrange"):
    """Single-frame upscale by an integer factor.

    High-res pixel centre q + 0.5 lies at low-res coordinate
        u = (q + 0.5) / scale - 0.5 - jitter
    (the jitter shifted where the low-res pixels were sampled).
    """
    h, w = low.shape
    q = np.arange(h * scale)
    ux = (np.arange(w * scale) + 0.5) / scale - 0.5 - jitter[0]
    uy = (q + 0.5) / scale - 0.5 - jitter[1]
    UX, UY = np.meshgrid(ux, uy)
    return np.clip(sample(low, UX, UY, m, method), 0.0, 1.0)
