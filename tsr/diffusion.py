"""Edge-preserving anisotropic diffusion solved with RK4 (Experiment 7).

The PDE   du/dt = div( D grad u )   smooths the image u over time. D is a
2x2 "diffusion tensor" per pixel, built from the structure tensor's
eigenvectors (tsr/edges.py):

    D = c1 * v1 v1^T + c2 * v2 v2^T
        v1 across the edge: c1 = 1 / (1 + l1 / K^2)  (strong edge -> ~0, no blur across)
        v2 along the edge:  c2 = 1                   (always smooth along it)

So noise and dotted texture get averaged along edges and rings while the
edges themselves stay sharp. Discretising space turns the PDE into a big
system of ODEs du/dt = A u ("method of lines"), stepped in time with RK4.

D is computed once from the input image, so A is a fixed linear operator.
Explicit time stepping is then stable only if dt * |lambda_max(A)| stays
inside the method's stability region: about 2.0 for Euler and 2.785 for
RK4. lambda_max is found with the power method.
"""
import numpy as np

from .edges import power_method, structure_tensor


def diffusion_tensor(img, K=0.05, rho=1.5):
    """Components (a, b, c) of D = [[a, b], [b, c]] at every pixel."""
    l1, _, (vx, vy), _ = power_method(*structure_tensor(img, rho), iters=30)
    c1 = 1.0 / (1.0 + np.maximum(l1, 0) / K ** 2)
    c2 = 1.0
    # v2 = (-vy, vx) is perpendicular to v1
    a = c1 * vx * vx + c2 * vy * vy
    b = (c1 - c2) * vx * vy
    c = c1 * vy * vy + c2 * vx * vx
    return a, b, c


def operator(D):
    """Return A(u) = div(D grad u) on the pixel grid (edges reflect, no flux out).

    Diagonal terms use compact differences between neighbours, so even the
    finest 1-pixel dot pattern is damped. Mixed terms use central differences.
    """
    a, b, c = D
    ah = 0.5 * (a[:, 1:] + a[:, :-1])          # a at x+1/2 faces
    ch = 0.5 * (c[1:, :] + c[:-1, :])          # c at y+1/2 faces

    def A(u):
        out = np.zeros_like(u)
        fx = ah * (u[:, 1:] - u[:, :-1])       # flux a * u_x through each face
        out[:, :-1] += fx
        out[:, 1:] -= fx
        fy = ch * (u[1:, :] - u[:-1, :])
        out[:-1, :] += fy
        out[1:, :] -= fy
        p = np.pad(u, 1, mode="edge")
        ux = (p[1:-1, 2:] - p[1:-1, :-2]) / 2
        uy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
        gx = np.pad(b * uy, 1, mode="edge")    # d/dx (b u_y) + d/dy (b u_x)
        gy = np.pad(b * ux, 1, mode="edge")
        out += (gx[1:-1, 2:] - gx[1:-1, :-2]) / 2 + (gy[2:, 1:-1] - gy[:-2, 1:-1]) / 2
        return out

    return A


def spectral_radius(A, shape, iters=60, seed=0):
    """Largest |eigenvalue| of the linear operator A, by the power method."""
    v = np.random.default_rng(seed).standard_normal(shape)
    lam = 0.0
    for _ in range(iters):
        w = A(v)
        lam = np.linalg.norm(w) / np.linalg.norm(v)
        v = w / np.linalg.norm(w)
    return lam


def rk4_step(A, u, dt):
    k1 = A(u)
    k2 = A(u + 0.5 * dt * k1)
    k3 = A(u + 0.5 * dt * k2)
    k4 = A(u + dt * k3)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(A, u, dt):
    return u + dt * A(u)


def integrate(A, u, t_end, dt, method="rk4"):
    """March du/dt = A u from 0 to t_end with fixed steps (last step trimmed)."""
    step = rk4_step if method == "rk4" else euler_step
    t = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        u = step(A, u, h)
        t += h
    return u


def smooth(img, t_end=0.5, K=0.02, weight=None, safety=0.9):
    """Edge-preserving smoothing; dt chosen at `safety` x the RK4 stability limit.

    `weight` (same shape as img, in [0, 1]) scales the diffusion per pixel,
    so some pixels can be smoothed more than others.
    """
    D = diffusion_tensor(img, K)
    if weight is not None:
        D = tuple(d * weight for d in D)
    A = operator(D)
    dt = safety * 2.785 / spectral_radius(A, img.shape)
    return np.clip(integrate(A, img, t_end, dt), 0.0, 1.0)
