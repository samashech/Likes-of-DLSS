"""Numerical integration (Experiments 5 and 6), written from scratch.

Used to turn the continuous scene into pixels: a camera pixel does not see
one point, it records the *average* light over its square footprint, i.e.

    pixel = (1 / area) * double integral of scene(x, y) over the footprint

We approximate that double integral with a tensor-product rule: apply the
1D rule along x, then along y. That gives a grid of sample offsets and a
matching grid of weights.
"""
import numpy as np


def trapezoid_weights(n):
    """Nodes and weights of the composite trapezoidal rule on [0, 1] with n intervals."""
    x = np.linspace(0.0, 1.0, n + 1)
    w = np.full(n + 1, 1.0 / n)
    w[0] = w[-1] = 0.5 / n
    return x, w


def simpson_weights(n):
    """Nodes and weights of composite Simpson's 1/3 rule on [0, 1].

    n must be even. Weights follow the pattern (h/3) * [1, 4, 2, 4, ..., 2, 4, 1].
    """
    if n % 2:
        raise ValueError("Simpson's 1/3 rule needs an even number of intervals")
    h = 1.0 / n
    x = np.linspace(0.0, 1.0, n + 1)
    w = np.full(n + 1, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return x, w * h / 3.0


def integrate_1d(f, a, b, n, rule="simpson"):
    """Integrate f over [a, b] with n intervals. Used to check the rules against scipy."""
    x, w = (simpson_weights if rule == "simpson" else trapezoid_weights)(n)
    return (b - a) * np.sum(w * f(a + (b - a) * x))


def pixel_average(scene, x0, y0, size, n=4, rule="simpson"):
    """Average of scene over the squares [x0, x0+size] x [y0, y0+size].

    x0, y0 are arrays (one entry per pixel), so a whole image is integrated
    at once: the scene is evaluated (n+1)^2 times, each time on every pixel.
    Because the weights already sum to 1 on [0, 1], the sum is the average.
    """
    u, wu = (simpson_weights if rule == "simpson" else trapezoid_weights)(n)
    out = np.zeros(np.broadcast(x0, y0).shape)
    for ui, wi in zip(u, wu):
        for vj, wj in zip(u, wu):
            out += wi * wj * scene(x0 + ui * size, y0 + vj * size)
    return out
