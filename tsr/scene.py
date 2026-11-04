"""The "game": a continuous, analytic scene.

scene(x, y) returns brightness in [0, 1] at any real-valued point, measured in
high-res pixel units. Because it is defined everywhere, we can render it at
any resolution and from any sub-pixel camera position, and we always know
the exact ground truth.

Edges are smoothed with tanh over ~0.3 px instead of being hard steps. Real
renderers produce something similar, and it keeps the integrand smooth so
Simpson's rule converges at its proper rate.
"""
import numpy as np

EDGE = 0.3  # edge softness in high-res pixels


def _step(d):
    """Smooth 0 -> 1 transition across d = 0 (d is signed distance in px)."""
    return 0.5 * (1.0 + np.tanh(d / EDGE))


def checker(x, y, period=22.0, angle=0.35):
    """Rotated checkerboard: straight edges at an angle show jaggies clearly."""
    c, s = np.cos(angle), np.sin(angle)
    u = (c * x + s * y) * (np.pi / period)
    v = (-s * x + c * y) * (np.pi / period)
    # sin(u)*sin(v) changes sign on the checker edges; scale it to roughly px distance
    return _step(np.sin(u) * np.sin(v) * period / np.pi)


def zone_plate(x, y, cx=256.0, cy=256.0, k=0.0075):
    """cos(k r^2): ring frequency rises with radius. Classic aliasing test,
    the outer rings are too fine for the low-res camera and alias into moire."""
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return 0.5 + 0.5 * np.cos(k * r2)


def thin_lines(x, y, x0=60.0, y0=415.0):
    """Bars getting thinner left to right (like a lens resolution chart)."""
    out = np.zeros(np.broadcast(x, y).shape)
    xs = x0
    for width in (4.0, 3.0, 2.0, 1.5, 1.0, 0.75):
        for _ in range(3):
            out = np.maximum(out, _step(x - xs) * _step(xs + width - x))
            xs += 2 * width
        xs += 10.0
    return out * _step(y - y0) * _step(y0 + 80 - y)


SWITCH_FRAME = 20  # frame at which the "light" turns on


def scene(x, y, t=0):
    """Composite: checkerboard background, zone plate disc, thin-line chart.

    At frame t >= SWITCH_FRAME a square panel lights up. Nothing moves, so the
    motion vectors say "no change", yet the old history is now wrong there:
    the test case for anti-ghosting.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    img = 0.15 + 0.7 * checker(x, y)
    disc = _step(140.0 - np.hypot(x - 256.0, y - 256.0))
    img = img * (1 - disc) + zone_plate(x, y) * disc
    bars = thin_lines(x, y)
    band = _step(y - 410.0) * _step(500.0 - y) * _step(x - 50.0) * _step(250.0 - x)
    img = img * (1 - band) + (0.05 + 0.9 * bars) * band
    if t >= SWITCH_FRAME:
        panel = _step(x - 330.0) * _step(470.0 - x) * _step(y - 410.0) * _step(500.0 - y)
        img = img * (1 - panel) + 0.95 * panel
    return np.clip(img, 0.0, 1.0)
