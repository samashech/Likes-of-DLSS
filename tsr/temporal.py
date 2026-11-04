"""Temporal super-resolution: the DLSS-style core.

We keep a high-res "history" image H and a per-pixel confidence W (how many
samples' worth of evidence H is built from). Every frame:

  1. Reproject  H and W along the motion vector, so they line up with the new
                camera (cubic Lagrange sampling, Module 2).
  2. Gather     the new low-res samples near each high-res pixel, weighted by
                a Gaussian of their distance. Result: a new estimate C with
                confidence Wn.
  3. Clamp      H into the range of values the current frame sees nearby.
                History outside that range is stale (ghosting) and is cut off.
  4. Blend      H <- (W*H + Wn*C) / (W + Wn), a running weighted mean, with
                W capped at w_max so the image keeps adapting.

Step 4 is the weighted least-squares estimate of each pixel from every sample
it has seen. As samples from different jitter offsets pile up, the estimate
converges to the true high-res image, detail a single frame never had.
"""
from dataclasses import dataclass

import numpy as np

from .interp import sample, upscale


@dataclass
class Params:
    sigma: float = 0.6        # Gaussian width of a sample's footprint, high-res px
    w_max: float = 16.0       # cap on history confidence (1/w_max = min blend factor)
    clamp: str = "minmax"     # "minmax", "variance" or "none"
    gamma: float = 1.0        # variance clamp width in standard deviations
    w_prior: float = 0.05     # weight of the spatial upscale where history is empty
    m_reproject: int = 8      # Lagrange stencil size for history resampling


def _neighbourhood(low, op):
    """3x3 low-res neighbourhood statistic (edges repeat)."""
    p = np.pad(low, 1, mode="edge")
    h, w = low.shape
    stack = np.stack([p[a:a + h, b:b + w] for a in range(3) for b in range(3)])
    return op(stack)


def gather(low, jitter, scale, sigma, offset=(0.0, 0.0)):
    """New-sample estimate C and its confidence Wn for every high-res pixel.

    Low-res sample (i, j) sits at high-res position scale*(j+0.5+jitter) +
    offset; each high-res pixel takes the 3x3 nearest samples with Gaussian
    weights. `offset` places the samples on a grid that is not the screen
    (the world-aligned history grid).
    """
    h, w = low.shape
    qx = np.arange(w * scale) + 0.5 - offset[0]
    qy = np.arange(h * scale) + 0.5 - offset[1]
    ux = qx / scale - 0.5 - jitter[0]           # low-res coordinate of each hi pixel
    uy = qy / scale - 0.5 - jitter[1]
    UX, UY = np.meshgrid(ux, uy)
    cx, cy = np.rint(UX).astype(int), np.rint(UY).astype(int)
    num = np.zeros(UX.shape)
    den = np.zeros(UX.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ix, iy = cx + dx, cy + dy
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            d2 = ((ix - UX) ** 2 + (iy - UY) ** 2) * scale ** 2   # distance^2 in hi px
            wgt = np.exp(-d2 / (2 * sigma ** 2)) * ok
            num += wgt * low[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
            den += wgt
    return num / np.maximum(den, 1e-12), den, (cy.clip(0, h - 1), cx.clip(0, w - 1))


def shift_int(a, dx, dy):
    """b[q] = a[q + d] for an integer shift d; pixels shifted in are 0. Exact."""
    b = np.zeros_like(a)
    h, w = a.shape
    ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
    xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    b[yd, xd] = a[ys, xs]
    return b


class TemporalUpscaler:
    """mode="world":  history lives on a grid fixed to the world, so it only
                      ever moves by whole pixels (exact). Sub-pixel camera
                      offset goes into where new samples land. Interpolation
                      happens once, for display, and never feeds back.
                      Valid because our motion is a global translation.
       mode="screen": the textbook TAA way, history resampled to the new
                      screen every frame (works for any motion, but blurs:
                      numerical diffusion).
    """

    def __init__(self, scale=4, params=None, mode="world"):
        self.scale = scale
        self.p = params or Params()
        self.mode = mode
        self.H = None
        self.W = None
        self.cam = np.zeros(2)      # camera position, accumulated from motion vectors
        self.origin = np.zeros(2, int)  # world position of the history grid (world mode)

    def reproject(self, motion):
        """Pixel q now shows the world point that was at q + motion last frame."""
        hh, ww = self.H.shape
        X, Y = np.meshgrid(np.arange(ww) + motion[0], np.arange(hh) + motion[1])
        inside = (X >= 0) & (X <= ww - 1) & (Y >= 0) & (Y <= hh - 1)
        self.H = sample(self.H, X, Y, m=self.p.m_reproject)
        self.W = np.maximum(sample(self.W, X, Y, m=2), 0) * inside   # off-screen: no history

    def clamp(self, low, idx):
        if self.p.clamp == "minmax":
            lo = _neighbourhood(low, lambda s: s.min(0))
            hi = _neighbourhood(low, lambda s: s.max(0))
        elif self.p.clamp == "variance":
            mu = _neighbourhood(low, lambda s: s.mean(0))
            sd = _neighbourhood(low, lambda s: s.std(0))
            lo, hi = mu - self.p.gamma * sd, mu + self.p.gamma * sd
        else:
            return
        self.H = np.clip(self.H, lo[idx], hi[idx])

    def step(self, frame):
        """Feed one frame; returns the displayed high-res image."""
        self.cam = self.cam + frame.motion
        offset = (0.0, 0.0)
        if self.mode == "world":
            new_origin = np.rint(self.cam).astype(int)
            if self.H is not None:
                d = new_origin - self.origin
                self.H, self.W = shift_int(self.H, *d), shift_int(self.W, *d)
            self.origin = new_origin
            offset = self.cam - self.origin       # sub-pixel part, |offset| <= 0.5
        elif self.H is not None:
            self.reproject(frame.motion)

        C, Wn, idx = gather(frame.low, frame.jitter, self.scale, self.p.sigma, offset)
        if self.H is None:
            self.H, self.W = C, Wn
        else:
            self.clamp(frame.low, idx)
            W = np.minimum(self.W, self.p.w_max)
            self.H = (W * self.H + Wn * C) / (W + Wn)
            self.W = W + Wn

        H, W = self.H, self.W
        if self.mode == "world":                  # the single display resample
            hh, ww = H.shape
            X, Y = np.meshgrid(np.arange(ww) + offset[0], np.arange(hh) + offset[1])
            H, W = sample(H, X, Y, m=self.p.m_reproject), sample(W, X, Y, m=2)
        # Where evidence is thin (first frames, newly revealed edges) lean on
        # the single-frame Lagrange upscale instead of blocky raw samples.
        S = upscale(frame.low, self.scale, frame.jitter, m=4)
        return np.clip((W * H + self.p.w_prior * S) / (W + self.p.w_prior), 0, 1)


def run(frames, scale=4, params=None, mode="world"):
    """Run the whole sequence; returns the list of output frames."""
    tu = TemporalUpscaler(scale, params, mode)
    return [tu.step(f) for f in frames]
