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

from .diffusion import smooth
from .edges import edge_strength
from .interp import sample_grid, upscale
from .motion import estimate


@dataclass
class Params:
    sigma: float = 0.383      # Gaussian width of a sample's footprint, high-res px
    w_max: float = 16.0       # cap on history confidence (1/w_max = min blend factor)
    clamp: str = "edge"       # "edge", "minmax", "variance" or "none"
    gamma: float = 1.0        # variance clamp width in standard deviations
    alpha: float = 2.63       # edge clamp: box widened by alpha * local edge strength
    w_prior: float = 0.05     # weight of the spatial upscale where history is empty
    m_reproject: int = 8      # Lagrange stencil size for history resampling
    diffuse_t: float = 0.5    # RK4 anisotropic diffusion time on the output (0 = off)
    diffuse_K: float = 0.02   # edge threshold of the diffusion
    diffuse_w0: float = 1.0   # confidence at which diffusion is halved


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
       motion="given":     use the engine's motion vectors (like DLSS).
       motion="estimated": ignore them and find the shift by registering the
                           new frame against the history (tsr/motion.py).
       motion="refined":   start from the given vectors and refine them by
                           the same registration.
    """

    def __init__(self, scale=4, params=None, mode="world", motion="given", solver="gauss-newton"):
        self.scale = scale
        self.p = params or Params()
        self.mode = mode
        self.motion_source = motion
        self.solver = solver
        self.motions = []           # motion actually used each frame
        self.iterations = []        # solver iterates per frame (estimated motion)
        self.H = None
        self.W = None
        self.cam = np.zeros(2)      # camera position, accumulated from motion vectors
        self.origin = np.zeros(2, int)  # world position of the history grid (world mode)

    def reproject(self, motion):
        """Pixel q now shows the world point that was at q + motion last frame."""
        hh, ww = self.H.shape
        x, y = np.arange(ww) + motion[0], np.arange(hh) + motion[1]
        inside = ((y >= 0) & (y <= hh - 1))[:, None] & ((x >= 0) & (x <= ww - 1))[None, :]
        self.H = sample_grid(self.H, x, y, m=self.p.m_reproject)
        self.W = np.maximum(sample_grid(self.W, x, y, m=2), 0) * inside   # off-screen: no history

    def clamp(self, low, idx):
        if self.p.clamp in ("minmax", "edge"):
            lo = _neighbourhood(low, lambda s: s.min(0))
            hi = _neighbourhood(low, lambda s: s.max(0))
            if self.p.clamp == "edge":
                # Detail thinner than the sample spacing can sit outside the
                # min/max of the current samples. Widen the box where the
                # frame shows structure; flat areas (where ghosts would be
                # most visible) keep the tight box.
                e = self.p.alpha * edge_strength(low)
                lo, hi = lo - e, hi + e
        elif self.p.clamp == "variance":
            mu = _neighbourhood(low, lambda s: s.mean(0))
            sd = _neighbourhood(low, lambda s: s.std(0))
            lo, hi = mu - self.p.gamma * sd, mu + self.p.gamma * sd
        else:
            return
        self.H = np.clip(self.H, lo[idx], hi[idx])

    def step(self, frame):
        """Feed one frame; returns the displayed high-res image."""
        motion = frame.motion
        if self.motion_source != "given" and self.H is not None:
            # "estimated": constant-velocity guess from last frame's motion.
            # "refined": start from the engine's (possibly inaccurate) vector.
            if self.motion_source == "refined":
                d0 = np.asarray(frame.motion, float)
            else:
                d0 = self.motions[-1] if len(self.motions) > 1 else np.zeros(2)
            motion, hist = estimate(self.R, self.origin, self.cam, frame.low,
                                    frame.jitter, self.scale, d0, self.solver)
            self.iterations.append(hist)
        self.motions.append(np.asarray(motion, float))
        self.cam = self.cam + motion
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

        if self.motion_source != "given":
            # reference for registering the next frame: the displayed blend,
            # but kept on the history grid (spatial upscale shifted by offset)
            Sg = upscale(frame.low, self.scale, np.asarray(frame.jitter) + np.asarray(offset) / self.scale)
            self.R = (self.W * self.H + self.p.w_prior * Sg) / (self.W + self.p.w_prior)
        H, W = self.H, self.W
        if self.mode == "world":                  # the single display resample
            hh, ww = H.shape
            x, y = np.arange(ww) + offset[0], np.arange(hh) + offset[1]
            H, W = sample_grid(H, x, y, m=self.p.m_reproject), sample_grid(W, x, y, m=2)
        # Where evidence is thin (first frames, newly revealed edges) lean on
        # the single-frame Lagrange upscale instead of blocky raw samples.
        S = upscale(frame.low, self.scale, frame.jitter, m=4)
        out = np.clip((W * H + self.p.w_prior * S) / (W + self.p.w_prior), 0, 1)
        if self.p.diffuse_t > 0:
            # Display-only cleanup, never fed back into history. Few samples
            # means noisy, so smooth there; well-supported pixels stay as is.
            out = smooth(out, self.p.diffuse_t, self.p.diffuse_K,
                         weight=1.0 / (1.0 + (W / self.p.diffuse_w0) ** 2))
        return out


def run(frames, scale=4, params=None, mode="world", motion="given", solver="gauss-newton", upscaler=None):
    """Run the whole sequence; returns the list of output frames. Pass an
    `upscaler` to inspect its state (e.g. estimated motions) afterwards."""
    tu = upscaler or TemporalUpscaler(scale, params, mode, motion, solver)
    return [tu.step(f) for f in frames]
