"""Frame generator: plays the role of the game engine.

For each frame k it produces what DLSS receives from a game:
  * a low-res image rendered with a sub-pixel jitter,
  * the jitter offset used,
  * the camera motion since the previous frame (the motion vector).
It also produces the high-res ground truth, which a real game never has but
we need in order to measure error.

Coordinates are in high-res pixels. A low-res pixel is `scale` high-res
pixels wide. World position of a pixel = its position on screen + camera.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .quadrature import pixel_average
from .scene import scene


def halton(index, base):
    """index-th element (1-based) of the Halton sequence in `base`, in [0, 1).

    Halton points fill the unit square evenly without clumping, so after a few
    frames every part of a low-res pixel has been sampled. DLSS, FSR 2 and
    TAA all use this sequence (bases 2 and 3) for camera jitter.
    """
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def jitter_sequence(n_frames, phases=16):
    """Jitter per frame in low-res pixel units, in [-0.5, 0.5)^2."""
    return np.array([(halton(k % phases + 1, 2) - 0.5, halton(k % phases + 1, 3) - 0.5)
                     for k in range(n_frames)])


@dataclass
class Frame:
    low: np.ndarray        # low-res rendered image (H/scale, W/scale)
    jitter: np.ndarray     # (jx, jy) in low-res pixels
    camera: np.ndarray     # (cx, cy) camera position in high-res pixels, no jitter
    motion: np.ndarray     # camera change since last frame, high-res pixels
    truth: np.ndarray      # high-res ground truth at this camera position


FIELDS = [f.name for f in fields(Frame)]


def render(width, height, origin, spacing, t=0, n=2):
    """Render a width x height image whose pixel centres are `spacing` apart
    (in high-res px), starting at world position `origin`.

    Every pixel, low-res or high-res, integrates the scene over the same
    1 x 1 high-res-pixel footprint around its centre. A game rasterises one
    sharp sample per pixel; it does not average over the whole low-res
    pixel, so the fine detail survives (aliased) in the samples, and jitter
    can recover it.
    """
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    x0 = origin[0] + (j + 0.5) * spacing - 0.5
    y0 = origin[1] + (i + 0.5) * spacing - 0.5
    return pixel_average(lambda x, y: scene(x, y, t), x0, y0, 1.0, n=n)


VERSION = 2  # bump when scene.py or rendering changes, to invalidate the cache
CACHE = Path(__file__).resolve().parents[1] / "out" / "cache"


def make_sequence(n_frames=32, hi_size=512, scale=4, pan=(0.37, 0.21), n_quad=2):
    """Render a camera pan. `pan` is the motion per frame in high-res pixels;
    it is deliberately non-integer so frames never line up exactly.

    Rendering takes ~1 min, so results are cached in out/cache keyed by the
    arguments (delete the folder after changing scene.py)."""
    key = f"seq_v{VERSION}_{n_frames}_{hi_size}_{scale}_{pan[0]}_{pan[1]}_{n_quad}.npz"
    path = CACHE / key
    if path.exists():
        d = np.load(path)
        return [Frame(*(d[name][k] for name in FIELDS)) for k in range(n_frames)]
    frames = _render_sequence(n_frames, hi_size, scale, pan, n_quad)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.stack([getattr(f, name) for f in frames])
                                 for name in FIELDS})
    return frames


def _render_sequence(n_frames, hi_size, scale, pan, n_quad):
    lo = hi_size // scale
    pan = np.asarray(pan, float)
    jit = jitter_sequence(n_frames)
    frames = []
    for k in range(n_frames):
        cam = pan * k
        low = render(lo, lo, cam + jit[k] * scale, scale, t=k, n=n_quad)
        truth = render(hi_size, hi_size, cam, 1.0, t=k, n=n_quad)
        frames.append(Frame(low, jit[k], cam, pan if k else np.zeros(2), truth))
    return frames
