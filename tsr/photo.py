"""A real image as the scene: simulate the game's renderer on a screenshot.

The screenshot (downscaled to the working size) is the ground truth world.
A camera pans across it; every frame the "renderer" takes one sample per
low-res pixel at a Halton-jittered position, exactly as the analytic scene
is sampled in frames.py. Between pixels the world is read with cubic
Lagrange interpolation (Exp 3), so any sub-pixel position has a value.

Colour images are handled one channel at a time: each channel becomes its
own list of Frame objects and goes through the unchanged upscaler.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .frames import Frame, jitter_sequence
from .interp import sample_grid

MARGIN = 16  # extra world pixels around the view, so the camera can pan


def load_world(path, view=(1280, 720)):
    """Image -> float RGB array of size (view + 2*MARGIN), cover-cropped.

    Downscaling from 4K uses Lanczos, a good-quality reference resample; the
    result is treated as exact ground truth from here on.
    """
    w, h = view[0] + 2 * MARGIN, view[1] + 2 * MARGIN
    im = Image.open(path).convert("RGB")
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)), float) / 255.0


def render_channel(world, cam, positions_x, positions_y):
    """World values at (positions + cam), pixel centres at integers."""
    return sample_grid(world, positions_x + cam[0] + MARGIN, positions_y + cam[1] + MARGIN, m=4)


def make_photo_sequence(world, n_frames=24, view=(1280, 720), scale=4, pan=(0.37, 0.21)):
    """Returns one list of Frames per colour channel (same jitter and motion)."""
    W, H = view
    lw, lh = W // scale, H // scale
    pan = np.asarray(pan, float)
    jit = jitter_sequence(n_frames)
    qx, qy = np.arange(W, dtype=float), np.arange(H, dtype=float)
    channels = []
    for c in range(world.shape[2]):
        ch = world[..., c]
        seq = []
        for k in range(n_frames):
            cam = pan * k
            # low-res sample j sits at scale*(j + 0.5 + jitter) - 0.5 in high-res pixel units
            lx = scale * (np.arange(lw) + 0.5 + jit[k][0]) - 0.5
            ly = scale * (np.arange(lh) + 0.5 + jit[k][1]) - 0.5
            low = np.clip(render_channel(ch, cam, lx, ly), 0, 1)
            truth = np.clip(render_channel(ch, cam, qx, qy), 0, 1)
            seq.append(Frame(low, jit[k], cam, pan if k else np.zeros(2), truth))
        channels.append(seq)
    return channels


def psnr_rgb(img, ref, border=8):
    """PSNR over all three channels, ignoring a border (as metrics.psnr)."""
    sl = (slice(border, -border), slice(border, -border))
    return 10 * np.log10(1.0 / np.mean((img[sl] - ref[sl]) ** 2))


def stack(channels_out, k):
    """Frame k of each channel's output -> one HxWx3 image."""
    return np.stack([ch[k] for ch in channels_out], -1)


def save_rgb(img, path, quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=quality, method=4)
