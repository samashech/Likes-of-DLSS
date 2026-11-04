"""Sanity checks for the temporal core on a small, fast sequence."""
import numpy as np

from tsr.frames import Frame, jitter_sequence
from tsr.temporal import Params, run, shift_int


def test_shift_int_is_exact():
    a = np.arange(20.0).reshape(4, 5)
    b = shift_int(a, 2, -1)            # b[q] = a[q + (dx=2, dy=-1)]
    assert b[1, 0] == a[0, 2] and b[0].sum() == 0 and b[:, -2:].sum() == 0


def test_more_frames_beat_one_frame():
    # smooth stripe pattern finer than the low-res grid: needs jitter to resolve
    scene = lambda x, y: 0.5 + 0.5 * np.sin(2 * np.pi * x / 5.0)
    s, n, pan = 4, 32, np.array([0.37, 0.21])
    jit = jitter_sequence(16)
    frames = []
    for k in range(16):
        cam = pan * k
        q = np.arange(n * s) + 0.5
        lx = cam[0] + s * (np.arange(n) + 0.5 + jit[k][0])
        ly = cam[1] + s * (np.arange(n) + 0.5 + jit[k][1])
        low = scene(*np.meshgrid(lx, ly))
        truth = scene(*np.meshgrid(cam[0] + q, cam[1] + q))
        frames.append(Frame(low, jit[k], cam, pan if k else np.zeros(2), truth))
    out = run(frames, s, Params(clamp="none", sigma=0.4))
    err = [np.abs(o - f.truth)[8:-8, 8:-8].mean() for o, f in zip(out, frames)]
    assert err[-1] < 0.5 * err[0]
