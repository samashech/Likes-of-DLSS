"""Module 2 demo: single-frame upscaling, the baseline temporal must beat.

Run: .venv/bin/python scripts/02_spatial.py   -> out/02_spatial.png
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.frames import make_sequence  # noqa: E402
from tsr.interp import upscale  # noqa: E402
from tsr.metrics import psnr  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
S = 4

f = make_sequence()[0]
methods = {"Nearest (m=1)": 1, "Bilinear (m=2)": 2, "Lagrange cubic (m=4)": 4}
ups = {name: upscale(f.low, S, f.jitter, m) for name, m in methods.items()}

# Lagrange vs Newton: same polynomial, compare result and speed
t = time.perf_counter(); lag = upscale(f.low, S, f.jitter, 4, "lagrange"); t_lag = time.perf_counter() - t
t = time.perf_counter(); new = upscale(f.low, S, f.jitter, 4, "newton"); t_new = time.perf_counter() - t

# Error vs polynomial degree: does a higher degree keep helping?
orders = [1, 2, 4, 6, 8, 10, 12]
scores = [psnr(upscale(f.low, S, f.jitter, m), f.truth) for m in orders]

# Overshoot on an edge: one row through the bar chart
row = 455
profile = {m: upscale(f.low, S, f.jitter, m)[row, 50:200] for m in (2, 4, 10)}

fig = plt.figure(figsize=(16, 10))
crop_zone = (slice(180, 330), slice(290, 440))
crop_bars = (slice(405, 505), slice(50, 200))
panels = [("Ground truth", f.truth)] + list(ups.items())
for c, (name, img) in enumerate(panels):
    for r, crop in enumerate((crop_zone, crop_bars)):
        a = fig.add_subplot(3, 4, 1 + c + 4 * r)
        a.imshow(img[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if r == 0:
            a.set_title(name if name == "Ground truth" else f"{name}\n{psnr(img, f.truth):.2f} dB")
        a.axis("off")

a = fig.add_subplot(3, 4, 9)
a.plot(orders, scores, "o-")
a.set(xlabel="points per axis m (degree m-1)", ylabel="PSNR (dB)",
      title="Higher degree amplifies aliasing")
a.grid(True)

a = fig.add_subplot(3, 4, 10)
a.plot(np.arange(50, 200), f.truth[row, 50:200], "k", lw=2, label="truth")
for m, p in profile.items():
    a.plot(np.arange(50, 200), p, label=f"m={m}")
a.set(xlabel="x (px)", ylabel="brightness", title=f"Row {row}: thin bars blur to grey")
a.legend(fontsize=8)
a.grid(True)

a = fig.add_subplot(3, 4, 11)
a.imshow(np.abs(lag - f.truth)[crop_zone], cmap="magma", vmin=0, vmax=0.5)
a.set_title("|Lagrange cubic - truth|: error = lost detail")
a.axis("off")

a = fig.add_subplot(3, 4, 12)
a.bar(["Lagrange", "Newton"], [t_lag * 1e3, t_new * 1e3], color=["C0", "C1"])
a.set(ylabel="ms per 512x512 upscale",
      title=f"Same polynomial (max diff {np.abs(lag - new).max():.0e})")
fig.tight_layout()
fig.savefig(OUT / "02_spatial.png", dpi=100)

for name, img in ups.items():
    print(f"{name:22s} {psnr(img, f.truth):6.2f} dB")
print(f"Lagrange vs Newton max diff {np.abs(lag - new).max():.1e}; "
      f"time {t_lag * 1e3:.0f} ms vs {t_new * 1e3:.0f} ms")
print("PSNR by m:", dict(zip(orders, np.round(scores, 2))))
print("saved", OUT / "02_spatial.png")
