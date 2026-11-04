"""Module 3 demo: temporal super-resolution vs single-frame upscaling.

Run: .venv/bin/python scripts/03_temporal.py   -> out/03_temporal.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.frames import make_sequence  # noqa: E402
from tsr.interp import sample, upscale  # noqa: E402
from tsr.metrics import psnr  # noqa: E402
from tsr.scene import SWITCH_FRAME  # noqa: E402
from tsr.temporal import Params, run  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"

frames = make_sequence()
spatial = [upscale(f.low, 4, f.jitter, m=4) for f in frames]
runs = {
    "Spatial only (Lagrange cubic)": spatial,
    "Temporal, screen-space history": run(frames, params=Params(clamp="none"), mode="screen"),
    "Temporal, world-aligned history": run(frames, params=Params(clamp="none")),
    "Temporal, world + min/max clamp": run(frames, params=Params(clamp="minmax")),
}
scores = {name: [psnr(o, f.truth) for o, f in zip(out, frames)] for name, out in runs.items()}

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(4, 4)

a = fig.add_subplot(gs[0, :2])
for name, s in scores.items():
    a.plot(s, "o-", ms=3, label=name)
a.axvline(SWITCH_FRAME, color="gray", ls="--")
a.text(SWITCH_FRAME + 0.3, a.get_ylim()[0] + 0.5, "light switches on", color="gray")
a.set(xlabel="frame", ylabel="PSNR (dB)", title="Quality vs frame: temporal accumulation recovers detail")
a.legend(fontsize=8)
a.grid(True)

# Numerical diffusion: how much of a fine (period 4 px) pattern survives
# being resampled by a 0.37 px shift every frame.
a = fig.add_subplot(gs[0, 2:])
x = np.arange(200.0)
img = np.tile(np.cos(2 * np.pi * x / 4), (8, 1))
for m in (2, 4, 8):
    keep = np.abs(sample(img, x[None, :] + 0.37 + 0 * img, 3 + 0 * img, m=m)[3, 20:180]).max()
    a.semilogy(np.arange(17), keep ** np.arange(17) * 100, "o-", ms=3, label=f"resample m={m}: {keep:.0%}/frame")
a.axhline(100, color="k", label="world-aligned grid: no repeated resample")
a.set(xlabel="frames of history", ylabel="% of fine detail left",
      title="Why screen-space history blurs: numerical diffusion")
a.legend(fontsize=8)
a.grid(True, which="both")

k = SWITCH_FRAME - 1
crops = {"zone plate": (slice(180, 300), slice(290, 410)), "bars": (slice(405, 505), slice(50, 200))}
cols = ["Ground truth", "Spatial only (Lagrange cubic)", "Temporal, screen-space history",
        "Temporal, world-aligned history"]
for r, (cname, crop) in enumerate(crops.items(), start=1):
    for c, name in enumerate(cols):
        a = fig.add_subplot(gs[r, c])
        img = frames[k].truth if name == "Ground truth" else runs[name][k]
        a.imshow(img[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if r == 1:
            a.set_title(name if name == "Ground truth" else f"{name}\n{scores[name][k]:.1f} dB (frame {k})",
                        fontsize=9)
        a.axis("off")

k2 = SWITCH_FRAME + 2
crop = (slice(395, 505), slice(310, 490))
ghost = [("Truth, frame %d" % k2, frames[k2].truth),
         ("No clamp: ghost of old checkerboard", runs["Temporal, world-aligned history"][k2]),
         ("Min/max clamp: ghost removed", runs["Temporal, world + min/max clamp"][k2]),
         ("Spatial only", spatial[k2])]
for c, (name, img) in enumerate(ghost):
    a = fig.add_subplot(gs[3, c])
    a.imshow(img[crop], cmap="gray", vmin=0, vmax=1)
    a.set_title(name, fontsize=9)
    a.axis("off")

fig.tight_layout()
fig.savefig(OUT / "03_temporal.png", dpi=100)

for name, s in scores.items():
    print(f"{name:34s} frame0 {s[0]:5.2f}  frame{SWITCH_FRAME - 1} {s[SWITCH_FRAME - 1]:5.2f}  "
          f"frame{SWITCH_FRAME + 2} {s[SWITCH_FRAME + 2]:5.2f}  last {s[-1]:5.2f}")
print("saved", OUT / "03_temporal.png")
