"""Module 1 demo: render the scene, generate jittered frames, and show why
Simpson's rule was chosen for pixel integration.

Run: .venv/bin/python scripts/01_frames.py   -> out/01_frames.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.frames import jitter_sequence, make_sequence, render  # noqa: E402
from tsr.quadrature import pixel_average  # noqa: E402
from tsr.scene import scene  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)

frames = make_sequence(n_frames=16)
f0, f1 = frames[0], frames[1]

# Quadrature convergence: average a 4x4 low-res pixel patch (zone-plate area)
# with increasing n, against a very fine reference.
j, i = np.meshgrid(np.arange(32), np.arange(32))
x0, y0 = 300 + 4.0 * j, 200 + 4.0 * i
ref = pixel_average(scene, x0, y0, 4.0, n=128)
ns = [2, 4, 8, 16, 32]
err = {r: [np.abs(pixel_average(scene, x0, y0, 4.0, n=n, rule=r) - ref).max() for n in ns]
       for r in ("trapezoid", "simpson")}

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(f0.truth, cmap="gray", vmin=0, vmax=1)
ax[0, 0].set_title("Ground truth 512x512 (never seen by the upscaler)")
ax[0, 1].imshow(f0.low, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0, 1].set_title("Frame 0: what the camera renders (128x128)")
ax[0, 2].imshow(f1.low - f0.low, cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="nearest")
ax[0, 2].set_title("Frame 1 - Frame 0: jitter reveals new detail")

jit = jitter_sequence(16)
ax[1, 0].scatter(jit[:, 0], jit[:, 1], c=np.arange(16), cmap="viridis")
for k, (a, b) in enumerate(jit):
    ax[1, 0].annotate(str(k), (a, b), textcoords="offset points", xytext=(4, 4))
ax[1, 0].set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal",
             title="Halton(2,3) jitter inside one low-res pixel")
ax[1, 0].grid(True)

ax[1, 1].loglog(ns, err["trapezoid"], "o-", label="Trapezoidal")
ax[1, 1].loglog(ns, err["simpson"], "s-", label="Simpson 1/3")
ax[1, 1].set(xlabel="intervals per axis n", ylabel="max pixel error",
             title="Pixel integration error vs n")
ax[1, 1].legend()
ax[1, 1].grid(True, which="both")

crop = (slice(192, 320), slice(300, 428))
ax[1, 2].imshow(np.kron(f0.low, np.ones((4, 4)))[crop], cmap="gray", vmin=0, vmax=1)
ax[1, 2].set_title("Zone plate rings, low-res: aliasing (moire)")
for a in ax.flat:
    if a not in (ax[1, 0], ax[1, 1]):
        a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "01_frames.png", dpi=110)

print("intervals:", ns)
for r, e in err.items():
    print(f"{r:9s}", " ".join(f"{v:.2e}" for v in e))
print("saved", OUT / "01_frames.png")
