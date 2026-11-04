"""Module 4 demo: power method edges, edge-aware clamp, RK4 diffusion.

Run: .venv/bin/python scripts/04_edges.py   -> out/04_edges.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.diffusion import (diffusion_tensor, integrate, operator,  # noqa: E402
                           spectral_radius)
from tsr.edges import power_method, structure_tensor  # noqa: E402
from tsr.frames import make_sequence  # noqa: E402
from tsr.metrics import psnr  # noqa: E402
from tsr.scene import SWITCH_FRAME  # noqa: E402
from tsr.temporal import Params, run  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
frames = make_sequence()

# --- Power method convergence vs eigenvalue ratio ------------------------
a, b, c = structure_tensor(frames[0].truth, 1.0)
l1, l2, (vx, vy), _, trace = power_method(a, b, c, iters=40, tol=0, history=True)
ev = np.linalg.eigvalsh(np.stack([np.stack([a, b], -1), np.stack([b, c], -1)], -2))[..., 1]
ratio = np.abs(l2) / np.maximum(l1, 1e-30)
strong = l1 > 1e-3                       # skip flat pixels, nothing to measure there
bins = [(0.0, 0.1, "edge (l2/l1 < 0.1)"), (0.4, 0.6, "mixed (0.4-0.6)"), (0.8, 0.95, "texture (0.8-0.95)")]

# --- Clamp variants ------------------------------------------------------
variants = {
    "no clamp": Params(clamp="none", diffuse_t=0),
    "min/max clamp": Params(clamp="minmax", diffuse_t=0),
    "edge-aware clamp": Params(clamp="edge", diffuse_t=0),
    "edge-aware clamp + RK4 diffusion": Params(),
}
outs = {k: run(frames, params=p) for k, p in variants.items()}
scores = {k: [psnr(o, f.truth) for o, f in zip(v, frames)] for k, v in outs.items()}

# --- Euler vs RK4 on the diffusion ODE -----------------------------------
u0 = outs["edge-aware clamp"][2]
A = operator(diffusion_tensor(u0))
rho = spectral_radius(A, u0.shape)
T = 1.0
ref = integrate(A, u0, T, 0.01, "rk4")
dts = [0.025, 0.05, 0.1, 0.2, 0.25, 0.3]
err = {m: [np.abs(integrate(A, u0, T, dt, m) - ref).max() for dt in dts] for m in ("euler", "rk4")}

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 4)

ax = fig.add_subplot(gs[0, 0])
for lo, hi, name in bins:
    sel = strong & (ratio >= lo) & (ratio < hi)
    e = [np.median(np.abs(t[sel] - ev[sel]) / ev[sel]) for t in trace]
    ax.semilogy(np.arange(1, 41), np.maximum(e, 1e-17), label=name)
ax.set(xlabel="power iterations", ylabel="median relative error of l1",
       title="Power method: speed set by l2/l1", ylim=(1e-17, 1))
ax.legend(fontsize=8)
ax.grid(True, which="both")

crop = (slice(180, 300), slice(290, 410))
ax = fig.add_subplot(gs[0, 1])
ax.imshow(np.sqrt(np.maximum(l1, 0))[crop], cmap="magma")
ys, xs = np.mgrid[0:120:8, 0:120:8]
vxc, vyc = vx[crop][::8, ::8], vy[crop][::8, ::8]
ax.quiver(xs, ys, -vyc, vxc, color="cyan", angles="xy", scale_units="xy", scale=0.2,
          headwidth=0, headlength=0, headaxislength=0, pivot="mid")
ax.set_title("Edge strength sqrt(l1); lines = edge direction v2", fontsize=9)
ax.axis("off")

ax = fig.add_subplot(gs[0, 2:])
for k, s in scores.items():
    ax.plot(s, "o-", ms=3, label=k)
ax.axvline(SWITCH_FRAME, color="gray", ls="--")
ax.set(xlabel="frame", ylabel="PSNR (dB)", title="Edge-aware clamp keeps detail and still removes the ghost")
ax.legend(fontsize=8)
ax.grid(True)

ax = fig.add_subplot(gs[1, 0])
ax.loglog(dts, err["euler"], "o-", label="Euler (order 1)")
ax.loglog(dts, err["rk4"], "s-", label="RK4 (order 4)")
ax.axvline(2.0 / rho, color="C0", ls=":", label=f"Euler limit 2/rho = {2 / rho:.3f}")
ax.axvline(2.785 / rho, color="C1", ls=":", label=f"RK4 limit 2.785/rho = {2.785 / rho:.3f}")
ax.set(xlabel="time step dt", ylabel="max error at t=1", title=f"Accuracy and stability (rho = {rho:.2f})")
ax.legend(fontsize=7)
ax.grid(True, which="both")

# Stability: march many steps and watch the largest value
steps = 60
ax = fig.add_subplot(gs[1, 1])
blown = None
for method, dt, style in (("euler", 0.24, "C0-"), ("euler", 0.30, "C0--"), ("rk4", 0.30, "C1-"), ("rk4", 0.37, "C1--")):
    u, peak = u0.copy(), []
    step = {"euler": lambda v: v + dt * A(v),
            "rk4": lambda v: integrate(A, v, dt, dt, "rk4")}[method]
    for _ in range(steps):
        u = step(u)
        peak.append(np.abs(u).max())
    ax.semilogy(peak, style, label=f"{method} dt={dt} ({'stable' if dt * rho < (2 if method == 'euler' else 2.785) else 'unstable'})")
    if method == "euler" and dt == 0.30:
        blown = u
ax.set(xlabel="time steps", ylabel="max |u|", title="Inside vs outside the stability region")
ax.legend(fontsize=7)
ax.grid(True, which="both")

ax = fig.add_subplot(gs[1, 2])
ax.imshow(np.clip(blown, 0, 1)[crop], cmap="gray", vmin=0, vmax=1)
ax.set_title(f"Euler dt=0.3 after {steps} steps: blown up\n(max |u| = {np.abs(blown).max():.2g})", fontsize=9)
ax.axis("off")

ax = fig.add_subplot(gs[1, 3])
ax.imshow(np.abs(outs["min/max clamp"][19] - frames[19].truth)[crop]
          - np.abs(outs["edge-aware clamp"][19] - frames[19].truth)[crop], cmap="RdBu", vmin=-0.3, vmax=0.3)
ax.set_title("Error saved by edge-aware clamp, frame 19\n(blue = better)", fontsize=9)
ax.axis("off")

k = 2
crop2 = (slice(405, 505), slice(50, 200))
panels = [("Ground truth", frames[k].truth), ("Edge clamp, raw", outs["edge-aware clamp"][k]),
          ("Edge clamp + RK4 diffusion", outs["edge-aware clamp + RK4 diffusion"][k])]
for col, (name, img) in enumerate(panels):
    ax = fig.add_subplot(gs[2, col])
    ax.imshow(img[crop], cmap="gray", vmin=0, vmax=1)
    lab = name if name == "Ground truth" else f"{name}\n{psnr(img, frames[k].truth):.2f} dB"
    ax.set_title(f"Frame {k}: {lab}", fontsize=9)
    ax.axis("off")
ax = fig.add_subplot(gs[2, 3])
ax.imshow(outs["edge-aware clamp + RK4 diffusion"][k][crop2], cmap="gray", vmin=0, vmax=1)
ax.set_title("Frame 2 bars after diffusion", fontsize=9)
ax.axis("off")

fig.tight_layout()
fig.savefig(OUT / "04_edges.png", dpi=100)

print(f"spectral radius {rho:.3f}; Euler limit {2 / rho:.3f}, RK4 limit {2.785 / rho:.3f}")
print("Euler err", " ".join(f"{e:.1e}" for e in err["euler"]))
print("RK4 err  ", " ".join(f"{e:.1e}" for e in err["rk4"]))
for name, s in scores.items():
    print(f"{name:34s} f2 {s[2]:.2f} f19 {s[19]:.2f} f21 {s[21]:.2f} last {s[-1]:.2f}")
print("saved", OUT / "04_edges.png")
