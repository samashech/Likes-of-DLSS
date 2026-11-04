"""Module 6 demo: can the upscaler find the camera motion itself?

Run: .venv/bin/python scripts/06_motion.py   -> out/06_motion.png
"""
import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.frames import make_sequence, render  # noqa: E402
from tsr.interp import upscale  # noqa: E402
from tsr.metrics import psnr  # noqa: E402
from tsr.motion import Registration, gauss_newton  # noqa: E402
from tsr.temporal import TemporalUpscaler, run  # noqa: E402
from scipy.optimize import fsolve  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
frames = make_sequence()

# 1. Registration accuracy depends on the reference image ------------------
k = 8
prev, f = frames[k - 1], frames[k]
origin = np.rint(prev.camera).astype(int)
off = prev.camera - origin
refs = {"ground truth": render(512, 512, origin.astype(float), 1.0, t=k - 1),
        "1-frame Lagrange upscale": upscale(prev.low, 4, np.asarray(prev.jitter) + off / 4)}
regions = {"checkerboard": (slice(2, 30), slice(2, 126)), "zone plate (aliased)": (slice(36, 92), slice(36, 92))}
acc = {}
gn_hist = None
for rname, R in refs.items():
    for gname, (ys, xs) in regions.items():
        reg = Registration(R, origin, prev.camera, f.low, f.jitter, 4)
        m = np.zeros_like(reg.mask)
        m[ys, xs] = True
        reg.mask &= m
        d, h = gauss_newton(reg, np.zeros(2), robust=False)
        acc[(rname, gname)] = np.linalg.norm(d - f.motion)
        if rname == "ground truth" and gname == "checkerboard":
            gn_hist = h
            d_fsolve = fsolve(reg.F, np.zeros(2), xtol=1e-10)

# 2. Full sequence: exact / noisy / refined / estimated-only ----------------
rng = np.random.default_rng(0)
noise = rng.normal(0, 0.15, (len(frames), 2))
noise[0] = 0
noisy = [dataclasses.replace(fr, motion=fr.motion + n) for fr, n in zip(frames, noise)]
cases = {"exact motion vectors": ("given", frames), "noisy vectors (0.15 px)": ("given", noisy),
         "noisy + refined by registration": ("refined", noisy), "estimated only (no vectors)": ("estimated", frames)}
res = {}
for name, (src, seq) in cases.items():
    tu = TemporalUpscaler(motion=src)
    out = run(seq, upscaler=tu)
    cam = np.cumsum(np.array(tu.motions), axis=0)
    true_cam = np.array([fr.camera for fr in frames])
    res[name] = dict(camerr=np.linalg.norm(cam - true_cam, axis=1),
                     psnr=[psnr(o, fr.truth) for o, fr in zip(out, frames)])

fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
names = list(regions)
x = np.arange(len(names))
for i, rname in enumerate(refs):
    ax[0].bar(x + (i - 0.5) * 0.4, [acc[(rname, g)] for g in names], 0.4, label=f"reference: {rname}")
ax[0].set_yscale("log")
ax[0].set_xticks(x, names)
ax[0].set(ylabel="motion error (px)", title=f"Registration error, frame {k}")
ax[0].legend(fontsize=8)

err = [np.linalg.norm(h - f.motion) for h in gn_hist]
ax[1].semilogy(err, "o-", label="Gauss-Newton (ours)")
ax[1].axhline(np.linalg.norm(d_fsolve - f.motion), color="C1", ls="--", label="scipy fsolve result")
ax[1].set(xlabel="iteration", ylabel="|d_k - d_true| (px)", title="Newton for systems, perfect reference")
ax[1].legend(fontsize=8)
ax[1].grid(True, which="both")

for name, r in res.items():
    ax[2].plot(r["camerr"], label=name)
    ax[3].plot(r["psnr"], label=name)
ax[2].set(xlabel="frame", ylabel="camera position error (px)", title="Drift: vector errors add up, registration anchors")
ax[2].legend(fontsize=8)
ax[2].grid(True)
ax[3].set(xlabel="frame", ylabel="PSNR (dB)", title="Upscaler quality by motion source")
ax[3].legend(fontsize=8)
ax[3].grid(True)
fig.tight_layout()
fig.savefig(OUT / "06_motion.png", dpi=100)

for key, v in acc.items():
    print(f"frame {k} reference {key[0]:25s} {key[1]:22s} error {v:.4f} px")
print(f"GN iterations {len(gn_hist) - 1}, GN vs fsolve difference {np.linalg.norm(gn_hist[-1] - d_fsolve):.1e} px")
for name, r in res.items():
    print(f"{name:32s} final drift {r['camerr'][-1]:.2f} px  max drift {r['camerr'].max():.2f}  "
          f"PSNR mean {np.mean(r['psnr']):.2f} last {r['psnr'][-1]:.2f}")
print("saved", OUT / "06_motion.png")
