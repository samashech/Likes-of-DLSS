"""Module 5 demo: tune the upscaler's parameters with root-finding.

This is our version of "training": DLSS fits a network to ground-truth
renders; we fit two parameters to ground truth by solving equations.

  1. sigma (gather width): maximise mean PSNR  ->  solve  dPSNR/dsigma = 0
  2. alpha (clamp width):  largest alpha whose ghost error stays in budget
                           ->  solve  ghost(alpha) - budget = 0

Each f evaluation is a full upscaler run, so methods are compared by how
many runs they need.

Run: .venv/bin/python scripts/05_tuning.py   -> out/05_tuning.png, out/05_tuning.json
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.frames import make_sequence  # noqa: E402
from tsr.interp import upscale  # noqa: E402
from tsr.metrics import psnr  # noqa: E402
from tsr.rootfind import Counted, bisection, newton, regula_falsi, secant  # noqa: E402
from tsr.scene import SWITCH_FRAME  # noqa: E402
from tsr.temporal import Params, run  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
frames = make_sequence()
PANEL = (slice(415, 495), slice(335, 465))   # the light-switch panel
GHOST_FRAME = SWITCH_FRAME + 1
BUDGET = 0.05                                # max mean ghost error, 5% of full brightness
TOL = 1e-3

# ---- objectives (diffusion off: it is display-only and tuned separately) ----


def mean_psnr(sigma, alpha=3.0):
    out = run(frames, params=Params(sigma=sigma, alpha=alpha, diffuse_t=0))
    return np.mean([psnr(o, f.truth) for o, f in zip(out, frames)])


def ghost(alpha, sigma):
    out = run(frames[:GHOST_FRAME + 1], params=Params(sigma=sigma, alpha=alpha, diffuse_t=0))
    return np.abs(out[GHOST_FRAME][PANEL] - frames[GHOST_FRAME].truth[PANEL]).mean()


def solve_all(f_runs, g, dg, bracket, x0, x1, label):
    """Run the four methods on g(x) = 0; f_runs counts the pipeline runs."""
    res = {}
    for name, solver in [
        ("Bisection", lambda: bisection(g, *bracket, tol=TOL)),
        ("Regula falsi", lambda: regula_falsi(g, *bracket, tol=TOL)),
        ("Secant", lambda: secant(g, x0, x1, tol=TOL)),
        ("Newton-Raphson", lambda: newton(g, dg, x0, tol=TOL)),
    ]:
        f_runs.cache.clear()
        t = time.perf_counter()
        root, hist = solver()
        res[name] = dict(root=root, hist=hist, runs=f_runs.evals, seconds=time.perf_counter() - t)
        print(f"  {label} {name:15s} root {root:.4f}  iterations {len(hist):2d}  "
              f"pipeline runs {f_runs.evals:2d}  ({res[name]['seconds']:.0f}s)")
    return res


# ---- problem 1: sigma, optimum of mean PSNR --------------------------------
# dP/dsigma and d2P/dsigma2 by central differences (h = 0.01): the first and
# second derivative of the quadratic through 3 points (Newton differences).
H = 0.01
P = Counted(mean_psnr)
g_sigma = lambda s: (P(s + H) - P(s - H)) / (2 * H)
dg_sigma = lambda s: (P(s + H) - 2 * P(s) + P(s - H)) / H ** 2
print("Problem 1: sigma maximising mean PSNR (solve dPSNR/dsigma = 0)")
sig_res = solve_all(P, g_sigma, dg_sigma, (0.3, 0.6), 0.45, 0.5, "sigma")
sigma_star = sig_res["Newton-Raphson"]["root"]
sweep_s = np.linspace(0.3, 0.6, 7)
sweep_P = [P(s) for s in sweep_s]

# ---- problem 2: alpha, ghost budget -----------------------------------------
Gc = Counted(lambda a: ghost(a, sigma_star))
g_alpha = lambda a: Gc(a) - BUDGET
dg_alpha = lambda a: (Gc(a + 0.05) - Gc(a - 0.05)) / 0.1
print(f"Problem 2: alpha with ghost error = {BUDGET} (sigma = {sigma_star:.4f})")
alp_res = solve_all(Gc, g_alpha, dg_alpha, (0.0, 8.0), 1.0, 4.0, "alpha")
alpha_star = alp_res["Newton-Raphson"]["root"]
sweep_a = np.linspace(0, 8, 9)
sweep_G = [Gc(a) for a in sweep_a]

# ---- before / after ----------------------------------------------------------
old = run(frames, params=Params(sigma=0.6, alpha=3.0))
new = run(frames, params=Params(sigma=sigma_star, alpha=alpha_star))
spatial = [upscale(f.low, 4, f.jitter, m=2) for f in frames]
s_old = [psnr(o, f.truth) for o, f in zip(old, frames)]
s_new = [psnr(o, f.truth) for o, f in zip(new, frames)]
s_sp = [psnr(o, f.truth) for o, f in zip(spatial, frames)]

json.dump(dict(sigma=sig_res, alpha=alp_res, sigma_star=sigma_star, alpha_star=alpha_star,
               psnr_old=s_old, psnr_new=s_new, psnr_spatial=s_sp),
          open(OUT / "05_tuning.json", "w"), indent=1, default=float)

# ---- figure ------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
ax[0, 0].plot(sweep_s, sweep_P, "ko-")
ax[0, 0].axvline(sigma_star, color="C3", ls="--", label=f"sigma* = {sigma_star:.3f}")
ax[0, 0].set(xlabel="sigma (px)", ylabel="mean PSNR over 32 frames (dB)", title="Problem 1: best gather width")
ax[0, 0].legend()
ax[0, 0].grid(True)

ax[1, 0].plot(sweep_a, sweep_G, "ko-")
ax[1, 0].axhline(BUDGET, color="gray", ls=":", label=f"budget {BUDGET}")
ax[1, 0].axvline(alpha_star, color="C3", ls="--", label=f"alpha* = {alpha_star:.3f}")
ax[1, 0].set(xlabel="alpha", ylabel=f"ghost error, frame {GHOST_FRAME}", title="Problem 2: clamp width within ghost budget")
ax[1, 0].legend()
ax[1, 0].grid(True)

for row, (res, star, name) in enumerate([(sig_res, sigma_star, "sigma"), (alp_res, alpha_star, "alpha")]):
    a = ax[row, 1]
    for m, r in res.items():
        a.semilogy(np.arange(1, len(r["hist"]) + 1), np.maximum(np.abs(np.array(r["hist"]) - star), 1e-7),
                   "o-", ms=4, label=m)
    a.set(xlabel="iteration", ylabel=f"|{name}_k - {name}*|", title=f"Convergence ({name})")
    a.legend(fontsize=8)
    a.grid(True, which="both")

names = list(sig_res)
x = np.arange(len(names))
ax[0, 2].bar(x - 0.2, [sig_res[n]["runs"] for n in names], 0.4, label="sigma problem")
ax[0, 2].bar(x + 0.2, [alp_res[n]["runs"] for n in names], 0.4, label="alpha problem")
ax[0, 2].set_xticks(x, names, fontsize=8)
ax[0, 2].set(ylabel="full upscaler runs", title=f"Real cost: pipeline runs to tol {TOL}")
ax[0, 2].legend()

ax[1, 2].plot(s_sp, label="single-frame bilinear")
ax[1, 2].plot(s_old, label="hand-picked (sigma 0.6, alpha 3)")
ax[1, 2].plot(s_new, label=f"tuned (sigma {sigma_star:.2f}, alpha {alpha_star:.2f})")
ax[1, 2].axvline(SWITCH_FRAME, color="gray", ls="--")
ax[1, 2].set(xlabel="frame", ylabel="PSNR (dB)", title="Before vs after tuning (with diffusion)")
ax[1, 2].legend(fontsize=8)
ax[1, 2].grid(True)
fig.tight_layout()
fig.savefig(OUT / "05_tuning.png", dpi=100)

print(f"sigma* {sigma_star:.4f}  alpha* {alpha_star:.4f}")
print(f"mean PSNR: hand-picked {np.mean(s_old):.2f}  tuned {np.mean(s_new):.2f}  spatial {np.mean(s_sp):.2f}")
print(f"frame {GHOST_FRAME}: hand-picked {s_old[GHOST_FRAME]:.2f} tuned {s_new[GHOST_FRAME]:.2f};  "
      f"last: {s_old[-1]:.2f} -> {s_new[-1]:.2f}")
print("saved", OUT / "05_tuning.png")
