"""Interactive demo of the temporal super-resolution upscaler.

Run: .venv/bin/streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

from tsr.frames import make_sequence
from tsr.interp import upscale
from tsr.metrics import psnr
from tsr.scene import SWITCH_FRAME
from tsr.temporal import Params, TemporalUpscaler, run

SCALE = 4
PATHS = {"Tuning path (0.37, 0.21) px/frame": (0.37, 0.21),
         "Held-out path (-0.29, 0.43) px/frame": (-0.29, 0.43)}
REGIONS = {"Full frame": (slice(0, 512), slice(0, 512)),
           "Zone plate (aliasing test)": (slice(196, 316), slice(126, 246)),
           "Thin bars (resolution test)": (slice(405, 505), slice(50, 210)),
           "Light panel (ghosting test)": (slice(395, 505), slice(315, 485)),
           "Checkerboard edges": (slice(10, 130), slice(10, 170))}


def spatial(m):
    return lambda frames: [upscale(f.low, SCALE, f.jitter, m) for f in frames]


def temporal(**kw):
    motion = kw.pop("motion", "given")
    return lambda frames: run(frames, upscaler=TemporalUpscaler(SCALE, Params(**kw), motion=motion))


METHODS = {
    "Ground truth": lambda frames: [f.truth for f in frames],
    "Low-res input (nearest)": spatial(1),
    "Bilinear (single frame)": spatial(2),
    "Lagrange cubic (single frame)": spatial(4),
    "Temporal: no clamp": temporal(clamp="none", diffuse_t=0),
    "Temporal: min/max clamp": temporal(clamp="minmax", diffuse_t=0),
    "Temporal: edge-aware clamp": temporal(diffuse_t=0),
    "Temporal: full (edge clamp + RK4)": temporal(),
    "Temporal: full, motion estimated": temporal(motion="estimated"),
}


@st.cache_resource(show_spinner="Rendering 32 frames of the scene (first time only, ~40 s)...")
def frames_for(pan):
    return make_sequence(pan=pan)


@st.cache_resource(show_spinner="Running the upscaler over all 32 frames...")
def outputs(method, pan):
    frames = frames_for(pan)
    out = METHODS[method](frames)
    scores = [psnr(o, f.truth) for o, f in zip(out, frames)]
    return np.stack([(np.clip(o, 0, 1) * 255).astype(np.uint8) for o in out]), scores


def zoom(img, region, width=720):
    crop = img[region]
    k = max(1, width // crop.shape[1])
    return np.kron(crop, np.ones((k, k), np.uint8)), k


def wipe(a, b, frac):
    """Left part from a, right part from b, with a red divider."""
    out = np.stack([b] * 3, -1)
    cut = int(round(frac * a.shape[1]))
    out[:, :cut] = np.stack([a[:, :cut]] * 3, -1)
    if 0 < cut < a.shape[1]:
        out[:, max(cut - 1, 0):cut + 1] = (230, 40, 40)
    return out


st.set_page_config(page_title="Temporal Upscaler", layout="wide")
st.title("Temporal super-resolution from numerical methods")
st.caption("128×128 jittered low-res frames → 512×512, DLSS-style: no neural network, "
           "only interpolation, quadrature, eigenvalues, ODEs and root-finding.")

with st.sidebar:
    st.header("Compare")
    left = st.selectbox("Left side", list(METHODS), index=3)
    right = st.selectbox("Right side", list(METHODS), index=7)
    region_name = st.selectbox("Region", list(REGIONS), index=1)
    path_name = st.selectbox("Camera path", list(PATHS))
    st.divider()
    st.markdown(f"The light panel switches on at **frame {SWITCH_FRAME}**. Nothing moves, so motion "
                "vectors cannot warn the upscaler: watch the no-clamp version ghost.")

pan = PATHS[path_name]
frames = frames_for(pan)
imgs_l, ps_l = outputs(left, pan)
imgs_r, ps_r = outputs(right, pan)
region = REGIONS[region_name]

c1, c2 = st.columns([3, 1])
with c1:
    k = st.slider("Frame", 0, len(frames) - 1, SWITCH_FRAME - 1)
with c2:
    frac = st.slider("Wipe position", 0.0, 1.0, 0.5)
play = st.button("▶ Play all frames")

view = st.empty()
stats = st.empty()


def show(i):
    a, z = zoom(imgs_l[i], region)
    b, _ = zoom(imgs_r[i], region)
    view.image(wipe(a, b, frac), caption=f"Frame {i}  ·  left: {left}  |  right: {right}  ·  zoom ×{z}")
    m1, m2, m3 = stats.columns(3)
    m1.metric(f"Left PSNR, frame {i}", f"{ps_l[i]:.2f} dB")
    m2.metric(f"Right PSNR, frame {i}", f"{ps_r[i]:.2f} dB", f"{ps_r[i] - ps_l[i]:+.2f} dB vs left")
    m3.metric("Error reduction (RMS)", f"{10 ** ((ps_r[i] - ps_l[i]) / 20):.2f}×")


if play:
    for i in range(len(frames)):
        show(i)
        time.sleep(0.25)
else:
    show(k)

st.subheader("Quality over time")
chart = pd.DataFrame({left: ps_l, right: ps_r}, index=pd.Index(range(len(frames)), name="frame"))
st.line_chart(chart, y_label="PSNR (dB, higher is better)")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**What the game renders at frame {k}** (128×128, one sharp sample per pixel)")
    low = (np.clip(frames[k].low, 0, 1) * 255).astype(np.uint8)
    st.image(np.kron(low, np.ones((3, 3), np.uint8)))
    j = frames[k].jitter
    st.caption(f"Sub-pixel jitter this frame: ({j[0]:+.3f}, {j[1]:+.3f}) low-res px (Halton 2,3)")
with col_b:
    with st.expander("How it works", expanded=True):
        st.markdown("""
1. **Render** each low-res pixel as a 1×1 px area average, computed with **Simpson's 1/3 rule**. The camera jitters by a Halton sub-pixel offset every frame.
2. **Gather:** every new sample adds to nearby high-res pixels with Gaussian weights.
3. **Accumulate:** a confidence-weighted running mean, on a world-aligned grid, so the history is never re-interpolated (no numerical diffusion).
4. **Clamp:** history is limited to the range of nearby current samples. The range is widened by the edge strength from the **power method** (structure-tensor eigenvalues), which stops ghosts without cutting detail.
5. **Display:** one **Lagrange** resample, then edge-preserving diffusion stepped with **RK4**.
6. **Tuning:** σ and α were found with **bisection, regula falsi, secant and Newton-Raphson**.
7. **Motion estimation** (optional) solves a 2×2 nonlinear system by **Gauss-Newton**, cross-checked against `fsolve`.

Design decisions and measurements: `decisions.md`.
""")
