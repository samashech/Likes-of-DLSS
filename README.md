# Likes of DLSS: temporal super-resolution from numerical methods

A **4× temporal upscaler** (128×128 → 512×512) in the style of NVIDIA DLSS and AMD FSR 2. It is built only from the numerical methods in our Numerical Methods lab syllabus: no neural network and no image-processing libraries. Every core method is written by hand in numpy and checked against scipy.

![Temporal accumulation vs single-frame upscaling](out/03_temporal.png)

## Headline result

| Method | Mean PSNR (32 frames) | Last frame |
|---|---|---|
| Best single-frame interpolation (bilinear) | 14.00 dB | 14.05 dB |
| **Temporal upscaler (this project)** | **20.85 dB** | **23.66 dB** |

On the last frame that's **+9.6 dB, about 3× lower RMS error**. It recovers detail that no single frame contains: the real rings of the zone plate and bars thinner than one low-res pixel. On a camera path held out from tuning it still gains +8.3 dB on the last frame (+5.9 dB mean).

## How it works

DLSS gets jittered low-res frames and motion vectors from the game, and combines many frames into one sharp image. We do the same thing classically:

1. **Render:** an analytic test scene, so ground truth is exact. Each pixel is an area integral, computed with **Simpson's 1/3 rule**. The camera jitters by a sub-pixel **Halton (2, 3)** offset every frame.
2. **Accumulate:** each new sample adds to nearby high-res pixels, forming a confidence-weighted least-squares estimate. History lives on a world-aligned grid, so it is never re-interpolated and doesn't blur from repeated resampling.
3. **Anti-ghosting:** history is clamped to what the current frame sees nearby. The clamp is widened where the **power method** (structure-tensor eigenvalues) finds edges, so it removes ghosts without cutting detail.
4. **Display:** one **Lagrange** resample, then edge-preserving anisotropic diffusion stepped with **RK4**.
5. **Tuning:** two parameters fitted to ground truth by **bisection / regula falsi / secant / Newton-Raphson**. This is our counterpart to DLSS training.
6. **Motion estimation** (optional): registration solved as a nonlinear 2×2 system by **Gauss-Newton**, cross-checked against `fsolve`.

## Syllabus coverage

| Exp | Topic | Where it's used | Evidence |
|---|---|---|---|
| 2(a–d) | Bisection, Newton-Raphson, Secant, Regula Falsi | Parameter tuning (`tsr/rootfind.py`) | `out/05_tuning.png`: convergence and cost in pipeline runs |
| 2(e) | Nonlinear systems, `linalg.solve`, `fsolve` | Motion estimation (`tsr/motion.py`) | `out/06_motion.png` |
| 3 | Lagrange interpolation | Upscaling and resampling (`tsr/interp.py`) | `out/02_spatial.png`, Lebesgue constants |
| 4 | Newton forward/backward differences | Equivalent interpolation forms; finite-difference gradients and derivatives | verified equal to Lagrange to 7×10⁻¹⁶ |
| 5, 6 | Trapezoidal, Simpson's 1/3 | Pixel integration (`tsr/quadrature.py`) | `out/01_frames.png`: O(h²) vs O(h⁴) |
| 7 | RK4 | Anisotropic diffusion (`tsr/diffusion.py`) | `out/04_edges.png`: stability limits and order 1 vs order 4 |
| 8 | Power method | Edge detection, and the spectral radius for a stable time step (`tsr/edges.py`) | convergence rate (λ₂/λ₁)² |

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests     # 16 tests
.venv/bin/python scripts/03_temporal.py # any module's figure → out/
```

The first run renders the scene, which takes about 40 s; after that it's cached in `out/cache/`.

### The demo (React)

```bash
.venv/bin/python scripts/export_web.py  # precompute every method × frame, about 3 min, 72 MB
cd web && npm install && npm run dev    # open the printed localhost URL
```

`web/` is a static React + Vite app, so no Python runs at demo time and it works offline (fonts are self-hosted). It shows:
- two methods with a draggable wipe (arrow keys move it too)
- frame scrubbing and playback
- stress-test regions: the zone plate, thin bars, and the light panel used for the ghosting test
- both camera paths, including the held-out one
- a PSNR table and one large PSNR chart you can click to seek

**Game scenes.** Add any high-resolution game screenshot as a scene:

```bash
.venv/bin/python scripts/import_scene.py image uploads/cyberpunk.jpg --name "Cyberpunk 2077"    # about 3 min
.venv/bin/python scripts/import_scene.py raw my_frames/ --name "My renders"   # your own jittered frames + frames.json
```

In image mode the screenshot becomes the ground-truth world: a simulated renderer pans across it and sends 320×180 jittered frames, which are upscaled to 1280×720. On Cyberpunk 2077 that's 25.6 dB (bilinear) vs 36.1 dB (temporal) on the last frame. Raw mode takes real rendered frames and their jitter; it has no ground truth, so no scores (`decisions.md` §22).

The view is kept in the URL, so a link reopens the exact comparison. `npm run build` produces `web/dist/`, which any static host can serve.

The UI follows the [Hallmark](https://github.com/Nutlope/hallmark) design skill (Almanac theme, app-screen scope).

`app.py` is an older Streamlit version of the same demo (`.venv/bin/streamlit run app.py`).

## Project layout

```
tsr/          the upscaler: quadrature, scene, frames, interp, temporal, edges,
              diffusion, rootfind, motion, metrics
scripts/      01–06: one demo and figure per module
tests/        hand-written methods checked against scipy / numpy.linalg
decisions.md  every core design decision, with the measurement behind it
web/          React demo (data from scripts/export_web.py)
app.py        Streamlit demo (older)
```

## Limitations

- The scene is synthetic, and the motion is a global camera pan. The world-aligned history depends on that: real per-object motion would need per-pixel screen-space reprojection, which reintroduces blur (measured in `decisions.md` §12).
- Without engine motion vectors, estimated motion is biased by aliasing (0.2–0.9 px), which costs about 2.7 dB. That is the reason DLSS requires motion vectors (§21).
- Parameters were tuned on one scene. A held-out camera path confirms the gain, but a different scene would be a stronger test.
