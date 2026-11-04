# Understanding the project: Likes of DLSS

A complete guide to the project: what it is, what it does, how it's built, which numerical methods it uses and why, the tech stack, a glossary, and a 50-question quiz with answers.

---

## 1. The definition, in one paragraph

**Likes of DLSS is a temporal super-resolution upscaler built entirely from classical numerical methods.** It takes a sequence of low-resolution frames (for example 128×128 or 320×180). Each frame is rendered with a tiny sub-pixel camera shift called *jitter*. The upscaler combines those frames into a sharp image at 4× the resolution (512×512 or 1280×720).

This is the same core idea as NVIDIA DLSS, AMD FSR 2 and Unreal's TAAU. The difference is that a neural network decides how much of the old frames to trust in DLSS, while here that job is done by hand-written numerical methods: interpolation, quadrature, eigenvalues, ODE solvers, root-finding and nonlinear systems.

**Headline result:** on the last frame it is **+9.6 dB** better than the best single-frame method on the synthetic test scene, and **+8.8 to +10.6 dB** better on four real game screenshots. That means roughly **3× lower error**.

---

## 2. What the project does, in detail

### 2.1 The problem

A game that renders at low resolution runs fast but looks blurry and *aliased*: it has jagged edges, shimmering, and fake patterns (moiré) on fine detail. The obvious fix is to upscale each frame by interpolation (bilinear, cubic, Lanczos). But a single low-res frame simply **does not contain** detail finer than its pixels, and interpolation cannot invent it.

The project measures this limit: no single-frame method gets above about 14 dB on the test scene, and the sharper the filter, the worse it does.

### 2.2 The idea

Shift the camera by a different fraction of a pixel every frame. Each frame then samples the scene at different points. After 16 frames, every low-res pixel has been sampled at 16 well-spread spots, which is 16× more information than any one frame holds. If you know where each sample landed (the jitter) and how the camera moved (the motion vectors), you can pile the samples into a high-resolution image. That is temporal accumulation, the heart of DLSS.

### 2.3 What goes in and what comes out

| | Input | Output |
|---|---|---|
| Test scene | 32 frames of 128×128, Halton jitter, camera pan (0.37, 0.21) px/frame | 512×512 frames, scored against exact ground truth |
| Game scenes | a screenshot (Cyberpunk 2077, Ghost of Tsushima, Phantom Blade Zero, Sifu) as the "world"; 24 simulated 320×180 jittered frames | 1280×720 colour frames, scored against the screenshot |
| Raw frames | your own low-res jittered frames plus a `frames.json` with each frame's jitter and motion | 4× upscaled frames; no scores, because there is no ground truth |

### 2.4 The pieces

1. **A synthetic "game".** An analytic scene defined at every real point: a rotated checkerboard, a zone plate (rings that get finer outward, a classic aliasing test), thin bars, and a light panel that switches on at frame 20 (a ghosting test).
2. **A frame generator** that renders jittered low-res frames, the motion vectors, and the high-res ground truth.
3. **Single-frame baselines:** nearest, bilinear, Lagrange cubic, Lanczos-3.
4. **The temporal upscaler:** gather, accumulate, clamp, display, diffuse.
5. **Tuning** of its parameters by root-finding.
6. **Motion estimation**, an optional mode in which the upscaler finds the camera motion itself.
7. **Game mode:** a screenshot becomes the world, and a simulated renderer produces the jittered frames.
8. **A React demo:** a before/after wipe, frame playback, region zoom, an error heatmap, PSNR charts, a benchmarks page, and in-page import of your own images or frames.

---

## 3. How it is made: the pipeline, step by step

### Step 0: Rendering (the "game engine"), in `tsr/scene.py` and `tsr/frames.py`
- The scene is a formula `scene(x, y, t)` giving brightness at any real point, measured in high-res pixel units. Edges are softened with `tanh` over 0.3 px so the integrand is smooth.
- **Every pixel is an area average** over a 1×1 high-res-pixel footprint around its centre, computed with a 2D **Simpson's 1/3 rule** at 0.5 px sample spacing. Low-res pixels use the same 1×1 footprint, just spaced 4 px apart and shifted by the jitter. This models a game rasteriser, which takes one sharp sample per pixel; that is why games alias.
- **Jitter** comes from the **Halton (2, 3)** sequence, cycling every 16 frames, as an offset inside one low-res pixel in [−0.5, 0.5)².
- The **camera pans** by (0.37, 0.21) high-res px per frame. These are deliberately non-integer so frames never line up. The pan is handed to the upscaler as **motion vectors**, as a game engine hands them to DLSS.

### Step 1: Gather, in `tsr/temporal.py: gather`
For every high-res pixel, take the 3×3 nearest low-res samples of the *current* frame and weight each by a Gaussian of its distance: w = exp(−d²/2σ²), with σ = 0.383 px after tuning. The result is a new estimate **C** and its confidence **Wn**, the sum of the weights.

### Step 2: Accumulate (the running weighted mean)
Each high-res pixel keeps a history value **H** and a confidence **W**. Each frame:

  **H ← (W·H + Wn·C) / (W + Wn)**,  **W ← W + Wn**

This is the **weighted least-squares** estimate of the pixel from every sample it has ever seen. W is capped at **w_max = 16**, so old evidence fades and the image can react to change.

### Step 3: World-aligned history (no numerical diffusion)
The textbook approach (TAA) re-interpolates the history to the new camera position every frame. Each re-interpolation is a mild blur, and the blurs compound: after 16 frames a cubic resample keeps only 1.2% of a 4-px-period pattern. This is **numerical diffusion**.

The project instead keeps the history on a grid **fixed to the world**. It only ever shifts by whole pixels, which is an exact array copy, and the sub-pixel part of the camera offset is applied to where new samples land. Interpolation happens **once**, for display (Lagrange, m = 8), and never feeds back. This works because the motion is a global camera translation.

### Step 4: Clamp (anti-ghosting)
If the scene changes without moving (the light panel at frame 20), the motion vectors say "nothing changed", and the old history would show a **ghost**. So the history is clamped into the [min, max] range of the 3×3 current low-res samples around each pixel.

Plain min/max also cuts real detail thinner than the sample spacing: in fine regions, 12.6% of true values fell outside the box. The **edge-aware clamp** therefore widens the box by **α·√λ₁**, where λ₁ is the edge strength from the structure tensor (computed with the **power method**), and α = 2.63 after tuning. Flat areas keep a tight box (catching ghosts); detailed areas get room.

### Step 5: Display
The history is resampled once to the screen position with Lagrange interpolation (m = 8). It's blended with the single-frame Lagrange upscale where confidence is low (weight **w_prior = 0.05**), so the first frame and newly revealed edges aren't blocky.

### Step 6: Edge-preserving diffusion, a display-only clean-up, in `tsr/diffusion.py`
A partial differential equation, **∂u/∂t = div(D ∇u)**, smooths noise *along* edges but not *across* them. D is built from the structure tensor's eigenvectors: across an edge c₁ = 1/(1 + λ₁/K²), along it c₂ = 1, with K = 0.02.

Discretising space turns the PDE into a system of ODEs (the *method of lines*), which is stepped with **RK4** to time T = 0.5. The time step is chosen from the operator's largest eigenvalue, found with the **power method** (ρ ≈ 7.85), at 0.9 × the RK4 stability limit of 2.785/ρ. The smoothing is weighted by 1/(1 + (W/w₀)²) with w₀ = 1, so only low-confidence pixels are smoothed. It never feeds back into the history.

### Step 7: Tuning, in `scripts/05_tuning.py`
σ and α were chosen by solving equations against ground truth, our counterpart to training DLSS:
- **σ:** solve dPSNR/dσ = 0 (the maximum of mean PSNR). The derivatives come from central differences. **σ\* = 0.383.**
- **α:** solve ghost(α) = 0.05 (the largest α whose ghost error stays within a 5% budget). **α\* = 2.633.**

All four root-finders were used and compared by real cost, measured in full upscaler runs.

### Step 8 (optional): Motion estimation, in `tsr/motion.py`
The camera shift d = (dx, dy) is found by making the predicted frame match the observed one. That means solving the 2 nonlinear equations **Jᵀr(d) = 0** by **Gauss-Newton**, with `np.linalg.solve` on the 2×2 normal equations. It's cross-checked against `scipy.optimize.fsolve`.

It works to 0.004 px against a perfect reference, but it is biased by 0.2–0.9 px against a reference built from aliased frames. That bias is why DLSS requires motion vectors from the engine.

### Step 9: Game mode, in `tsr/photo.py` and `scripts/import_scene.py`
A screenshot is downscaled (Lanczos, via Pillow) to 1280×720 plus a 16 px margin and treated as the ground-truth world. A simulated renderer pans across it and takes Halton-jittered samples at 320×180, reading between pixels with Lagrange interpolation. The unchanged upscaler then runs once per R, G and B channel. The jitter is simulated, not the game's own.

### Project layout
```
tsr/quadrature.py   Simpson and trapezoidal rules, pixel integration
tsr/scene.py        the analytic test scene
tsr/frames.py       Halton jitter, frame rendering, ground truth, caching
tsr/interp.py       Lagrange, Newton forward/backward, Lanczos, matrix-form sampling
tsr/metrics.py      PSNR
tsr/temporal.py     the temporal upscaler (gather, accumulate, clamp, display)
tsr/edges.py        gradients, structure tensor, vectorised power method
tsr/diffusion.py    anisotropic diffusion, spectral radius, RK4 and Euler
tsr/rootfind.py     bisection, regula falsi, secant, Newton-Raphson
tsr/motion.py       Gauss-Newton registration, fsolve cross-check
tsr/photo.py        game screenshots as the world
scripts/01-06       one demo and figure per module
scripts/export_web.py, import_scene.py   data for the React demo
tests/              every hand-written method checked against scipy / numpy
web/                the React + Vite demo
decisions.md        every core design decision with its measurement
```

---

## 4. The numerical methods: where each is used and what it does

| Exp | Method | Where it is used | What it does here | Key measured result |
|---|---|---|---|---|
| 1 | Harmonic series | not used | the one syllabus topic with no natural fit | — |
| 2(a) | **Bisection** | tuning σ and α | halves a bracket that contains the root of dPSNR/dσ or ghost(α) − 0.05 | 22 / 15 pipeline runs, steady but slow |
| 2(b) | **Newton-Raphson** | tuning; Gauss-Newton is its multi-variable form | x ← x − f/f′; quadratic convergence | 5 iterations to 1e-7, but each derivative costs 2 extra runs |
| 2(c) | **Secant** | tuning | Newton with the slope through the last two points; order about 1.6, no derivative | cheapest overall: **5 runs** for α |
| 2(d) | **Regula falsi** | tuning | a bracket split where the chord crosses zero | linear convergence, one end stuck (classic weakness) |
| 2(e) | **Nonlinear systems**, `linalg.solve`, `linalg.norm`, `fsolve` | motion estimation | solves Jᵀr(d) = 0 for the 2-D camera shift | 0.004 px with a perfect reference; matches fsolve to 3e-6 px |
| 3 | **Lagrange interpolation** | upscaling, display resampling, reading the photo world | the polynomial through the m nearest pixels, as fixed weights per offset | cubic is exact for cubic surfaces; higher degree amplifies aliasing (Lebesgue constant 1.0 → 1.62) |
| 4 | **Newton forward/backward differences** | equivalent interpolation forms; finite-difference gradients and derivatives | builds the same polynomial from a difference table; central differences give gradients | equals Lagrange to 7e-16 (uniqueness of the interpolating polynomial) |
| 5 | **Trapezoidal rule** | pixel integration (comparison) | integrates the scene over a pixel's area | error O(h²) |
| 6 | **Simpson's 1/3 rule** | pixel integration (used) | the same, with weights 1, 4, 2, 4, …, 1 | error O(h⁴): 34× more accurate at n = 32 |
| 7 | **Runge-Kutta 4** | anisotropic diffusion clean-up | steps du/dt = Au | order 4 (vs Euler's 1); stable to dt = 0.355 vs Euler's 0.255 |
| 8 | **Power method** | edge strength and direction (structure tensor); stable time step (spectral radius) | v ← Jv/‖Jv‖ gives the dominant eigenpair | converges at (λ₂/λ₁)² per step; ρ(A) = 7.85 ≈ theory's 8 |

Further techniques that support these, all hand-written:
- **Gaussian weighting** in the gather step.
- **Huber IRLS** (iteratively reweighted least squares) as a robust option for motion estimation.
- **Interpolation as a matrix product**, M_y · img · M_xᵀ. It is linear algebra and gave a 4.3× speed-up.
- **The Lanczos-3** windowed-sinc baseline.

---

## 5. Tech stack

| Layer | Tools | Role |
|---|---|---|
| Language | Python 3.12 (virtual environment made with `uv`) | all numerics |
| Numerics | NumPy 2.5 | arrays and the vectorised methods; every core method is written on top of it by hand |
| Verification | SciPy 1.18 (`integrate`, `interpolate`, `optimize`), `numpy.linalg` | used only in tests, to check the hand-written methods |
| Images | Pillow 13 | loading screenshots, saving PNG/WebP frames |
| Plots | Matplotlib 3.11 | the six module figures |
| Tests | pytest | 17 tests |
| Old demo | Streamlit 1.66 | the first version of the demo (`app.py`) |
| Web demo | React 18.3, Vite 5.4, `@vitejs/plugin-react` | the interactive demo (`web/`) |
| Rendering in the browser | HTML canvas (the wipe and heatmap), inline SVG (chart and logo) | no chart or image library |
| Fonts | `@fontsource`: Hanken Grotesk, Newsreader, IBM Plex Mono | self-hosted, so the demo works offline |
| Design | the Hallmark design skill, Almanac theme | design rules: tokens, mono only for figures, one accent colour |
| Import bridge | a Vite dev-server plugin (Node) | lets the page upload files and run the Python importer |
| Version control | git, GitHub (`samashech/Likes-of-DLSS`) | |

---

## 6. Key results

| Scene (last frame) | Bilinear | Lagrange | Lanczos-3 | **Temporal** | Gain |
|---|---|---|---|---|---|
| Test scene, tuning path | 14.04 | 13.71 | 13.25 | **23.66** | +9.62 dB |
| Test scene, held-out path | 13.98 | 13.66 | 13.20 | **22.31** | +8.33 dB |
| Cyberpunk 2077 | 25.60 | 25.48 | 25.11 | **36.13** | +10.53 dB |
| Ghost of Tsushima | 26.72 | 26.43 | 25.99 | **35.49** | +8.77 dB |
| Phantom Blade Zero | 27.64 | 27.46 | 27.05 | **37.22** | +9.58 dB |
| Sifu | 29.11 | 29.04 | 28.69 | **39.74** | +10.63 dB |

### Limitations (say these before you're asked)
- **The motion is a global camera pan.** The world-aligned history depends on that. Real games have per-object motion, which forces screen-space reprojection and brings back blur.
- **The jitter in game mode is simulated** from screenshots; these are not the game engine's own frames.
- **Without engine motion vectors,** estimated motion is biased by aliasing, which costs about 2.7 dB.
- **σ and α were tuned on the test scene.** A held-out path and the game scenes confirm that they transfer, but a wider study would be stronger.

---

## 7. Glossary

**Aliasing.** When detail finer than the pixel grid is sampled, it shows up as false, coarser patterns: jaggies, shimmer, moiré. It happens because one sample per pixel can't represent frequencies above the Nyquist limit.

**Accumulation.** Adding each new frame's samples into a running high-res estimate.

**Anisotropic diffusion.** Smoothing that is direction-dependent: strong along edges, weak across them.

**Area average (pixel integration).** A pixel's value as the integral of the scene over the pixel's footprint, divided by the footprint's area.

**Bilinear interpolation.** Linear interpolation in x then y between the 2×2 nearest pixels (m = 2). The best single-frame method here, because it amplifies aliasing least.

**Bisection.** A root-finding method that halves an interval [a, b] with f(a)·f(b) < 0, keeping the half with the sign change. Guaranteed to converge; error halves each step.

**Camera pan.** A pure translation of the camera. Here (0.37, 0.21) px per frame, or (−0.29, 0.43) on the held-out path.

**Central difference.** f′(x) ≈ (f(x+h) − f(x−h)) / 2h, the derivative of the 3-point interpolating polynomial at its middle node.

**Clamp (history clamp).** Limiting history to the range of nearby current samples, to reject stale data and stop ghosting.

**Confidence (W).** Per high-res pixel, how many samples' worth of evidence the history holds. Capped at w_max.

**Convergence order.** How quickly the error shrinks: linear (bisection, regula falsi), about 1.618 (secant), quadratic (Newton).

**DLSS (Deep Learning Super Sampling).** NVIDIA's upscaler: temporal accumulation plus a neural network that decides per pixel how much history to trust.

**Edge-aware clamp.** This project's clamp. The min/max box is widened by α·√λ₁, so detailed regions keep detail and flat regions stay strict.

**Euler's method.** u ← u + dt·f(u). First order; stable here only for dt < 2/ρ = 0.255.

**FSR 2 / TAAU.** AMD's and Unreal's temporal upscalers. They work like DLSS without a network, which puts this project in the same family.

**fsolve.** SciPy's solver for systems of nonlinear equations. Used here only as a cross-check for our Gauss-Newton.

**Gather.** Collecting the current frame's nearby samples for each high-res pixel, weighted by a Gaussian of their distance.

**Gauss-Newton.** Newton's method for nonlinear least squares: linearise the residual r(d) and solve the normal equations (JᵀJ)Δ = −Jᵀr.

**Ghosting.** Stale history lingering after the scene has changed, such as the old checkerboard showing through the newly lit panel.

**Ground truth.** The exact high-res image, known here because the scene is a formula (or the screenshot itself). Used only to measure error, never by the upscaler.

**Halton sequence.** A low-discrepancy sequence (bases 2 and 3 here) whose points spread evenly with no clumping. It's used for jitter, as in DLSS, FSR 2 and TAA. The cycle here is 16 frames.

**Held-out path.** A camera path never used for tuning, used to check that the tuned parameters generalise. Like a test set.

**History (H).** The upscaler's running high-res estimate.

**Huber weighting / IRLS.** Robust least squares: residuals beyond a threshold c get weight c/|r|, recomputed every iteration.

**Jacobian (J).** The matrix of partial derivatives of the residuals with respect to the unknowns. Here the image gradient at each sample, with respect to dx and dy.

**Jitter.** The sub-pixel camera offset applied each frame, in low-res pixels, in [−0.5, 0.5).

**Lagrange interpolation.** The polynomial through m points written as Σ vₖ Lₖ(p), with Lₖ(p) = Π_{j≠k} (p − j)/(k − j).

**Lanczos-3.** A resampling filter using the windowed sinc L(x) = sinc(x)·sinc(x/3) for |x| < 3 (6 taps), normalised. The standard high-quality resize, and the lowest-scoring single-frame filter here on aliased input.

**Lebesgue constant.** The maximum of Σ|Lₖ(p)|: how much interpolation can amplify errors in the data. It rises with degree (1.0, 1.25, 1.39, 1.49, 1.62 for m = 2, 4, 6, 8, 12).

**Method of lines.** Discretising a PDE in space only, which turns it into a system of ODEs in time that an ODE solver (RK4) can step.

**Moiré.** False large-scale patterns from aliasing, very visible on the zone plate.

**Motion vectors.** The engine-supplied shift of each pixel since the last frame. Global here.

**Nearest neighbour.** Copying the nearest pixel (m = 1). Blocky.

**Newton-Raphson.** x ← x − f(x)/f′(x). Quadratic convergence near a simple root.

**Newton forward/backward differences.** Interpolation formulas built from a difference table, starting from the first node (forward) or the last (backward). They produce the same polynomial as Lagrange.

**Normal equations.** (JᵀJ)Δ = −Jᵀr, the linear system whose solution minimises ‖JΔ + r‖².

**Numerical diffusion.** Blur that accumulates when data is re-interpolated repeatedly. The same phenomenon seen in low-order advection schemes.

**Power method.** v ← Av/‖Av‖ converges to the dominant eigenvector. The eigenvalue comes from the Rayleigh quotient vᵀAv. The error falls by about |λ₂/λ₁| per step (|λ₂/λ₁|² for the eigenvalue of a symmetric matrix).

**PSNR (peak signal-to-noise ratio).** 10·log₁₀(1/MSE) for images in [0, 1], in decibels. Higher is better. +6 dB halves the RMS error, and +20 dB divides it by 10. An 8 px border is ignored.

**Rayleigh quotient.** vᵀAv for a unit vector v: the eigenvalue estimate in the power method.

**Regula falsi (false position).** Bisection that splits at the chord's zero instead of the midpoint. Can stall with one end stuck.

**Registration.** Aligning one image to another by finding the shift that best matches them.

**Reprojection.** Moving history to where the scene now appears under the new camera position.

**Residual (r).** Predicted minus observed values. Root-finding and least squares drive it towards zero.

**RK4 (classic Runge-Kutta, 4th order).** u ← u + dt/6·(k₁ + 2k₂ + 2k₃ + k₄). Fourth-order accurate. Stable for dt·ρ < about 2.785 on this kind of problem.

**RMS error.** The root-mean-square error, √MSE. "3× lower RMS error" corresponds to about +9.5 dB.

**Runge's phenomenon.** Wild oscillation near the ends when one high-degree polynomial is fitted through many equally spaced points. Avoided here by using small local stencils.

**Secant method.** Newton with f′ replaced by the slope through the last two iterates. Order about 1.618, no derivative needed.

**σ (sigma).** The Gaussian gather width in high-res px. Tuned to 0.383.

**Simpson's 1/3 rule.** Composite quadrature with weights (h/3)·[1, 4, 2, 4, …, 2, 4, 1]. Fourth-order, but needs an even number of intervals.

**Spectral radius (ρ).** The largest |eigenvalue| of an operator. It sets the stable explicit time step. Here 7.85.

**Stability region.** The set of values dt·λ for which an explicit ODE method doesn't blow up: up to 2 for Euler, about 2.785 for RK4 (on the negative real axis).

**Stencil.** The set of neighbouring points a formula uses, for example 4 points per axis for cubic.

**Structure tensor.** J = blur([Ix², IxIy; IxIy, Iy²]), a 2×2 matrix per pixel. Its eigenvectors give the edge direction and its eigenvalues the edge strength.

**Super-resolution.** Producing an image at a higher resolution than the input. *Temporal* super-resolution uses multiple frames over time.

**TAA (temporal anti-aliasing).** The same accumulation trick used to remove aliasing at the same resolution.

**Temporal.** Over time, across frames. "Temporal upscaling" means using many frames, not one.

**Trapezoidal rule.** Composite quadrature with weights h·[½, 1, …, 1, ½]. Second-order.

**Tuning.** Choosing parameters (σ, α) by solving equations against ground truth: our counterpart to training a network.

**Uniqueness theorem (interpolation).** Exactly one polynomial of degree ≤ m − 1 passes through m points with distinct nodes. So Lagrange and Newton agree.

**Upscaling factor (scale).** 4 here: 128×128 → 512×512 on the test scene, and 320×180 → 1280×720 on the game scenes.

**w_max.** The cap on history confidence (16). 1/w_max is the minimum blend weight for new data.

**w_prior.** The weight (0.05) of the single-frame upscale in the display, which matters where history is thin.

**World-aligned grid.** History stored in world coordinates, shifted only by whole pixels. No repeated interpolation, so no numerical diffusion.

**Zone plate.** cos(k·r²), rings whose frequency rises with radius. A standard test for aliasing.

---

## 8. Parameter cheat sheet

| Parameter | Value | Meaning |
|---|---|---|
| scale | 4 | upscale factor |
| Test scene | 128×128 → 512×512, 32 frames | |
| Game scenes | 320×180 → 1280×720, 24 frames, RGB | |
| Pan | (0.37, 0.21) px/frame; held-out (−0.29, 0.43) | camera motion |
| Halton | bases 2 and 3, 16-frame cycle | jitter |
| Pixel footprint | 1×1 high-res px | point-like, game-style sampling |
| Quadrature spacing | 0.5 px (n_quad = 2) | Simpson samples per pixel |
| Edge softness | 0.3 px (tanh) | keeps the scene smooth for Simpson |
| σ | 0.383 px (tuned) | gather width |
| w_max | 16 | history confidence cap |
| α | 2.63 (tuned) | edge-aware clamp width |
| w_prior | 0.05 | single-frame fallback weight |
| m_reproject | 8 | Lagrange stencil for the one display resample |
| diffuse_t | 0.5 | RK4 diffusion time |
| diffuse_K | 0.02 | diffusion edge threshold |
| diffuse_w0 | 1.0 | confidence where diffusion halves (quadratic falloff) |
| Light switch | frame 20 | ghosting test |
| PSNR border | 8 px | ignored edge band |
| Ghost budget | 0.05 | target for tuning α |

---

## 9. Quiz: 50 questions

Answer them before checking the key in section 10.

### A. The definition and what the project does (1–8)
1. In one sentence, what does the project do?
2. Why can't single-frame interpolation recover detail finer than a low-res pixel?
3. What is jitter, and what range does it take here?
4. What two pieces of information does DLSS get from the game engine, and which does this project also use?
5. What part of DLSS does this project replace with numerical methods instead of a neural network?
6. What are the input and output resolutions for the test scene and for the game scenes?
7. What happens at frame 20 of the test scene, and what is it testing?
8. In game mode, what plays the role of the ground truth, and what is simulated?

### B. How it is made (9–20)
9. Write the accumulation update for H and W.
10. Why is W capped at w_max = 16?
11. What is numerical diffusion, and how does the world-aligned history avoid it?
12. What assumption makes the world-aligned history valid?
13. Why did the first version of the frame generator (averaging the whole 4×4 low-res area) make temporal recovery impossible?
14. What does the gather step compute for each high-res pixel, and what does σ control?
15. Why does plain min/max clamping lose about 2 dB on static detail?
16. How does the edge-aware clamp fix that, and what is λ₁?
17. What is w_prior for?
18. Why is the RK4 diffusion "display-only"?
19. Why is the diffusion weighted by 1/(1 + (W/w₀)²)?
20. How is colour handled in game mode?

### C. Numerical methods (21–36)
21. Which syllabus experiment is not used, and why?
22. What are the convergence orders of the trapezoidal and Simpson's rules, and what was measured?
23. Why was Simpson *worse* than trapezoidal at n = 2 and 4?
24. Write the Lagrange basis polynomial Lₖ(p).
25. Why do Lagrange and Newton forward/backward give identical results?
26. Why does Lagrange win over Newton in the pipeline?
27. What is the Lebesgue constant, and what did it explain?
28. How does the project avoid Runge's phenomenon?
29. What does the power method compute for the structure tensor, and what sets its convergence speed?
30. What second job does the power method do in the diffusion module?
31. What are the stability limits for Euler and RK4 here, and what happened at dt = 0.3?
32. What error slopes were measured for Euler and RK4?
33. Which equation was solved to tune σ, and how were its derivatives obtained?
34. Which equation was solved to tune α?
35. Newton converges fastest per iteration, so why did secant win on the α problem?
36. What system does motion estimation solve, and with which method?

### D. Results and limits (37–42)
37. What are the last-frame PSNRs for bilinear and temporal on the test scene?
38. What was the Cyberpunk 2077 gain, and what does "3.36× lower RMS error" mean?
39. Why is Lanczos-3 the worst single-frame filter here, even though it's the "high-quality" resize?
40. What did the held-out path test, and what was the result?
41. Why is motion estimation biased without motion vectors, and what does that imply about DLSS?
42. Name two limitations of the project.

### E. Tech stack and glossary (43–50)
43. What is SciPy used for in this project, and what is it *not* used for?
44. Why is the web demo static, and how does the in-page import still run Python?
45. What do PSNR +6 dB and +20 dB mean in terms of RMS error?
46. Define temporal anti-aliasing (TAA) and how it relates to temporal upscaling.
47. What is a zone plate, and why is it in the test scene?
48. What is the Halton sequence, and why is it used instead of random jitter?
49. Define the method of lines.
50. What is the spectral radius, and what was its measured value?

---

## 10. Answer key

1. It combines many low-res frames, each rendered with a different sub-pixel jitter, into one sharp 4× higher-resolution image (DLSS-style temporal super-resolution), using only hand-written numerical methods.
2. A single frame never sampled that detail. Interpolation only fills between existing samples and can't create information. The best single-frame score was about 14 dB on the test scene.
3. A sub-pixel camera offset that changes every frame, so samples land in different places. It's in [−0.5, 0.5) low-res pixels per axis, from the Halton (2, 3) sequence.
4. The jitter and the motion vectors. The project uses both: the frame generator supplies them, like an engine.
5. The decision of how much history to trust per pixel. Here that's the confidence-weighted mean plus the edge-aware clamp, instead of a network.
6. Test scene: 128×128 → 512×512. Game scenes: 320×180 → 1280×720.
7. A panel lights up while nothing moves, so the motion vectors can't flag the change. It tests ghosting and the clamp.
8. The screenshot, downscaled to 1280×720 plus a margin, is the ground truth. The renderer is simulated: the camera pan, the Halton jitter and the 320×180 sampling.
9. H ← (W·H + Wn·C)/(W + Wn) and W ← W + Wn, with W capped at w_max first.
10. So old evidence fades and the image can still adapt to change. 1/w_max is the smallest weight a new frame gets.
11. Blur that compounds from re-interpolating history every frame (a cubic resample keeps 76% of a 4-px pattern per frame, 1.2% after 16 frames). The world-aligned grid only shifts history by whole pixels, which is exact, and interpolates once for display.
12. The motion is a global camera translation, with no per-object motion, rotation or perspective.
13. Averaging the whole 4×4 area is a box filter that erases detail before the upscaler sees it: a 4-px-period pattern keeps 0% of its contrast. Games take one sharp sample per pixel, so the fix was a 1×1 footprint.
14. A Gaussian-weighted estimate C of the pixel from the 3×3 nearest current samples, plus its confidence Wn. σ is the kernel width: wider averages more samples (better early), narrower keeps more detail (better late). Tuned to 0.383.
15. Features thinner than the 4-px sample spacing can lie outside the min/max of the 9 current samples, so the clamp cuts real detail. 12.6% of true values were outside the box in the fine regions.
16. It widens the box by α·√λ₁. λ₁ is the dominant eigenvalue of the structure tensor (the edge strength), computed by the power method. Detailed areas get room; flat areas stay strict.
17. To blend in the single-frame Lagrange upscale where history confidence is low (frame 0, newly revealed edges), so the image isn't blocky.
18. Its output is never written back into the history, so its smoothing can't compound over frames.
19. Low-confidence pixels are noisy and benefit from smoothing; well-supported pixels contain real detail that smoothing would blur. The quadratic falloff fades smoothing out quickly as W grows.
20. The unchanged upscaler runs once per R, G and B channel, with the same jitter and motion. PSNR is computed over all three channels.
21. Exp 1, the harmonic series. No part of the pipeline naturally needs it, and it wasn't forced in.
22. Trapezoidal O(h²) (error ×4 smaller per halving); Simpson O(h⁴) (×16). Simpson was 34× more accurate at n = 32.
23. Simpson's error bound assumes a smooth integrand. At coarse n a sharp edge falls between samples, so the bound doesn't apply.
24. Lₖ(p) = Π_{j≠k} (p − j)/(k − j), over nodes j = 0 … m−1.
25. The interpolating polynomial through m points is unique, so every form gives the same polynomial. Measured difference 7×10⁻¹⁶ (rounding).
26. Its weights depend only on the fractional offset, not on the data, so they can be precomputed and reused. 74 ms vs 102 ms.
27. max Σ|Lₖ(p)|, the factor by which interpolation can amplify errors in the data. It explained why higher degree, and Lanczos, score worse on aliased samples.
28. By using small local stencils (2 to 8 points) instead of one high-degree polynomial through many points.
29. The dominant eigenvalue λ₁ (edge strength) and eigenvector v₁ (across the edge) of every 2×2 tensor at once. The error falls by about (λ₂/λ₁)² per iteration: fast on clean edges, slow on isotropic texture.
30. It finds the spectral radius ρ of the diffusion operator A, which sets the largest stable RK4 time step.
31. Euler dt < 2/ρ = 0.255; RK4 dt < 2.785/ρ = 0.355. At dt = 0.3, Euler blew up (max |u| about 10⁴ in 60 steps) while RK4 stayed stable.
32. Slope 1 for Euler (first order) and slope 4 for RK4 (fourth order; error ×18 per halving of dt).
33. dPSNR/dσ = 0, the maximum of mean PSNR. The derivatives came from central differences with h = 0.01: P′ from P(σ±h), and P″ from P(σ±h) and P(σ). Root σ\* = 0.383.
34. ghost(α) − 0.05 = 0: the largest α whose ghost error one frame after the switch stays within a 5% budget. Root α\* = 2.633.
35. Each evaluation is a full upscaler run, and Newton's derivative needs 2 extra runs per iteration. Secant needs no derivative, so it used 5 runs vs Newton's 12. Choose by cost per evaluation, not order alone.
36. The 2 nonlinear equations Jᵀr(d) = 0 (the gradient of ½‖r‖² is zero) for the shift d = (dx, dy). It's solved by Gauss-Newton with `np.linalg.solve` on the 2×2 normal equations, and cross-checked with `fsolve`.
37. Bilinear 14.04 dB; temporal 23.66 dB (+9.62 dB).
38. +10.53 dB (25.60 → 36.13). The RMS error is 10^(10.53/20) ≈ 3.36 times smaller.
39. A sharper reconstruction amplifies the aliasing errors already in one frame's samples (a larger Lebesgue-type amplification). No single-frame filter can recover unsampled detail.
40. Whether σ and α overfit the tuning camera path. On the unseen path the tuned values still won: +1.87 dB on the last frame and +0.96 dB mean.
41. The reference image is built from aliased frames, whose errors depend on where they sampled, so the minimum of the mismatch moves: a 0.2–0.9 px bias. That is why DLSS takes exact motion vectors from the engine; here they're worth about 2.7 dB.
42. Any two of: global pan only (no per-object motion); simulated jitter in game mode; motion estimation is biased without vectors; parameters tuned on one scene; Exp 1 unused.
43. Only in tests, to verify the hand-written methods (simpson, trapezoid, lagrange, bisect, newton, root_scalar, fsolve). It's not used to do the actual work, apart from fsolve as a cross-check option.
44. The browser can't run the NumPy upscaler, so every frame is precomputed into images and a JSON manifest, which makes it fast and offline. A Vite dev-server plugin receives uploads and runs `scripts/import_scene.py`, so in-page import needs `npm run dev`.
45. +6 dB halves the RMS error; +20 dB divides it by 10.
46. Accumulating jittered frames over time to remove aliasing at the same resolution. Temporal upscaling uses the same accumulation but reconstructs at a higher resolution.
47. cos(k·r²), rings whose frequency increases with radius. It contains detail finer than the low-res grid, so it shows aliasing (moiré) clearly and makes recovery measurable.
48. A low-discrepancy sequence (bases 2 and 3) whose first N points always spread evenly. Random jitter clumps, leaving parts of each pixel unsampled; Halton covers the pixel in 16 frames.
49. Discretising a PDE in space only, which turns it into a system of ODEs in time (du/dt = Au) that an ODE solver like RK4 can step.
50. The largest absolute eigenvalue of an operator; it sets the stable explicit time step. Measured ρ(A) = 7.85, close to the theoretical bound of 8 for the 2-D stencil.
