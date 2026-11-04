# Core Decisions

Only the decisions that shape the project, plus the reason for each, so you can defend them in a viva. New entries get added as modules are built.

---

## 1. What we are building: temporal super-resolution, not a neural network

DLSS has two halves:
1. **Temporal accumulation.** Gather samples from many jittered frames.
2. **A neural network** that decides how much of the old frames to trust.

We build half 1 completely. For half 2 we use classical numerical methods: interpolation, clamping, eigenvalues and ODEs.

**Why:** a network needs a training dataset and GPU time, which don't fit in 12 hours. More to the point, the extra detail in DLSS comes from half 1. The network only cleans up mistakes. This approach is the same family as AMD FSR 2 and Unreal TAAU, both of which work without a network.

## 2. The "game" is an analytic function, not a game engine or a photo

`tsr/scene.py` defines brightness `scene(x, y)` at **any real point**: a rotated checkerboard, a zone plate and a line chart.

**Why:**
- We can render from any sub-pixel camera position, which jitter needs.
- We get **exact ground truth** at high resolution, so every result has a measured error (PSNR) instead of "looks better".
- Each pattern is a known stress test. Angled edges show jaggies, the zone plate shows aliasing (moiré), and the thin bars show the resolution limit.

A photo would only give us pixels, so it can't be rendered at sub-pixel offsets without first interpolating it. That would hide the very errors we want to measure.

## 3. Every pixel integrates a 1×1 high-res-pixel footprint, using Simpson's 1/3 rule (Experiments 5, 6)

Each pixel's value is the average of the scene over a small square around its centre:

$$p = \iint_{\text{1×1 footprint}} \text{scene}(x,y)\,dx\,dy$$

We compute this with a hand-written 2D Simpson rule (`tsr/quadrature.py`), checked against `scipy.integrate.simpson` on the lab's arctan integral. The ground truth uses the same footprint, and so does each **low-res** sample: low-res pixels are just spaced 4 px apart and jittered.

**Revised in Module 3.** At first, each low-res pixel averaged its whole 4×4 area, like a blurry camera. That box filter **destroys** fine detail before the upscaler sees it: a pattern with a 4-px period keeps 0% of its contrast. A game rasteriser instead takes **one sharp sample per pixel**. That is why games alias, and it's why jitter can recover the detail.

**Why Simpson over trapezoidal:** the convergence plot in `out/01_frames.png` shows it.
- Trapezoidal error falls ×4 per halving of h, so it's O(h²).
- Simpson error falls ×16 per halving once the edges are resolved, so it's O(h⁴).
- At n=32, Simpson is 34× more accurate for the same cost.

**Honest finding:** at very coarse n (2 or 4), Simpson is *worse* than trapezoidal. Its error formula assumes a smooth integrand, and a sharp edge between samples breaks that assumption. That's why the scene edges are smoothed with `tanh` over 0.3 px.

## 4. Jitter comes from the Halton (2, 3) sequence, 16 frames per cycle

Each frame the camera shifts by a sub-pixel amount inside one low-res pixel. x uses base-2 Halton and y uses base-3.

**Why:** random jitter clumps, so some regions of the pixel would never get sampled. A regular grid lines up with the scene's own patterns. Halton is a low-discrepancy sequence, so any first *N* points spread evenly. After 16 frames each low-res pixel has been sampled at 16 well-spread spots, which is 16× the information of one frame. That is the whole source of the upscaling. DLSS, FSR 2 and Unreal TAA use the same sequence.

## 5. Motion is a known camera pan, like a game's motion vectors

The camera moves (0.37, 0.21) high-res px per frame. Those values are deliberately non-integer, so frames never line up on the pixel grid. The frame generator outputs this motion the way a game engine hands DLSS its motion vectors.

**Why:** DLSS doesn't guess motion; the game engine tells it. Giving the motion vectors exactly keeps the core algorithm honest and testable. Estimating motion from images alone was tested in Module 6 (decisions 20–21), and the results show why engines supply motion vectors.

## 6. Scale factor 4× (128×128 → 512×512)

**Why:** this matches DLSS "Ultra Performance" mode territory, and it's large enough that single-frame interpolation clearly fails. At 2×, the gap between methods would be too small to show.

## 7. Plain numpy, every core method written by hand

scipy is used only in tests, to verify our own implementations.

**Why:** the point of the lab is the numerical methods themselves. Calling `scipy.ndimage.zoom` would hide all of them.

---

## 8. Interpolation uses local stencils, and the single-frame baseline is bilinear-grade (Experiments 3, 4)

To read the image at a fractional position, we fit a polynomial through the m nearest pixels on each axis. We do x first, then y (`tsr/interp.py`).

**Why local:** one high-degree polynomial through a whole row of equally spaced pixels causes **Runge's phenomenon**, wild swings near the ends. Small stencils avoid that.

**Measured on the sharp, aliased frames:**

| m | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| PSNR (dB) | 12.50 | **14.05** | 13.75 | 13.57 | 13.46 | 13.32 |
| Lebesgue constant | 1 | 1.00 | 1.25 | 1.39 | 1.49 | 1.62 |

Higher degree makes things *worse*. The **Lebesgue constant**, max Σ|Lₖ(p)|, is how much interpolation can amplify errors in the data. Aliased samples are full of such errors, so a bigger stencil amplifies them more.

This is a mild relative of Runge, and it's why we don't use high degree for display. Cubic stays the default because it is exact for cubic surfaces and sharper on smooth regions. We report bilinear as the honest single-frame best.

## 9. Lagrange form in the pipeline, Newton forms kept as a verified equivalent

We implemented the Lagrange form plus the Newton forward- and backward-difference forms. Newton uses forward differences in the first half of the stencil and backward in the second, the textbook rule.

**Why all three agree:** through m points there is exactly **one** polynomial of degree m−1. Measured max difference: 7×10⁻¹⁶, which is just rounding.

**Why Lagrange wins in practice:** its weights depend only on the fractional position, not on the pixel values, so they can be reused. It was 74 ms vs 102 ms for Newton per 512×512 upscale.

## 10. The baseline to beat: about 14 dB, limited by aliasing

No single-frame method gets above **14.05 dB**: the samples simply don't contain the detail. In the zone plate, every method shows moiré (fake rings) instead of the real ones. We verified the coordinate convention by flipping the jitter sign, which made results worse, so `u = (q+0.5)/s − 0.5 − jitter` is correct.

---

## 11. Temporal core = confidence-weighted running mean (`tsr/temporal.py`)

Each high-res pixel keeps a value H and a confidence W (how many samples' worth of evidence it holds). Each frame:
1. **Gather** new samples near the pixel, with Gaussian weights on distance. This gives an estimate C with confidence Wn.
2. **Blend:** H ← (W·H + Wn·C)/(W + Wn), then W ← W + Wn.

This is the **weighted least-squares** estimate of the pixel from every sample it has ever seen. W is capped at `w_max` = 16, so old evidence slowly fades and the image can still react to change. Where W is still tiny (frame 0, newly revealed edges), the output leans on the single-frame Lagrange upscale.

**Result:** 13.7 dB → **21.4 dB** by frame 19, where the spatial baseline is 13.6. That's +7.8 dB, meaning the RMS error is about 2.5× smaller. Rings and thin bars that no single frame contains are visibly recovered (`out/03_temporal.png`).

## 12. History lives on a world-aligned grid, not re-sampled to the screen every frame

The textbook approach (TAA) resamples the history to the new camera position every frame. Each resample is a mild low-pass filter, and they compound. This is **numerical diffusion**, the same smearing that low-order schemes cause when solving advection equations.

Measured: a 0.37 px cubic resample keeps 76% of a 4-px-period pattern, so after 16 frames 1.2% is left. Even m=8 leaves only 4%.

**Our fix:** the history grid is fixed to the world. It only ever moves by whole pixels (an exact array copy), and the sub-pixel camera offset is applied to where new samples land. Interpolation happens **once**, for display, and never feeds back. Screen-space history (also m=8) reaches 20.3 dB; world-aligned reaches **21.4 dB**.

**Limitation (be upfront about it):** this works because our motion is a global camera translation. With per-object motion, rotation or perspective, a game has to use screen-space reprojection. That's why DLSS and FSR need sharpening, and why DLSS's network learns to undo this blur. `mode="screen"` is kept for that comparison.

The single display resample uses m=8 (+0.3 dB over m=4). It's cheap because it runs once per frame, and with no repeated resampling there's no build-up.

## 13. Anti-ghosting: min/max neighbourhood clamp, a measured trade-off

At frame 20 a panel lights up, but nothing moves, so the motion vectors say "nothing changed". Unclamped history keeps showing the old checkerboard, a **ghost**.

The clamp limits history to the [min, max] range of the 3×3 current low-res samples around each pixel. If history falls outside what the current frame can see, it must be stale.

| | before switch (frame 19) | 2 frames after switch | last frame |
|---|---|---|---|
| no clamp | **21.4** | 17.7 (ghost) | 19.5 |
| min/max clamp | 19.3 | **19.5** | **20.0** |

The clamp costs about 2 dB on static detail, because it sometimes cuts off real detail that the 3×3 low-res neighbourhood can't see. In exchange it removes ghosts almost immediately. Variance clipping (mean ± γσ) was tested and did slightly worse here (17.3 dB, or 17.8 with γ=2).

**This trade-off is exactly what DLSS's neural network exists to solve:** deciding per pixel when history can be trusted. It's our best target for improvement.

## 14. Gather width: the best σ depends on history length (tuned in decision 18)

| σ (px) | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| frame 4 | 17.66 | **17.96** | 17.77 | 17.33 |
| frame 19 | **22.42** | 21.39 | 19.79 | 18.38 |

Early on, few samples exist, so a wider kernel averages more of them. Later there are many, so a narrower kernel keeps more detail. Choosing σ is therefore a real optimisation problem. It's left for the root-finding module (Experiments 2a–d) instead of being hand-picked.

---

## 15. Edges come from the structure tensor, with eigenpairs from a vectorised power method (Experiment 8)

At each pixel, J = blur([Ix², IxIy; IxIy, Iy²]) is a 2×2 symmetric matrix. Gradients use central differences, the derivative of the 3-point Newton interpolant.
- **Dominant eigenvector v₁:** points across the edge.
- **v₂:** points along the edge.
- **λ₁:** the edge strength.

We run the power method on all 262,144 matrices at once (`tsr/edges.py`), and it matches `numpy.linalg.eig` to 10⁻⁶.

**Measured convergence** (`out/04_edges.png`): the error shrinks by about (λ₂/λ₁)² per iteration.
- Clean edges (ratio < 0.1) converge in about 4 iterations.
- Mixed regions (0.4–0.6) take about 23.
- Isotropic texture (0.8–0.95) is still at 10⁻⁶ after 40.

This is the textbook result, seen on real data.

## 16. Edge-aware clamp: the box is widened by α·√λ₁ (fixes most of decision 13's 2 dB cost)

**Diagnosis first:** with the min/max clamp, **12.6%** of true pixel values in the zone plate and the bars fell *outside* the clamp box, against 0% on the flat checkerboard. Detail thinner than the 4-px sample spacing slips between the 3×3 current samples, so the clamp was cutting real detail.

**Fix:** widen the box by α·√λ₁, the local edge strength of the current frame. Structured regions get a looser box, so detail survives. Flat regions keep a tight box, which is where a ghost would be most visible.

| | f19 (static detail) | f21 (just after switch) | last |
|---|---|---|---|
| min/max | 19.34 | 19.10 | 20.03 |
| **edge-aware, α=3** | **20.96** | **20.16** | **21.65** |
| no clamp | 21.39 | 17.20 (ghost) | 19.52 |

Raising α trades ghosting against detail. Mean ghost error in the panel rises steadily (0.046 at α=2, 0.087 at α=10), while f19 improves. α=3 gives the best last-frame score for now; Module 5 will choose it by root-finding.

**Limitation:** a ghost over a *textured* area gets a loose box and fades more slowly. Visually, texture masks it. DLSS's network makes this call per pixel and better.

## 17. RK4 anisotropic diffusion as a display-only cleanup (Experiment 7)

The PDE is ∂u/∂t = div(D∇u), with D = c₁v₁v₁ᵀ + c₂v₂v₂ᵀ:
- **Across edges:** c₁ = 1/(1+λ₁/K²), so there's no blurring across.
- **Along edges:** c₂ = 1, full smoothing.

Discretising space turns it into a linear ODE system du/dt = Au (the method of lines), which RK4 steps forward in time.
- **Compact differences for the diagonal terms:** central differences can't see a 1-pixel checker pattern, which is exactly the noise we want to remove.
- **Time step from the power method again (Exp 8):** applied to A it gives ρ(A) = 7.85, close to the theoretical 8 for this stencil. RK4 is stable when dt·ρ < 2.785 (dt < 0.355); Euler only when dt·ρ < 2 (dt < 0.255).

**Verified in `out/04_edges.png`:** Euler at dt=0.3 grows to 10⁴ in 60 steps, while RK4 at the same dt stays bounded. RK4 at dt=0.37 also blows up, as predicted. Error vs dt has slope 1 for Euler and 4 for RK4 (×18 per halving). At dt=0.25, RK4 is about 100× more accurate than Euler.

**Weighted by confidence:** D is multiplied by 1/(1+(W/w₀)²). Uniform smoothing helped early frames (+0.6 dB at frame 2) but hurt later ones, because by then the "dots" are mostly real detail. The 1/(1+W/w₀) falloff still cost 0.1 dB late on, so we use the quadratic, which fades out faster.

The result, T=1, K=0.02, w₀=1: **+0.68 dB at frame 2**, +0.13 at frame 19, −0.03 at the last frame. It's display-only and never fed back into the history, so it can't compound.

**Honest scale of the gain:** diffusion mainly helps right after a cut or reveal, when history is thin. Once history is full, the temporal accumulation has already done the work.

---

## 18. Parameters are tuned by root-finding against ground truth (Experiments 2a–d)

This is our counterpart to DLSS training. Nvidia fits a network to ground-truth renders; we fit two numbers to ground truth by solving equations (`scripts/05_tuning.py`).

**Problem 1, σ (gather width).** Maximise the mean PSNR over 32 frames, which means solving P′(σ) = 0. P′ and P″ come from central differences with h = 0.01: the first and second derivative of the quadratic through 3 points, the Newton-difference idea from Exp 4. Bracket [0.3, 0.6]. **σ\* = 0.383.**

**Problem 2, α (clamp width).** Find the largest α whose ghost error one frame after the switch stays under a **5% budget**: solve ghost(α) − 0.05 = 0. ghost(α) rises smoothly, so it has exactly one root. Bracket [0, 8]. **α\* = 2.633.**

The real cost is pipeline runs, not iterations, because every evaluation runs the whole upscaler. Tolerance 10⁻³:

| method | σ: iterations / runs | α: iterations / runs |
|---|---|---|
| Bisection | 9 / 22 | 13 / 15 |
| Regula falsi | 6 / 16 | 8 / 10 |
| Secant | 6 / 12 | 5 / **5** |
| Newton-Raphson | 5 / 12 | 5 / 12 |

**What the convergence plots show** (`out/05_tuning.png`):
- **Bisection** gains one bit per step, a straight line on the log plot, and it doesn't settle steadily, because the midpoint jumps around the root.
- **Regula falsi** converges linearly. One end of the bracket stays stuck, the classic weakness.
- **Secant** has order about 1.6.
- **Newton** is quadratic, reaching 10⁻⁷ in 5 iterations.

**The lesson for the viva:** Newton converges fastest *per iteration*, but each iteration needs f′. In problem 2 that costs 2 extra runs, so secant wins on real cost (5 runs vs 12). In problem 1, P′ already needs P(σ±h), and P″ needs only one more value, P(σ), so Newton and secant tie. Choose the method by cost per evaluation, not by order of convergence.

**Why tuning uses ground truth:** this is offline calibration, just as DLSS is trained offline on reference renders. At run time the upscaler only sees low-res frames.

## 19. Result of tuning

| | mean PSNR | frame 21 (after switch) | last frame |
|---|---|---|---|
| single-frame bilinear | 14.00 | – | 14.05 |
| hand-picked (σ 0.6, α 3, T 1) | 19.90 | 20.17 | 21.62 |
| tuned (σ 0.383, α 2.63, T 1) | 20.75 | 21.30 | 23.53 |
| **tuned + diffusion T 0.5 (defaults)** | **20.85** | **21.37** | **23.66** |

**+9.6 dB over single-frame on the last frame, about 3× lower RMS error.**

With the narrower tuned σ, diffusion helps more (mean +0.37 dB at T=0.5, vs +0.1 before): a narrow kernel keeps detail but also noise, and the diffusion cleans up exactly that noise. T was picked by a coarse 3-point scan (0, 0.5, 1), not by root-finding.

**Held-out check (train/test split):** σ and α were tuned on the pan (0.37, 0.21). On an **unseen** pan (−0.29, 0.43), the tuned values still win:

| held-out pan | mean | frame 21 | last |
|---|---|---|---|
| hand-picked | 18.98 | 19.63 | 20.44 |
| **tuned** | **19.94** | **20.91** | **22.31** |
| bilinear | 14.00 | – | – |

That's +0.96 dB mean and +1.87 dB on the last frame, so the tuning generalises rather than overfitting to one camera path. It's the same scene, though; a different scene would be the stronger test.

---

## 20. Motion estimation is a 2×2 nonlinear system, solved by Gauss-Newton (Experiment 2e)

Unknown: the shift d = (dx, dy). For a guessed d, read the reference image at the new samples' positions to predict the low-res frame. The mismatch r(d) is smallest at the true shift, so we solve **F(d) = Jᵀr = 0**: 2 nonlinear equations, where J is the image gradient at the samples (`tsr/motion.py`).

**Gauss-Newton:** linearise r, solve the 2×2 normal equations (JᵀJ)Δ = −Jᵀr with `np.linalg.solve`, and stop when `np.linalg.norm(Δ)` < 10⁻⁴. A tiny damping term keeps the matrix invertible.

**Verified against `scipy.optimize.fsolve`:** they agree to 3×10⁻⁶ px. With a perfect reference it converges in **4–5 iterations** to 0.004–0.013 px, the quadratic convergence of Newton's method.

**What failed along the way, and why (each was measured):**
- **A confidence mask.** It selected 0 samples. Halton jitter puts each new sample in the gaps between old ones, exactly where confidence is lowest. The jitter works as designed, which defeats this mask.
- **Raw history as the reference.** Early history is blocky, one flat patch per sample, so the gradient is zero inside patches, J vanishes and the system is singular.
- **A blurred history as the reference.** Comparing a blurred prediction against sharp samples moves the minimum, biasing the result by up to 0.23 px even with a perfect history.
- **Huber reweighting (IRLS).** No improvement, because outliers weren't the main problem.

## 21. Finding: registration accuracy is set by the reference, so engines must provide motion vectors

| reference image | checkerboard | zone plate (aliased) |
|---|---|---|
| ground truth | 0.013 px | 0.004 px |
| 1-frame Lagrange upscale | 0.18 px | **0.88 px** |

The solver is fine. The reference is the bottleneck. Interpolation and aliasing errors in an upscaled aliased frame are tied to the previous frame's sample positions, so they bias the solution. Without good motion the history stays poor, so the reference stays poor: a vicious circle.

| motion source | final camera drift | mean PSNR | last frame |
|---|---|---|---|
| exact vectors (default) | 0 | **20.85** | **23.66** |
| vectors with 0.15 px noise | 1.05 px | 17.68 | 18.75 |
| noisy vectors + registration refinement | 0.45 px | 18.45 | 20.63 |
| **estimated only, no vectors** | 0.41 px | 18.62 | 20.99 |
| single-frame bilinear | – | 14.00 | 14.05 |

**Three takeaways:**
1. **Exact motion vectors are worth about 2.7 dB.** That's why DLSS, FSR 2 and XeSS all require them from the game engine. Our experiment measures why.
2. **Estimation alone still works.** It's +7 dB over single-frame with no motion vectors at all, though each frame's estimate is off by 0.2–0.9 px.
3. **Registration stops drift.** Noisy vectors add up their errors like a random walk (1.05 px drift by frame 31). Registering against the history anchors each frame to the world map, so the drift stays bounded at about 0.4 px. Per-step error is the wrong metric; the absolute camera position error is the one that matters.

The default stays `motion="given"`, like a real engine. `"estimated"` and `"refined"` remain as options.
