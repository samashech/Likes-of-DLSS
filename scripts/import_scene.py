"""Add a scene to the React demo.

Game image mode (ground truth known, PSNR shown):
    .venv/bin/python scripts/import_scene.py image uploads/cyberpunk.jpg --name "Cyberpunk 2077"

    The image becomes the world; a simulated renderer pans across it and
    renders 320x180 jittered frames, which are upscaled 4x to 1280x720.

Raw frames mode (your own jittered renders, no ground truth):
    .venv/bin/python scripts/import_scene.py raw path/to/folder --name "My renders"

    The folder holds the low-res frames in name order (PNG or JPEG, all the
    same size) and frames.json:
        {"scale": 4,
         "jitter": [[jx, jy], ...],     sub-pixel offset per frame, in low-res pixels, in [-0.5, 0.5)
         "motion": [[dx, dy], ...]}     camera shift since the previous frame, in high-res pixels
    "motion" is required: a renderer knows its camera. --estimate-motion
    estimates it by registration instead, which is unreliable on detailed
    images (decision 21); the result says so.

Writes web/public/data/scenes/<id>/ and lists the scene in web/public/data/scenes.json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsr.frames import Frame  # noqa: E402
from tsr.interp import upscale  # noqa: E402
from tsr.photo import load_world, make_photo_sequence, psnr_rgb, save_rgb, stack  # noqa: E402
from tsr.temporal import Params, TemporalUpscaler, run  # noqa: E402

DATA = ROOT / "web" / "public" / "data"
VIEW = (1280, 720)
TILE = (160, 90)


def spatial(m):
    return lambda ch: [upscale(f.low, f_scale(ch), f.jitter, m) for f in ch]


def f_scale(ch):
    return ch[0].truth.shape[1] // ch[0].low.shape[1]


METHODS = [
    ("nearest", "Low-res input (nearest)", "single frame", spatial(1)),
    ("bilinear", "Bilinear", "single frame", spatial(2)),
    ("lagrange", "Lagrange cubic", "single frame", spatial(4)),
    ("lanczos", "Lanczos-3", "single frame", lambda ch: [upscale(f.low, f_scale(ch), f.jitter, method="lanczos") for f in ch]),
    ("edge", "Temporal, edge-aware clamp", "temporal", lambda ch: run(ch, f_scale(ch), Params(diffuse_t=0))),
    ("full", "Temporal, edge clamp + RK4", "temporal", lambda ch: run(ch, f_scale(ch))),
]


def best_tiles(err_a, err_b, n=3):
    """Top-n non-overlapping tiles where method b removes the most squared error."""
    H, W = err_a.shape
    tw, th = TILE
    gain = err_a - err_b
    cands = []
    for y in range(8, H - th - 8, th // 3):
        for x in range(8, W - tw - 8, tw // 3):
            cands.append((gain[y:y + th, x:x + tw].mean(), x, y))
    cands.sort(reverse=True)
    picked = []
    for g, x, y in cands:
        if all(abs(x - px) >= tw or abs(y - py) >= th for _, px, py in picked):
            picked.append((g, x, y))
        if len(picked) == n:
            break
    return picked


def load_raw(folder, estimate=False):
    folder = Path(folder)
    meta = json.loads((folder / "frames.json").read_text())
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if len(files) != len(meta["jitter"]):
        sys.exit(f"{len(files)} images but {len(meta['jitter'])} jitter entries in frames.json")
    scale = int(meta.get("scale", 4))
    imgs = [np.asarray(Image.open(p).convert("RGB"), float) / 255.0 for p in files]
    lh, lw = imgs[0].shape[:2]
    if any(im.shape[:2] != (lh, lw) for im in imgs):
        sys.exit("all frames must be the same size")
    jit = np.asarray(meta["jitter"], float)
    motion = meta.get("motion")
    blank = np.zeros((lh * scale, lw * scale))
    if motion is None and not estimate:
        sys.exit('frames.json has no "motion". Add the camera shift per frame, or pass '
                 '--estimate-motion (unreliable on detailed images, see decisions.md 21).')
    if motion is None:
        print("warning: estimating motion by registration; expect errors of a pixel or more")
        # estimate once on luminance, then reuse the shifts for every channel
        lum = [Frame(im @ [0.299, 0.587, 0.114], jit[k], np.zeros(2), np.zeros(2), blank) for k, im in enumerate(imgs)]
        tu = TemporalUpscaler(scale, motion="estimated")
        run(lum, scale, upscaler=tu)
        motion = [m.tolist() for m in tu.motions]
        print("estimated motion:", np.round(motion, 3).tolist())
    motion = np.asarray(motion, float)
    cam = np.cumsum(motion, axis=0)
    channels = [[Frame(im[..., c], jit[k], cam[k], motion[k], blank) for k, im in enumerate(imgs)]
                for c in range(3)]
    return channels, scale, motion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["image", "raw"])
    ap.add_argument("source")
    ap.add_argument("--name", required=True)
    ap.add_argument("--id", help="folder name (default: from --name)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--estimate-motion", action="store_true", help="raw mode: estimate motion if frames.json has none")
    args = ap.parse_args()
    sid = args.id or re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
    out = DATA / "scenes" / sid
    t0 = time.perf_counter()

    if args.mode == "image":
        world = load_world(args.source, VIEW)
        pan = (0.37, 0.21)
        channels = make_photo_sequence(world, args.frames, VIEW, 4, pan)
        scale, has_truth = 4, True
        motion_note = f"camera pan ({pan[0]}, {pan[1]}) px per frame, motion vectors given"
    else:
        channels, scale, motion = load_raw(args.source, args.estimate_motion)
        pan = motion[1:].mean(0).round(3).tolist() if len(motion) > 1 else [0, 0]
        has_truth = False
        motion_note = "motion from frames.json" if "motion" in json.loads(
            (Path(args.source) / "frames.json").read_text()) else "motion estimated by registration, unreliable"
    n = len(channels[0])
    H, W = channels[0][0].truth.shape
    print(f"progress 0/{len(METHODS)}", flush=True)
    lh, lw = channels[0][0].low.shape
    print(f"{args.name}: {n} frames, {lw}x{lh} -> {W}x{H}")

    truth = [stack([[f.truth for f in ch] for ch in channels], k) for k in range(n)] if has_truth else None
    psnr, outputs = {}, {}
    for mid, label, _, make in METHODS:
        t = time.perf_counter()
        outs = [make(ch) for ch in channels]
        imgs = [stack(outs, k) for k in range(n)]
        outputs[mid] = imgs[-1]
        for k, im in enumerate(imgs):
            save_rgb(im, out / "game" / mid / f"{k}.webp")
        if has_truth:
            psnr[mid] = [round(float(psnr_rgb(im, truth[k])), 3) for k, im in enumerate(imgs)]
        note = f"last frame {psnr[mid][-1]:.2f} dB" if has_truth else ""
        print(f"  {label:28s} {time.perf_counter() - t:6.1f}s  {note}", flush=True)
        print(f"progress {[m for m, *_ in METHODS].index(mid) + 1}/{len(METHODS)}", flush=True)
    for k in range(n):
        save_rgb(stack([[f.low for f in ch] for ch in channels], k), out / "game" / "low" / f"{k}.webp")

    methods = [{"id": m, "label": lab, "group": g} for m, lab, g, _ in METHODS]
    if has_truth:
        for k, im in enumerate(truth):
            save_rgb(im, out / "game" / "truth" / f"{k}.webp")
        methods = [{"id": "truth", "label": "Ground truth (the screenshot)", "group": "reference"}] + methods
        ref = truth[-1]
    else:
        ref = outputs["full"]   # no truth: pick regions by where temporal and Lagrange differ most
    err_a = ((outputs["lagrange"] - ref) ** 2).mean(-1) if has_truth else ((outputs["lagrange"] - outputs["full"]) ** 2).mean(-1)
    err_b = ((outputs["full"] - ref) ** 2).mean(-1) if has_truth else np.zeros_like(err_a)
    tiles = best_tiles(err_a, err_b)
    regions = [{"id": f"auto{i + 1}", "label": ["Biggest gain", "Second biggest", "Third biggest"][i],
                "test": "picked automatically", "x": int(x), "y": int(y), "w": TILE[0], "h": TILE[1]}
               for i, (_, x, y) in enumerate(tiles)]
    regions.append({"id": "full", "label": "Full frame", "test": "everything", "x": 0, "y": 0, "w": W, "h": H})

    overview = stack([[f.truth for f in ch] for ch in channels], 0) if has_truth else outputs["full"]
    save_rgb(np.asarray(Image.fromarray((overview * 255).astype(np.uint8)).resize((640, 640 * H // W)), float) / 255,
             out / "overview.webp", quality=85)

    manifest = {
        "kind": "game" if has_truth else "raw", "label": args.name, "version": int(time.time()), "base": f"scenes/{sid}", "ext": "webp",
        "width": W, "height": H, "lowWidth": lw, "lowHeight": lh, "scale": scale, "hasTruth": has_truth,
        "switchFrame": None, "source": Path(args.source).name, "motionNote": motion_note,
        "methods": methods, "regions": regions,
        "paths": {"game": {"label": "Camera pan", "pan": list(pan), "frames": n,
                           "jitter": [[round(float(v), 4) for v in f.jitter] for f in channels[0]],
                           "psnr": psnr if has_truth else None}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    index_path = DATA / "scenes.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [s for s in index if s["id"] != sid] + [{"id": sid, "label": args.name, "kind": manifest["kind"]}]
    index_path.write_text(json.dumps(index, indent=1))
    print(f"wrote {out} ({time.perf_counter() - t0:.0f}s); listed in {index_path}")
    print(f"scene {sid}", flush=True)


if __name__ == "__main__":
    main()
