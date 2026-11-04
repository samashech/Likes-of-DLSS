"""Export every method's output frames for the React demo (web/).

Writes web/public/data/<path>/<method>/<frame>.png (8-bit grey, 512x512),
the low-res input frames, and web/public/data/manifest.json with PSNR
curves, jitter and the scene facts the UI shows.

Run: .venv/bin/python scripts/export_web.py      (about 3 minutes)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsr.frames import make_sequence  # noqa: E402
from tsr.interp import upscale  # noqa: E402
from tsr.metrics import psnr  # noqa: E402
from tsr.scene import SWITCH_FRAME  # noqa: E402
from tsr.temporal import Params, TemporalUpscaler, run  # noqa: E402

OUT = ROOT / "web" / "public" / "data"
SCALE = 4
PATHS = {"tuning": {"label": "Tuning path", "pan": (0.37, 0.21)},
         "heldout": {"label": "Held-out path", "pan": (-0.29, 0.43)}}


def temporal(**kw):
    motion = kw.pop("motion", "given")
    return lambda frames: run(frames, upscaler=TemporalUpscaler(SCALE, Params(**kw), motion=motion))


# id, label, group, how it is made
METHODS = [
    ("truth", "Ground truth", "reference", lambda fr: [f.truth for f in fr]),
    ("nearest", "Nearest neighbour", "single frame", lambda fr: [upscale(f.low, SCALE, f.jitter, 1) for f in fr]),
    ("bilinear", "Bilinear", "single frame", lambda fr: [upscale(f.low, SCALE, f.jitter, 2) for f in fr]),
    ("lagrange", "Lagrange cubic", "single frame", lambda fr: [upscale(f.low, SCALE, f.jitter, 4) for f in fr]),
    ("noclamp", "Temporal, no clamp", "temporal", temporal(clamp="none", diffuse_t=0)),
    ("minmax", "Temporal, min/max clamp", "temporal", temporal(clamp="minmax", diffuse_t=0)),
    ("edge", "Temporal, edge-aware clamp", "temporal", temporal(diffuse_t=0)),
    ("full", "Temporal, edge clamp + RK4", "temporal", temporal()),
    ("estimated", "Temporal, motion estimated", "temporal", temporal(motion="estimated")),
]


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)


manifest = {"scale": SCALE, "size": 512, "lowSize": 128, "switchFrame": SWITCH_FRAME,
            "methods": [{"id": m, "label": lab, "group": g} for m, lab, g, _ in METHODS],
            "regions": [
                {"id": "zone", "label": "Zone plate", "test": "aliasing", "x": 126, "y": 196, "w": 120, "h": 120},
                {"id": "bars", "label": "Thin bars", "test": "resolution", "x": 50, "y": 405, "w": 160, "h": 100},
                {"id": "panel", "label": "Light panel", "test": "ghosting", "x": 315, "y": 395, "w": 170, "h": 110},
                {"id": "edges", "label": "Checker edges", "test": "jaggies", "x": 10, "y": 10, "w": 160, "h": 120},
                {"id": "full", "label": "Full frame", "test": "everything", "x": 0, "y": 0, "w": 512, "h": 512}],
            "paths": {}}

for pid, info in PATHS.items():
    frames = make_sequence(pan=info["pan"])
    entry = {"label": info["label"], "pan": list(info["pan"]), "frames": len(frames),
             "jitter": [[round(float(v), 4) for v in f.jitter] for f in frames], "psnr": {}}
    for k, f in enumerate(frames):
        save(f.low, OUT / pid / "low" / f"{k}.png")
    for mid, label, _, make in METHODS:
        out = make(frames)
        if mid != "truth":
            entry["psnr"][mid] = [round(float(psnr(o, f.truth)), 3) for o, f in zip(out, frames)]
        for k, o in enumerate(out):
            save(o, OUT / pid / mid / f"{k}.png")
        print(f"{pid:8s} {label:28s} last frame "
              f"{entry['psnr'].get(mid, [float('nan')])[-1]:.2f} dB", flush=True)
    manifest["paths"][pid] = entry

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote", OUT)
