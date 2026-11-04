"""Make a slimmed copy of the built demo for hosting (web/deploy/).

Vercel's Hobby plan caps a CLI upload at 100 MB and the full demo data is
about 200 MB, so the hosted copy:
  * re-encodes the test scene's frames from PNG to WebP (quality 95),
  * re-encodes the game scenes' frames to WebP quality 85,
  * drops the test scene's held-out path frames (its scores stay, so the
    benchmarks page still shows them; the demo hides that path),
  * drops raw-frame scenes (no ground truth; a local feature).
PSNR scores were computed before any compression, so they stay exact.
The local web/public/data is never modified.

Run after `npm run build` in web/:
    .venv/bin/python scripts/prepare_deploy.py
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "web" / "dist"
OUT = ROOT / "web" / "deploy"
TEST_QUALITY, GAME_QUALITY = 95, 85


def reencode(folder, quality, src_ext, keep_ext=True):
    for p in sorted(folder.rglob(f"*.{src_ext}")):
        dst = p.with_suffix(".webp")
        Image.open(p).save(dst, "WEBP", quality=quality, method=6)
        if p != dst:
            p.unlink()


def size(folder):
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def main():
    if not (DIST / "index.html").exists():
        sys.exit("web/dist is missing: run `npm run build` in web/ first")
    # fresh copy, but keep the Vercel link (.vercel) between runs
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name != ".vercel":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(DIST, OUT, dirs_exist_ok=True)
    data = OUT / "data"

    # test scene: PNG -> WebP, held-out path keeps scores but not frames
    manifest = json.loads((data / "manifest.json").read_text())
    reencode(data / "tuning", TEST_QUALITY, "png")
    shutil.rmtree(data / "heldout", ignore_errors=True)
    manifest["ext"] = "webp"
    manifest["paths"]["heldout"]["noFrames"] = True
    (data / "manifest.json").write_text(json.dumps(manifest))

    # game scenes: lower WebP quality; raw scenes removed
    index = json.loads((data / "scenes.json").read_text())
    kept = []
    for s in index:
        folder = data / "scenes" / s["id"]
        if s["kind"] == "raw":
            shutil.rmtree(folder, ignore_errors=True)
            continue
        reencode(folder / "game", GAME_QUALITY, "webp")
        kept.append(s)
    (data / "scenes.json").write_text(json.dumps(kept))

    mb = size(OUT) / 1e6
    files = sum(1 for f in OUT.rglob("*") if f.is_file() and ".vercel" not in f.parts)
    print(f"web/deploy: {mb:.1f} MB in {files} files ({len(kept)} game scenes)")
    if mb > 100:
        sys.exit("over the 100 MB Hobby upload limit")


if __name__ == "__main__":
    main()
