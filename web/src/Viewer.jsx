import { useEffect, useRef, useState } from "react";

const cache = new Map();

/** Load (once) and cache one frame image. Resolves to an HTMLImageElement. */
export function loadImage(src) {
  if (!cache.has(src)) {
    cache.set(src, new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => { cache.delete(src); reject(new Error(src)); };
      img.src = src;
    }));
  }
  return cache.get(src);
}

export const ERR_MAX = 0.15; // error that maps to the end of the ramp, as a share of full brightness

let ramp = null;
/** Resolve the ramp's three token colours to RGB by painting them once. */
function getRamp() {
  if (ramp) return ramp;
  const c = document.createElement("canvas");
  c.width = 3;
  c.height = 1;
  const ctx = c.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  ["--color-sheet", "--color-accent", "--color-ink"].forEach((name, i) => {
    ctx.fillStyle = css.getPropertyValue(name).trim();
    ctx.fillRect(i, 0, 1, 1);
  });
  const d = ctx.getImageData(0, 0, 3, 1).data;
  ramp = [0, 1, 2].map((i) => [d[4 * i], d[4 * i + 1], d[4 * i + 2]]);
  return ramp;
}

const heatCache = new Map();
/**
 * |image - truth| over the region, averaged over R, G and B, drawn through the
 * ramp (sheet, magenta, ink) on a linear scale, so a good frame reads as pale
 * and only real misses turn dark.
 */
function heatmap(img, truth, region) {
  const key = `${img.src}|${truth.src}|${region.x},${region.y},${region.w},${region.h}`;
  if (heatCache.has(key)) return heatCache.get(key);
  const { x, y, w, h } = region;
  const read = (im) => {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(im, x, y, w, h, 0, 0, w, h);
    return [c, ctx, ctx.getImageData(0, 0, w, h)];
  };
  const [out, ctx, a] = read(img);
  const t = read(truth)[2].data;
  const [c0, c1, c2] = getRamp();
  const d = a.data;
  for (let i = 0; i < d.length; i += 4) {
    const e = (Math.abs(d[i] - t[i]) + Math.abs(d[i + 1] - t[i + 1]) + Math.abs(d[i + 2] - t[i + 2])) / (3 * 255);
    const v = Math.min(1, e / ERR_MAX);
    const [lo, hi, f] = v < 0.5 ? [c0, c1, v * 2] : [c1, c2, v * 2 - 1];
    d[i] = lo[0] + (hi[0] - lo[0]) * f;
    d[i + 1] = lo[1] + (hi[1] - lo[1]) * f;
    d[i + 2] = lo[2] + (hi[2] - lo[2]) * f;
    d[i + 3] = 255;
  }
  ctx.putImageData(a, 0, 0);
  if (heatCache.size > 200) heatCache.delete(heatCache.keys().next().value);
  heatCache.set(key, out);
  return out;
}

/**
 * The wipe: left image up to the divider, right image after it, both cropped
 * to `region` and scaled by a whole number so pixels stay square.
 */
export default function Viewer({ leftSrc, rightSrc, truthSrc, error = false, region, split, onSplit, leftLabel, rightLabel }) {
  const canvas = useRef(null);
  const box = useRef(null);
  const [pair, setPair] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let live = true;
    setStatus((s) => (pair ? s : "loading"));
    const want = error && truthSrc ? [leftSrc, rightSrc, truthSrc] : [leftSrc, rightSrc];
    Promise.all(want.map(loadImage))
      .then((imgs) => { if (live) { setPair(imgs); setStatus("ready"); } })
      .catch(() => { if (live) setStatus("error"); });
    return () => { live = false; };
  }, [leftSrc, rightSrc, truthSrc, error]); // eslint-disable-line react-hooks/exhaustive-deps

  const zoom = Math.max(1, Math.floor(960 / region.w));
  const W = region.w * zoom;
  const H = region.h * zoom;

  useEffect(() => {
    if (!pair || !canvas.current) return;
    const ctx = canvas.current.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    const [a, b, t] = pair;
    const cut = Math.round(split * W);
    // in error mode each side is its heat map, already cropped to the region
    const [sa, sb, sx, sy] = t ? [heatmap(a, t, region), heatmap(b, t, region), 0, 0] : [a, b, region.x, region.y];
    ctx.drawImage(sb, sx, sy, region.w, region.h, 0, 0, W, H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, cut, H);
    ctx.clip();
    ctx.drawImage(sa, sx, sy, region.w, region.h, 0, 0, W, H);
    ctx.restore();
  }, [pair, split, region, W, H]);

  const fromPointer = (e) => {
    const r = box.current.getBoundingClientRect();
    onSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };

  const onKey = (e) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft") { onSplit(Math.max(0, split - step)); e.preventDefault(); }
    if (e.key === "ArrowRight") { onSplit(Math.min(1, split + step)); e.preventDefault(); }
    if (e.key === "Home") { onSplit(0); e.preventDefault(); }
    if (e.key === "End") { onSplit(1); e.preventDefault(); }
  };

  return (
    <div
      className={`viewer${status === "loading" && !pair ? " skeleton" : ""}`}
      ref={box}
      style={{ aspectRatio: `${region.w} / ${region.h}` }}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); fromPointer(e); }}
      onPointerMove={(e) => { if (e.buttons) fromPointer(e); }}
    >
      <canvas ref={canvas} width={W} height={H} aria-label={`${leftLabel} on the left, ${rightLabel} on the right`} role="img" />
      {status === "ready" && (
        <>
          <div className="viewer__divider" style={{ left: `${split * 100}%` }} />
          <button
            type="button"
            className="viewer__handle"
            style={{ left: `${split * 100}%` }}
            role="slider"
            aria-label="Wipe position"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(split * 100)}
            aria-valuetext={`${Math.round(split * 100)}% ${leftLabel}`}
            onKeyDown={onKey}
          />
        </>
      )}
      {status === "loading" && !pair && <div className="viewer__state">Loading frames</div>}
      {status === "error" && (
        <div className="viewer__state">
          This frame did not load. Re-run the export: <span className="num">python scripts/export_web.py</span>
        </div>
      )}
    </div>
  );
}
