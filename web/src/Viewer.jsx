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

/**
 * The wipe: left image up to the divider, right image after it, both cropped
 * to `region` and scaled by a whole number so pixels stay square.
 */
export default function Viewer({ leftSrc, rightSrc, region, split, onSplit, leftLabel, rightLabel }) {
  const canvas = useRef(null);
  const box = useRef(null);
  const [pair, setPair] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let live = true;
    setStatus((s) => (pair ? s : "loading"));
    Promise.all([loadImage(leftSrc), loadImage(rightSrc)])
      .then((imgs) => { if (live) { setPair(imgs); setStatus("ready"); } })
      .catch(() => { if (live) setStatus("error"); });
    return () => { live = false; };
  }, [leftSrc, rightSrc]); // eslint-disable-line react-hooks/exhaustive-deps

  const zoom = Math.max(1, Math.floor(960 / region.w));
  const W = region.w * zoom;
  const H = region.h * zoom;

  useEffect(() => {
    if (!pair || !canvas.current) return;
    const ctx = canvas.current.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    const [a, b] = pair;
    const cut = Math.round(split * W);
    ctx.drawImage(b, region.x, region.y, region.w, region.h, 0, 0, W, H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, cut, H);
    ctx.clip();
    ctx.drawImage(a, region.x, region.y, region.w, region.h, 0, 0, W, H);
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
