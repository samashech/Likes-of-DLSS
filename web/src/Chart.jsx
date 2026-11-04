import { useRef } from "react";

const W = 960;
const H = 300;
const PAD = { l: 44, r: 64, t: 16, b: 32 };

/**
 * One chart, set large: PSNR against frame. The right-hand method is the
 * series the page argues from (magenta); the left is ink; every other method
 * sits behind as a faint line for context. Click or drag to seek.
 */
export default function Chart({ series, left, right, frame, frames, switchFrame, onSeek }) {
  const svg = useRef(null);
  const all = Object.values(series).flat();
  const lo = Math.floor(Math.min(...all)) - 1;
  const hi = Math.ceil(Math.max(...all)) + 1;
  const x = (k) => PAD.l + (k / (frames - 1)) * (W - PAD.l - PAD.r);
  const y = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
  const path = (vals) => vals.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const seek = (e) => {
    const r = svg.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const k = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (frames - 1));
    onSeek(Math.min(frames - 1, Math.max(0, k)));
  };

  const yTicks = [];
  for (let v = Math.ceil(lo / 2) * 2; v <= hi; v += 2) yTicks.push(v);
  const xTicks = [];
  for (let k = 0; k < frames; k += 4) xTicks.push(k);
  if (xTicks.at(-1) !== frames - 1) xTicks.push(frames - 1);

  const rv = series[right][frame];
  const lv = series[left][frame];
  const ry = y(rv);
  // end-of-line labels, nudged apart when the two lines finish close together
  const rEnd = y(series[right].at(-1));
  let lEnd = y(series[left].at(-1));
  if (Math.abs(lEnd - rEnd) < 16) lEnd = rEnd + (lEnd >= rEnd ? 16 : -16);

  return (
    <svg
      ref={svg}
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`PSNR by frame. At frame ${frame}: ${right} ${rv.toFixed(2)} dB, ${left} ${lv.toFixed(2)} dB.`}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); seek(e); }}
      onPointerMove={(e) => { if (e.buttons) seek(e); }}
    >
      {yTicks.map((v) => (
        <g key={v}>
          <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
          <text className="tick" x={PAD.l - 8} y={y(v) + 4} textAnchor="end">{v}</text>
        </g>
      ))}
      <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />
      {xTicks.map((k) => (
        <text key={k} className="tick" x={x(k)} y={H - PAD.b + 18} textAnchor="middle">{k}</text>
      ))}
      <line className="event" x1={x(switchFrame)} x2={x(switchFrame)} y1={PAD.t} y2={H - PAD.b} />
      <text className="event-label" x={x(switchFrame) + 6} y={PAD.t + 12}>light switches on</text>

      {Object.entries(series).filter(([id]) => id !== left && id !== right).map(([id, vals]) => (
        <path key={id} className="series--ghost" d={path(vals)} />
      ))}
      <path className="series--left" d={path(series[left])} />
      <path className="series" d={path(series[right])} />

      <line className="cursor" x1={x(frame)} x2={x(frame)} y1={PAD.t} y2={H - PAD.b} />
      <circle className="dot--left" cx={x(frame)} cy={y(lv)} r={3.5} />
      <circle className="dot" cx={x(frame)} cy={ry} r={4} />
      <text className="label label--right" x={W - PAD.r + 8} y={rEnd + 4}>{series[right].at(-1).toFixed(2)}</text>
      <text className="label label--left" x={W - PAD.r + 8} y={lEnd + 4}>{series[left].at(-1).toFixed(2)}</text>
      <text className="tick" x={PAD.l} y={H - 2} textAnchor="start">frame</text>
      <text className="tick" x={4} y={PAD.t + 4} textAnchor="start">dB</text>
    </svg>
  );
}
