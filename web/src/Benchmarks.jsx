import { useEffect, useState } from "react";
import { SECTIONS } from "./benchmarks.js";

const DATA = `${import.meta.env.BASE_URL}data`;
const FIG = `${import.meta.env.BASE_URL}figures`;
const DECISIONS = "https://github.com/samashech/Likes-of-DLSS/blob/main/decisions.md";
const COLS = [0, 4, 8, 16];
const FIG_SIZE = { "01_frames.png": [1650, 1100], "02_spatial.png": [1600, 1000], "03_temporal.png": [1600, 1300],
  "04_edges.png": [1600, 1300], "05_tuning.png": [1700, 1000], "06_motion.png": [2000, 480] };

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/** Every scene's manifest, the test scene's two paths listed separately. */
function useAllScenes(scenes) {
  const [rows, setRows] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!scenes) return;
    Promise.all(scenes.map((s) => fetch(s.id === "test" ? `${DATA}/manifest.json` : `${DATA}/scenes/${s.id}/manifest.json`)
      .then((r) => r.json()).then((m) => ({ ...s, m }))))
      .then((all) => {
        const out = [];
        for (const { id, label, kind, m } of all) {
          if (kind === "raw" || m.hasTruth === false) continue;
          for (const [pid, p] of Object.entries(m.paths)) {
            out.push({
              key: `${id}-${pid}`, scene: id, label: kind === "synthetic" ? `Test scene, ${p.label.toLowerCase()}` : label,
              kind, methods: m.methods.filter((x) => x.id !== "truth"), psnr: p.psnr, frames: p.frames,
              res: kind === "synthetic" ? "128 × 128 → 512 × 512" : `${m.lowWidth} × ${m.lowHeight} → ${m.width} × ${m.height}`,
            });
          }
        }
        setRows(out);
      })
      .catch(() => setFailed(true));
  }, [scenes]);
  return [rows, failed];
}

export default function Benchmarks({ scenes, onOpen }) {
  const [rows, failed] = useAllScenes(scenes);

  return (
    <div className="container bench-page">
      <header className="pagehead">
        <div>
          <h1>Benchmarks</h1>
          <p>
            Every figure below was measured on this project's code, against exact ground truth. PSNR is in dB; higher is
            better and +6&nbsp;dB halves the RMS error.
          </p>
        </div>
      </header>

      <nav className="toc" aria-label="On this page">
        <a href="#quality">Quality by scene</a>
        {SECTIONS.map((s) => <a key={s.id} href={`#${s.id}`}>{s.title}</a>)}
        <a href="#setup">Test setup</a>
      </nav>

      <section id="quality" className="sheet bm" aria-labelledby="quality-title">
        <div className="sheet__head"><h2 id="quality-title">Quality by scene, last frame</h2></div>
        {failed && <p className="hint" role="alert">The scene data did not load. Run the export and import scripts, then reload.</p>}
        {!rows && !failed && <div className="skeleton bm__skeleton" aria-busy="true" />}
        {rows && (
          <div className="tablescroll">
            <table>
              <thead>
                <tr><th>Scene</th><th>Resolution</th><th className="num">Bilinear</th><th className="num">Lagrange</th><th className="num">Lanczos-3</th><th className="num">Temporal</th><th className="num">Gain</th><th className="num">RMS error</th></tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const g = r.psnr.full.at(-1) - r.psnr.bilinear.at(-1);
                  return (
                    <tr key={r.key}>
                      <td><button type="button" className="rowbtn" onClick={() => onOpen(r.scene)}>{r.label}</button></td>
                      <td>{r.res}</td>
                      <td className="num">{r.psnr.bilinear.at(-1).toFixed(2)}</td>
                      <td className="num">{r.psnr.lagrange.at(-1).toFixed(2)}</td>
                      <td className="num">{r.psnr.lanczos ? r.psnr.lanczos.at(-1).toFixed(2) : "not run"}</td>
                      <td className="num is-accent">{r.psnr.full.at(-1).toFixed(2)}</td>
                      <td className="num">+{g.toFixed(2)}</td>
                      <td className="num">{(10 ** (g / 20)).toFixed(2)}× <span className="unit">lower</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <p className="bm__finding">
          Temporal is the full method: edge-aware clamp plus RK4 diffusion. Gain is against bilinear, the best
          single-frame method on every scene. The game scenes use parameters tuned on the
          test scene and never re-tuned, so they double as a transfer test.
        </p>
      </section>

      {rows && rows.map((r) => (
        <details key={r.key} className="sheet bm bm--detail">
          <summary><h3>{r.label}: every method by frame</h3><span className="hint">{r.frames} frames</span></summary>
          <div className="tablescroll">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  {COLS.filter((k) => k < r.frames).map((k) => <th key={k} className="num">Frame {k}</th>)}
                  <th className="num">Last</th><th className="num">Mean</th>
                </tr>
              </thead>
              <tbody>
                {r.methods.map((m) => (
                  <tr key={m.id} className={m.id === "full" ? "is-right" : ""}>
                    <td>{m.label}</td>
                    {COLS.filter((k) => k < r.frames).map((k) => <td key={k} className="num">{r.psnr[m.id][k].toFixed(2)}</td>)}
                    <td className="num">{r.psnr[m.id].at(-1).toFixed(2)}</td>
                    <td className="num">{mean(r.psnr[m.id]).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ))}

      {SECTIONS.map((s) => (
        <section key={s.id} id={s.id} className="sheet bm" aria-labelledby={`${s.id}-title`}>
          <div className="sheet__head">
            <h2 id={`${s.id}-title`}>{s.title}</h2>
            <span className="tag">{s.exp}</span>
          </div>
          <p className="hint">{s.note}</p>
          <div className="tablescroll">
            <table>
              <thead><tr>{s.head.map((h, i) => <th key={h} className={i ? "num" : ""}>{h}</th>)}</tr></thead>
              <tbody>
                {s.rows.map((row, ri) => (
                  <tr key={row[0]} className={s.highlight?.[0] === ri ? "is-right" : ""}>
                    {row.map((c, i) => <td key={i} className={i ? "num" : ""}>{c}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="bm__finding">{s.finding}</p>
          <div className="bm__refs">
            {s.decision && <a href={`${DECISIONS}`}>decisions.md §{s.decision}</a>}
            {s.figure && <a href={`${FIG}/${s.figure}`}>Open the full figure</a>}
          </div>
          {s.figure && (
            <a className="bm__figure" href={`${FIG}/${s.figure}`}>
              <img src={`${FIG}/${s.figure}`} alt={`Figure for ${s.title.toLowerCase()}`} loading="lazy"
                width={FIG_SIZE[s.figure][0]} height={FIG_SIZE[s.figure][1]} />
            </a>
          )}
          {s.figure && (
            <p className="caption">
              Generated by <code className="code">scripts/{s.figure.replace(".png", ".py")}</code> when that module was built;
              curves in figures made before tuning use the hand-picked settings.
            </p>
          )}
        </section>
      ))}

      <section id="setup" className="sheet bm" aria-labelledby="setup-title">
        <div className="sheet__head"><h2 id="setup-title">Test setup</h2></div>
        <dl className="bm__setup">
          <dt>Test scene</dt><dd>Analytic: rotated checkerboard, zone plate, thin bars. 32 frames, 4×, a light panel switches on at frame 20.</dd>
          <dt>Game scenes</dt><dd>A screenshot is the ground truth; a simulated renderer pans across it and takes 320 × 180 Halton-jittered samples. 24 frames, rebuilt at 1280 × 720, one colour channel at a time.</dd>
          <dt>Camera</dt><dd>Pan of (0.37, 0.21) px per frame with exact motion vectors; held-out path (−0.29, 0.43).</dd>
          <dt>Metric</dt><dd>PSNR against ground truth, ignoring an 8 px border; computed before any image compression.</dd>
          <dt>Caveats</dt><dd>Synthetic jitter, not a game engine's own frames. Motion is a global pan. Parameters tuned on the test scene.</dd>
        </dl>
      </section>
    </div>
  );
}
