import { useCallback, useEffect, useMemo, useState } from "react";
import Chart from "./Chart.jsx";
import Viewer, { loadImage } from "./Viewer.jsx";

const DATA = `${import.meta.env.BASE_URL}data`;
const REPO = "https://github.com/samashech/Likes-of-DLSS";

const PIPELINE = [
  ["Render", "Each pixel is an area integral", "Simpson's 1/3 rule, Exp 5 and 6"],
  ["Upscale", "Polynomial through nearby samples", "Lagrange and Newton differences, Exp 3 and 4"],
  ["Edges", "Eigenvectors of the structure tensor", "Power method, Exp 8"],
  ["Clean up", "Edge-preserving diffusion in time", "Runge-Kutta 4, Exp 7"],
  ["Tune", "Best σ, ghost budget for α", "Bisection, regula falsi, secant, Newton, Exp 2a to 2d"],
  ["Motion", "Register frame against history", "Gauss-Newton and fsolve, Exp 2e"],
];

/** Read and write the view state in the query string, so a view can be shared. */
function useUrlState(defaults) {
  const [state, setState] = useState(() => {
    const q = new URLSearchParams(window.location.search);
    const s = { ...defaults };
    for (const key of Object.keys(defaults)) {
      if (q.has(key)) s[key] = typeof defaults[key] === "number" ? Number(q.get(key)) : q.get(key);
    }
    return s;
  });
  useEffect(() => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(state)) if (v !== defaults[k]) q.set(k, String(v));
    const qs = q.toString();
    window.history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
  }, [state]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = useCallback((patch) => setState((s) => ({ ...s, ...patch })), []);
  return [state, set];
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export default function App() {
  const [manifest, setManifest] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [view, setView] = useUrlState({ left: "lagrange", right: "full", region: "zone", path: "tuning", frame: 19 });
  const [split, setSplit] = useState(0.5);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch(`${DATA}/manifest.json`)
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setManifest)
      .catch(() => setLoadError(true));
  }, []);

  const path = manifest?.paths[view.path];
  const frames = path?.frames ?? 32;
  const frame = Math.min(Math.max(0, view.frame | 0), frames - 1);

  // playback: one frame every 250 ms, stopping on the last frame
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setView({ frame: frame + 1 >= frames ? frames - 1 : frame + 1 });
      if (frame + 1 >= frames - 1) setPlaying(false);
    }, 250);
    return () => clearInterval(id);
  }, [playing, frame, frames, setView]);

  // preload both sides' frames so scrubbing and playback never wait
  useEffect(() => {
    if (!manifest) return;
    for (let k = 0; k < frames; k += 1) {
      loadImage(`${DATA}/${view.path}/${view.left}/${k}.png`).catch(() => {});
      loadImage(`${DATA}/${view.path}/${view.right}/${k}.png`).catch(() => {});
    }
  }, [manifest, view.path, view.left, view.right, frames]);

  const byId = useMemo(() => Object.fromEntries((manifest?.methods ?? []).map((m) => [m.id, m])), [manifest]);
  const series = useMemo(() => {
    if (!path) return {};
    // the chart compares against ground truth, so the truth itself has no line
    return path.psnr;
  }, [path]);

  if (loadError) {
    return (
      <Shell>
        <div className="container error" role="alert">
          <h2>No exported frames found</h2>
          <p>
            The demo reads precomputed frames from <code>web/public/data</code>. Create them with{" "}
            <code>.venv/bin/python scripts/export_web.py</code>, then reload this page.
          </p>
        </div>
      </Shell>
    );
  }

  if (!manifest) {
    return (
      <Shell>
        <div className="container bench" aria-busy="true">
          <div className="sheet"><div className="viewer skeleton" style={{ aspectRatio: "1 / 1" }} /></div>
          <div className="panel"><p className="hint">Loading the manifest</p></div>
        </div>
      </Shell>
    );
  }

  const region = manifest.regions.find((r) => r.id === view.region) ?? manifest.regions[0];
  const src = (method) => `${DATA}/${view.path}/${method}/${frame}.png`;
  const score = (id, k = frame) => (id === "truth" ? null : path.psnr[id][k]);
  const fmt = (v) => (v == null ? "exact" : `${v.toFixed(2)} dB`);
  const leftM = byId[view.left];
  const rightM = byId[view.right];
  const chartable = view.left !== "truth" && view.right !== "truth";
  const best = path.psnr.full.at(-1);
  const baseline = path.psnr.bilinear.at(-1);
  const gain = best - baseline;
  const jit = path.jitter[frame];

  return (
    <Shell>
      <div className="container">
      <header className="pagehead">
        <div>
          <h1>Upscaling 4× by accumulating jittered frames</h1>
          <p>
            A 128 × 128 renderer, rebuilt at 512 × 512 from its own sub-pixel jitter. Drag across the image
            to compare two methods on the same frame.
          </p>
        </div>
        <div className="pagehead__figure">
          <span className="num">+{gain.toFixed(1)} dB</span>
          <small>last frame, temporal against bilinear ({path.label.toLowerCase()})</small>
        </div>
      </header>

      <div className="bench">
        <section className="sheet" aria-labelledby="viewer-title">
          <div className="sheet__head">
            <h2 id="viewer-title">{region.label}, frame {frame}</h2>
            <span className="tag">{region.test} test</span>
          </div>
          <div className="sides">
            <div><strong>{leftM.label}</strong> <span className="num">{fmt(score(view.left))}</span></div>
            <div className="is-right"><strong>{rightM.label}</strong> <span className="num">{fmt(score(view.right))}</span></div>
          </div>
          <Viewer
            leftSrc={src(view.left)}
            rightSrc={src(view.right)}
            region={region}
            split={split}
            onSplit={setSplit}
            leftLabel={leftM.label}
            rightLabel={rightM.label}
          />
          <p className="caption">
            {frame < manifest.switchFrame
              ? `Frames 0 to ${manifest.switchFrame - 1}: the scene is static while the camera pans.`
              : `From frame ${manifest.switchFrame} a panel is lit. Nothing moved, so only the clamp can catch it.`}
          </p>

          <div className="transport">
            <button type="button" className="btn btn--primary" onClick={() => {
              if (!playing && frame >= frames - 1) setView({ frame: 0 });
              setPlaying(!playing);
            }}>
              {playing ? "Pause" : "Play"}
            </button>
            <div className="transport__frame">frame <span className="num">{String(frame).padStart(2, "0")}</span> of <span className="num">{frames - 1}</span></div>
            <label className="visually-hidden" htmlFor="frame">Frame</label>
            <input id="frame" type="range" min={0} max={frames - 1} value={frame}
              onChange={(e) => { setPlaying(false); setView({ frame: Number(e.target.value) }); }} />
            <div className="transport__step">
              <button type="button" className="btn btn--icon" aria-label="Previous frame" disabled={frame === 0}
                onClick={() => { setPlaying(false); setView({ frame: frame - 1 }); }}>−</button>
              <button type="button" className="btn btn--icon" aria-label="Next frame" disabled={frame === frames - 1}
                onClick={() => { setPlaying(false); setView({ frame: frame + 1 }); }}>+</button>
            </div>
          </div>
        </section>

        <aside className="panel" aria-label="View settings">
          <div className="field">
            <label htmlFor="left">Left of the wipe</label>
            <select id="left" value={view.left} onChange={(e) => setView({ left: e.target.value })}>
              {manifest.methods.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="right">Right of the wipe</label>
            <select id="right" value={view.right} onChange={(e) => setView({ right: e.target.value })}>
              {manifest.methods.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </div>
          <div className="field">
            <fieldset>
              <legend>Region</legend>
              <div className="segmented">
                {manifest.regions.map((r) => (
                  <label key={r.id}>
                    <input type="radio" name="region" value={r.id} checked={view.region === r.id}
                      onChange={() => setView({ region: r.id })} />
                    <span>{r.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
          <div className="field">
            <fieldset>
              <legend>Camera path</legend>
              <div className="segmented">
                {Object.entries(manifest.paths).map(([id, p]) => (
                  <label key={id}>
                    <input type="radio" name="path" value={id} checked={view.path === id}
                      onChange={() => setView({ path: id })} />
                    <span>{p.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>
            <p className="hint">
              Pan <span className="num">({path.pan[0]}, {path.pan[1]})</span> px per frame.
              {view.path === "heldout" ? " Parameters were never tuned on this path." : " Parameters were tuned on this path."}
            </p>
          </div>

          <table>
            <caption className="visually-hidden">PSNR by method</caption>
            <thead>
              <tr><th>Method</th><th className="num">Frame {frame}</th><th className="num">Mean</th></tr>
            </thead>
            <tbody>
              {manifest.methods.filter((m) => m.id !== "truth").map((m) => (
                <tr key={m.id} className={m.id === view.right ? "is-right" : m.id === view.left ? "is-left" : ""}>
                  <td>
                    <button type="button" className="rowbtn" onClick={() => setView({ right: m.id })}
                      aria-label={`Show ${m.label} on the right`}>{m.label}</button>
                  </td>
                  <td className="num">{path.psnr[m.id][frame].toFixed(2)}</td>
                  <td className="num">{mean(path.psnr[m.id]).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">PSNR in dB against the ground truth; higher is better, +6 dB halves the error.</p>
        </aside>
      </div>

      <section className="chartwrap sheet" aria-labelledby="chart-title">
        <div className="sheet__head">
          <h2 id="chart-title">Quality as frames accumulate</h2>
          <p>Click or drag to move to a frame.</p>
        </div>
        {chartable ? (
          <>
            <div className="chartscroll">
            <Chart series={series} left={view.left} right={view.right} frame={frame} frames={frames}
              switchFrame={manifest.switchFrame} onSeek={(k) => { setPlaying(false); setView({ frame: k }); }} />
            </div>
            <div className="legend">
              <span className="is-right"><i />{rightM.label}</span>
              <span><i />{leftM.label}</span>
              <span className="is-ghost"><i />other methods</span>
            </div>
          </>
        ) : (
          <p className="hint">Ground truth scores perfectly by definition. Pick two methods to compare their curves.</p>
        )}
      </section>
      </div>

      <section className="band" aria-labelledby="input-title">
        <div className="container band__grid">
          <div>
            <h2 id="input-title">What the renderer sent for frame {frame}</h2>
            <img className="lowres" src={`${DATA}/${view.path}/low/${frame}.png`} width={128} height={128}
              alt={`Low-resolution input for frame ${frame}`} />
            <p className="caption">
              128 × 128, one sharp sample per pixel, shifted by <span className="num">({jit[0] >= 0 ? "+" : ""}{jit[0].toFixed(3)}, {jit[1] >= 0 ? "+" : ""}{jit[1].toFixed(3)})</span> of a pixel by the Halton sequence.
            </p>
          </div>
          <div>
            <h2>How each step is computed</h2>
            <table>
              <thead><tr><th>Step</th><th>What it does</th><th>Numerical method</th></tr></thead>
              <tbody>
                {PIPELINE.map(([step, what, how]) => (
                  <tr key={step}><td>{step}</td><td>{what}</td><td><span className="exp">{how}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </Shell>
  );
}

function Shell({ children }) {
  return (
    <>
      <nav className="topbar" aria-label="Site">
        <div className="container topbar__inner">
          <a className="wordmark" href="./" aria-label="Likes of DLSS, home">Likes of DLSS <span>4×</span></a>
          <div className="topbar__meta">
            <span>Numerical methods lab project</span>
            <a href={`${REPO}/blob/main/decisions.md`}>Design decisions</a>
            <a href={REPO}>Source</a>
          </div>
        </div>
      </nav>
      <main>{children}</main>
      <footer className="container footer">
        Every method is hand-written in numpy and checked against scipy. Figures are measured on a synthetic scene with exact ground truth.
      </footer>
    </>
  );
}
