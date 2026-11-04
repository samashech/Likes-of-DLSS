import { useCallback, useEffect, useMemo, useState } from "react";
import AddScene from "./AddScene.jsx";
import Benchmarks from "./Benchmarks.jsx";
import Chart from "./Chart.jsx";
import Logo from "./Logo.jsx";
import RegionPicker from "./RegionPicker.jsx";
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

const DEFAULTS = { page: "demo", scene: "test", left: "lagrange", right: "full", region: "", path: "", frame: -1 };

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

/** The synthetic test scene's manifest, in the same shape as an imported scene's. */
function normaliseTest(m) {
  return {
    ...m, kind: "synthetic", label: "Test scene", base: "", ext: "png",
    width: m.size, height: m.size, lowWidth: m.lowSize, lowHeight: m.lowSize, hasTruth: true,
    overview: "tuning/truth/0.png", motionNote: "camera pan, motion vectors given",
  };
}

/** A region id "x,y,w,h" is a custom box; anything else names a preset. */
function resolveRegion(scene, id) {
  const m = /^(\d+),(\d+),(\d+),(\d+)$/.exec(id || "");
  if (m) {
    const [x, y, w, h] = m.slice(1).map(Number);
    return { id, label: "Picked region", test: "your pick", x, y, w, h };
  }
  return scene.regions.find((r) => r.id === id) ?? scene.regions[0];
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const clamp = (v, lo, hi) => Math.min(Math.max(lo, v), hi);
const sceneCache = new Map();

export default function App() {
  const [scenes, setScenes] = useState(null);
  const [scene, setScene] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [view, setView] = useUrlState(DEFAULTS);
  const [split, setSplit] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [adding, setAdding] = useState(() => window.location.hash === "#add");

  // after an import: reload the list, forget any stale copy, open the new scene
  const onImported = useCallback((id) => {
    fetch(`${DATA}/scenes.json?t=${Date.now()}`).then((r) => r.json()).then((list) => {
      sceneCache.delete(id);
      setScenes([{ id: "test", label: "Test scene", kind: "synthetic" }, ...list]);
      setPlaying(false);
      setView({ scene: id, region: "", path: "", frame: -1 });
      setTimeout(() => setAdding(false), 1500);
    });
  }, [setView]);

  // the scene list: the synthetic test scene plus anything imported
  useEffect(() => {
    Promise.all([
      fetch(`${DATA}/manifest.json`).then((r) => { if (!r.ok) throw new Error(); return r.json(); }),
      fetch(`${DATA}/scenes.json`).then((r) => (r.ok ? r.json() : [])).catch(() => []),
    ]).then(([test, list]) => {
      sceneCache.set("test", normaliseTest(test));
      setScenes([{ id: "test", label: "Test scene", kind: "synthetic" }, ...list]);
    }).catch(() => setLoadError(true));
  }, []);

  const sceneId = scenes && (scenes.some((s) => s.id === view.scene) ? view.scene : "test");

  // the selected scene's manifest, fetched on first use
  useEffect(() => {
    if (!sceneId) return;
    if (sceneCache.has(sceneId)) { setScene(sceneCache.get(sceneId)); return; }
    setScene(null);
    fetch(`${DATA}/scenes/${sceneId}/manifest.json?t=${Date.now()}`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((m) => { sceneCache.set(sceneId, m); setScene(m); })
      .catch(() => setLoadError(true));
  }, [sceneId]);

  const pathId = scene && (scene.paths[view.path] ? view.path : Object.keys(scene.paths)[0]);
  const path = scene?.paths[pathId];
  const frames = path?.frames ?? 1;
  const frame = view.frame < 0 || view.frame >= frames
    ? (scene?.switchFrame ? scene.switchFrame - 1 : frames - 1)
    : view.frame | 0;
  const ids = useMemo(() => new Set((scene?.methods ?? []).map((m) => m.id)), [scene]);
  const left = ids.has(view.left) ? view.left : "lagrange";
  const right = ids.has(view.right) ? view.right : "full";
  const dir = scene ? `${DATA}/${scene.base ? `${scene.base}/` : ""}` : "";
  const ver = scene?.version ? `?v=${scene.version}` : "";
  const src = useCallback((method, k) => `${dir}${pathId}/${method}/${k}.${scene.ext}${ver}`, [dir, pathId, scene, ver]);

  // playback: one frame every 250 ms, stopping on the last frame
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setView({ frame: Math.min(frame + 1, frames - 1) });
      if (frame + 1 >= frames - 1) setPlaying(false);
    }, 250);
    return () => clearInterval(id);
  }, [playing, frame, frames, setView]);

  // preload both sides' frames so scrubbing and playback never wait
  useEffect(() => {
    if (!scene) return;
    for (let k = 0; k < frames; k += 1) {
      loadImage(src(left, k)).catch(() => {});
      loadImage(src(right, k)).catch(() => {});
    }
  }, [scene, src, left, right, frames]);

  if (loadError) {
    return (
      <Shell>
        <div className="container error" role="alert">
          <h2>No exported frames found</h2>
          <p>
            The demo reads precomputed frames from <code>web/public/data</code>. Create them with{" "}
            <code>.venv/bin/python scripts/export_web.py</code>, add game scenes with{" "}
            <code>scripts/import_scene.py</code>, then reload this page.
          </p>
        </div>
      </Shell>
    );
  }

  const nav = { page: view.page, onPage: (page) => { setPlaying(false); setView({ page }); window.scrollTo(0, 0); } };

  if (view.page === "benchmarks") {
    return (
      <Shell {...nav}>
        <Benchmarks scenes={scenes} onOpen={(id) => { setView({ page: "demo", scene: id, region: "", path: "", frame: -1 }); window.scrollTo(0, 0); }} />
      </Shell>
    );
  }

  const tabs = scenes && (
    <>
      <div className="scenes">
        <div className="segmented" role="radiogroup" aria-label="Scene">
          {scenes.map((s) => (
            <label key={s.id}>
              <input type="radio" name="scene" value={s.id} checked={sceneId === s.id}
                onChange={() => { setPlaying(false); setView({ scene: s.id, region: "", path: "", frame: -1 }); }} />
              <span>{s.label}</span>
            </label>
          ))}
        </div>
        {!adding && <button type="button" className="btn" onClick={() => setAdding(true)}>Add a scene</button>}
      </div>
      {adding && <AddScene onDone={onImported} onClose={() => setAdding(false)} />}
    </>
  );

  if (!scene) {
    return (
      <Shell {...nav}>
        <div className="container">
          <div className="pagehead"><div><h1>Loading the scene</h1></div></div>
          {tabs}
          <div className="bench" aria-busy="true">
            <div className="sheet"><div className="viewer skeleton" style={{ aspectRatio: "16 / 9" }} /></div>
            <div className="panel"><p className="hint">Fetching frames and scores.</p></div>
          </div>
        </div>
      </Shell>
    );
  }

  const region = resolveRegion(scene, view.region);
  const byId = Object.fromEntries(scene.methods.map((m) => [m.id, m]));
  const truthy = scene.hasTruth;
  const fmt = (id) => (!truthy ? "" : id === "truth" ? "reference" : `${path.psnr[id][frame].toFixed(2)} dB`);
  const leftM = byId[left];
  const rightM = byId[right];
  const chartable = truthy && left !== "truth" && right !== "truth";
  const gain = truthy ? path.psnr.full.at(-1) - path.psnr.bilinear.at(-1) : null;
  const jit = path.jitter[frame];
  const isGame = scene.kind !== "synthetic";
  const sizes = isGame ? [120, 160, 240] : [80, 120, 160];
  const aspect = scene.height / scene.width;
  const isFull = region.w >= scene.width;
  const pick = (r) => setView({ region: `${r.x},${r.y},${r.w},${r.h}` });
  const resize = (w) => {
    const h = Math.round(w * aspect);
    const cx = isFull ? scene.width / 2 : region.x + region.w / 2;
    const cy = isFull ? scene.height / 2 : region.y + region.h / 2;
    pick({ x: Math.round(clamp(cx - w / 2, 0, scene.width - w)), y: Math.round(clamp(cy - h / 2, 0, scene.height - h)), w, h });
  };
  // on "Full frame" the picker still needs a box size for the next click
  const pickerRegion = isFull ? { x: 0, y: 0, w: sizes[1], h: Math.round(sizes[1] * aspect) } : region;

  return (
    <Shell {...nav}>
      <div className="container">
        <header className="pagehead">
          <div>
            <h1>{isGame ? `${scene.label}, rebuilt from ${scene.lowWidth} × ${scene.lowHeight} frames` : "Upscaling 4× by accumulating jittered frames"}</h1>
            <p>
              {scene.kind === "game"
                ? `The screenshot is the world. A simulated renderer pans across it and sends ${scene.lowWidth} × ${scene.lowHeight} jittered frames; the upscaler rebuilds ${scene.width} × ${scene.height}.`
                : scene.kind === "raw"
                  ? `Your own ${scene.lowWidth} × ${scene.lowHeight} jittered frames, upscaled ${scene.scale}× to ${scene.width} × ${scene.height}. No ground truth, so no scores.`
                  : "A 128 × 128 renderer, rebuilt at 512 × 512 from its own sub-pixel jitter. Drag across the image to compare two methods on the same frame."}
            </p>
          </div>
          {truthy && (
            <div className="pagehead__figure">
              <span className="num">+{gain.toFixed(1)} dB</span>
              <small>last frame, temporal against bilinear</small>
            </div>
          )}
        </header>

        {tabs}

        <div className="bench">
          <section className="sheet" aria-labelledby="viewer-title">
            <div className="sheet__head">
              <h2 id="viewer-title">{region.label}, frame {frame}</h2>
              <span className="tag">{region.test}</span>
            </div>
            <div className="sides">
              <div><strong>{leftM.label}</strong> <span className="num">{fmt(left)}</span></div>
              <div className="is-right"><strong>{rightM.label}</strong> <span className="num">{fmt(right)}</span></div>
            </div>
            <Viewer leftSrc={src(left, frame)} rightSrc={src(right, frame)} region={region} split={split}
              onSplit={setSplit} leftLabel={leftM.label} rightLabel={rightM.label} />
            <p className="caption">
              {scene.kind === "synthetic"
                ? (frame < scene.switchFrame
                  ? `Frames 0 to ${scene.switchFrame - 1}: the scene is static while the camera pans.`
                  : `From frame ${scene.switchFrame} a panel is lit. Nothing moved, so only the clamp can catch it.`)
                : `Source: ${scene.source}; ${scene.motionNote}.`}
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
              <select id="left" value={left} onChange={(e) => setView({ left: e.target.value })}>
                {scene.methods.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="right">Right of the wipe</label>
              <select id="right" value={right} onChange={(e) => setView({ right: e.target.value })}>
                {scene.methods.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </div>
            <div className="field">
              <fieldset>
                <legend>Region</legend>
                <div className="segmented">
                  {scene.regions.map((r) => (
                    <label key={r.id}>
                      <input type="radio" name="region" value={r.id} checked={region.id === r.id}
                        onChange={() => setView({ region: r.id })} />
                      <span>{r.label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <RegionPicker src={`${dir}${scene.overview ?? "overview.webp"}${ver}`} width={scene.width} height={scene.height}
                region={pickerRegion} onPick={pick} showBox={!isFull} />
              <div className="picker__row">
                <span className="hint">Click the thumbnail to inspect any spot.</span>
                <div className="segmented" role="radiogroup" aria-label="Region size">
                  {sizes.map((w, i) => (
                    <label key={w}>
                      <input type="radio" name="size" checked={!isFull && region.w === w} onChange={() => resize(w)} />
                      <span>{["Close", "Medium", "Wide"][i]}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            {Object.keys(scene.paths).length > 1 && (
              <div className="field">
                <fieldset>
                  <legend>Camera path</legend>
                  <div className="segmented">
                    {Object.entries(scene.paths).map(([id, p]) => (
                      <label key={id}>
                        <input type="radio" name="path" value={id} checked={pathId === id}
                          onChange={() => setView({ path: id })} />
                        <span>{p.label}</span>
                      </label>
                    ))}
                  </div>
                </fieldset>
                <p className="hint">
                  Pan <span className="num">({path.pan[0]}, {path.pan[1]})</span> px per frame.
                  {pathId === "heldout" ? " Parameters were never tuned on this path." : " Parameters were tuned on this path."}
                </p>
              </div>
            )}

            {truthy ? (
              <>
                <table>
                  <caption className="visually-hidden">PSNR by method</caption>
                  <thead>
                    <tr><th>Method</th><th className="num">Frame {frame}</th><th className="num">Mean</th></tr>
                  </thead>
                  <tbody>
                    {scene.methods.filter((m) => m.id !== "truth").map((m) => (
                      <tr key={m.id} className={m.id === right ? "is-right" : m.id === left ? "is-left" : ""}>
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
                <p className="hint">PSNR in dB against the ground truth; higher is better, +6&nbsp;dB halves the error.</p>
              </>
            ) : (
              <p className="hint">These frames came without a high-resolution reference, so there is nothing to score against. Judge by eye with the wipe.</p>
            )}
          </aside>
        </div>

        <section className="chartwrap sheet" aria-labelledby="chart-title">
          <div className="sheet__head">
            <h2 id="chart-title">Quality as frames accumulate</h2>
            {chartable && <p>Click or drag to move to a frame.</p>}
          </div>
          {chartable ? (
            <>
              <div className="chartscroll">
                <Chart series={path.psnr} left={left} right={right} frame={frame} frames={frames}
                  switchFrame={scene.switchFrame} onSeek={(k) => { setPlaying(false); setView({ frame: k }); }} />
              </div>
              <div className="legend">
                <span className="is-right"><i />{rightM.label}</span>
                <span><i />{leftM.label}</span>
                <span className="is-ghost"><i />other methods</span>
              </div>
            </>
          ) : (
            <p className="hint">
              {truthy
                ? "Ground truth scores perfectly by definition. Pick two methods to compare their curves."
                : "No ground truth for these frames, so there is no curve to draw."}
            </p>
          )}
        </section>
      </div>

      <section className="band" aria-labelledby="input-title">
        <div className="container band__grid">
          <div>
            <h2 id="input-title">What the renderer sent for frame {frame}</h2>
            <img className="lowres" src={src("low", frame)} width={scene.lowWidth} height={scene.lowHeight}
              style={{ aspectRatio: `${scene.lowWidth} / ${scene.lowHeight}` }}
              alt={`Low-resolution input for frame ${frame}`} />
            <p className="caption">
              {scene.lowWidth} × {scene.lowHeight}, one sharp sample per pixel, shifted by <span className="num">({jit[0] >= 0 ? "+" : ""}{jit[0].toFixed(3)}, {jit[1] >= 0 ? "+" : ""}{jit[1].toFixed(3)})</span> of a pixel
              {scene.kind === "raw" ? "." : " by the Halton sequence."}
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

function Shell({ children, page = "demo", onPage = () => {} }) {
  return (
    <>
      <nav className="topbar" aria-label="Site">
        <div className="container topbar__inner">
          <a className="wordmark" href="./" aria-label="Likes of DLSS, home"><Logo />Likes of DLSS <span>4×</span></a>
          <div className="topbar__meta">
            <a href="?page=benchmarks" aria-current={page === "benchmarks" ? "page" : undefined}
              onClick={(e) => { e.preventDefault(); onPage(page === "benchmarks" ? "demo" : "benchmarks"); }}>
              {page === "benchmarks" ? "Back to the demo" : "Read the benchmarks"}
            </a>
            <a href={`${REPO}/blob/main/decisions.md`}>Design decisions</a>
            <a href={REPO}>Source</a>
          </div>
        </div>
      </nav>
      <main>{children}</main>
      <footer className="container footer">
        Every method is hand-written in numpy and checked against scipy. Game scenes use the screenshot as ground truth and a simulated renderer for the jitter.
      </footer>
    </>
  );
}
