import { useEffect, useRef, useState } from "react";

const STEPS = ["Rendering the frames", "Bilinear", "Lagrange cubic", "Temporal, edge-aware clamp", "Temporal, edge clamp + RK4"];
const EXAMPLE = `{
  "scale": 4,
  "jitter": [[0.0, -0.167], [-0.25, 0.167], ...],
  "motion": [[0, 0], [0.37, 0.21], ...]
}`;

const toBase64 = (file) => new Promise((ok, fail) => {
  const r = new FileReader();
  r.onload = () => ok(String(r.result).split(",")[1]);
  r.onerror = () => fail(new Error(`Could not read ${file.name}.`));
  r.readAsDataURL(file);
});

/** Inline panel to import a game screenshot or a folder of jittered frames. */
export default function AddScene({ onDone, onClose }) {
  const [server, setServer] = useState("checking");
  const [mode, setMode] = useState("image");
  const [name, setName] = useState("");
  const [image, setImage] = useState(null);
  const [frames, setFrames] = useState([]);
  const [framesJson, setFramesJson] = useState(null);
  const [estimate, setEstimate] = useState(false);
  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState("");
  const [job, setJob] = useState(null);
  const [now, setNow] = useState(Date.now());
  const nameRef = useRef(null);

  useEffect(() => {
    fetch("/api/health").then((r) => setServer(r.ok ? "ready" : "missing")).catch(() => setServer("missing"));
  }, []);
  useEffect(() => { if (server === "ready") nameRef.current?.focus(); }, [server]);

  // poll the running import once a second
  useEffect(() => {
    if (phase !== "running" || !job) return undefined;
    const id = setInterval(async () => {
      setNow(Date.now());
      try {
        const r = await fetch(`/api/import/${encodeURIComponent(job.id)}`);
        const j = await r.json();
        setJob((old) => ({ ...old, ...j }));
        if (j.status === "done") { setPhase("done"); onDone(j.scene); }
        if (j.status === "error") { setPhase("error"); setError("The import stopped. The last lines of its log are below."); }
      } catch {
        setPhase("error");
        setError("Lost contact with the dev server. Is npm run dev still running?");
      }
    }, 1000);
    return () => clearInterval(id);
  }, [phase, job, onDone]);

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    if (!name.trim()) { setError("Give the scene a name."); nameRef.current?.focus(); return; }
    if (mode === "image" && !image) { setError("Choose a screenshot."); return; }
    if (mode === "raw" && frames.length < 2) { setError("Choose at least two jittered frames."); return; }
    if (mode === "raw" && !framesJson) { setError("Choose the frames.json that lists each frame's jitter."); return; }
    setPhase("uploading");
    try {
      const list = mode === "image" ? [image] : [...frames].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      const files = await Promise.all(list.map(async (f) => ({ name: f.name, data: await toBase64(f) })));
      const body = { mode, name: name.trim(), files, estimateMotion: estimate,
        framesJson: mode === "raw" ? await framesJson.text() : undefined };
      const r = await fetch("/api/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `The server answered ${r.status}.`);
      setJob({ id: j.job, progress: [0, 5], log: [], started: Date.now() });
      setPhase("running");
    } catch (err) {
      setPhase("idle");
      setError(err.message);
    }
  };

  const elapsed = job ? Math.round(((job.finished ?? now) - job.started) / 1000) : 0;
  const [step, total] = job?.progress ?? [0, 5];
  const busy = phase === "uploading" || phase === "running";

  return (
    <section className="addscene sheet" aria-labelledby="add-title">
      <div className="sheet__head">
        <h2 id="add-title">Add a scene</h2>
        <button type="button" className="btn" onClick={onClose} disabled={busy}>Close</button>
      </div>

      {server === "checking" && <p className="hint">Checking for the import server.</p>}
      {server === "missing" && (
        <p className="hint">
          Importing needs the dev server, which runs the Python upscaler. Start the demo with{" "}
          <code className="code">npm run dev</code> in <code className="code">web/</code>, or run{" "}
          <code className="code">scripts/import_scene.py</code> from a terminal.
        </p>
      )}

      {server === "ready" && (phase === "idle" || phase === "uploading") && (
        <form className="addscene__form" onSubmit={submit} noValidate>
          <div className="segmented" role="radiogroup" aria-label="What are you adding">
            {[["image", "A game screenshot"], ["raw", "My own jittered frames"]].map(([id, label]) => (
              <label key={id}>
                <input type="radio" name="mode" checked={mode === id} onChange={() => { setMode(id); setError(""); }} disabled={busy} />
                <span>{label}</span>
              </label>
            ))}
          </div>

          <p className="hint">
            {mode === "image"
              ? "The screenshot becomes the ground truth. A simulated renderer sends 320 × 180 jittered frames of it, and the upscaler rebuilds 1280 × 720 with scores. Use 1920 × 1080 or larger."
              : "Frames you rendered yourself at low resolution, each with a known sub-pixel jitter. There is no reference image, so the scene has no scores."}
          </p>

          <div className="field">
            <label htmlFor="scene-name">Scene name</label>
            <input id="scene-name" ref={nameRef} type="text" name="scene-name" autoComplete="off" maxLength={80}
              value={name} onChange={(e) => setName(e.target.value)} disabled={busy} placeholder="Example: Night city street" />
          </div>

          {mode === "image" ? (
            <div className="field">
              <label htmlFor="scene-image">Screenshot (PNG, JPEG or WebP)</label>
              <input id="scene-image" type="file" accept="image/png,image/jpeg,image/webp" disabled={busy}
                onChange={(e) => setImage(e.target.files[0] ?? null)} />
            </div>
          ) : (
            <>
              <div className="field">
                <label htmlFor="scene-frames">Jittered frames, all the same size (taken in name order)</label>
                <input id="scene-frames" type="file" multiple accept="image/png,image/jpeg,image/webp" disabled={busy}
                  onChange={(e) => setFrames([...e.target.files])} />
                {frames.length > 0 && <span className="hint">{frames.length} frames chosen.</span>}
              </div>
              <div className="field">
                <label htmlFor="scene-json">frames.json</label>
                <input id="scene-json" type="file" accept="application/json,.json" disabled={busy}
                  onChange={(e) => setFramesJson(e.target.files[0] ?? null)} />
                <details className="hint">
                  <summary>What goes in frames.json</summary>
                  <pre className="code">{EXAMPLE}</pre>
                  <p>Jitter is the sub-pixel offset of each frame in low-res pixels, between −0.5 and 0.5. Motion is the camera shift since the previous frame in high-res pixels.</p>
                </details>
              </div>
              <label className="check">
                <input type="checkbox" checked={estimate} onChange={(e) => setEstimate(e.target.checked)} disabled={busy} />
                <span>My frames.json has no motion: estimate it (often wrong by a pixel or more)</span>
              </label>
            </>
          )}

          {error && <p className="addscene__error" role="alert">{error}</p>}
          <div className="addscene__actions">
            <button type="submit" className="btn" disabled={busy}>{phase === "uploading" ? "Uploading" : "Import scene"}</button>
            <span className="hint">Takes about 3 to 4 minutes for a 1280 × 720 scene.</span>
          </div>
        </form>
      )}

      {(phase === "running" || phase === "done" || phase === "error") && job && (
        <div className="addscene__status" aria-live="polite">
          <p>
            {phase === "done" ? "Imported. Showing the new scene now." : phase === "error" ? error
              : `Step ${Math.min(step + 1, total)} of ${total}: ${STEPS[Math.min(step, total - 1)]}`}
          </p>
          <progress max={total} value={phase === "done" ? total : step} aria-label="Import progress" />
          <p className="hint"><span className="num">{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</span> elapsed</p>
          {job.log?.length > 0 && <pre className="code addscene__log">{job.log.slice(-6).join("\n")}</pre>}
          {phase === "error" && <button type="button" className="btn" onClick={() => { setPhase("idle"); setJob(null); }}>Try again</button>}
        </div>
      )}
    </section>
  );
}
