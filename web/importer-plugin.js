// Dev-server endpoints that let the page add scenes: the browser uploads the
// files, this saves them under uploads/ and runs scripts/import_scene.py.
// Only active under `npm run dev` (Vite binds to localhost by default).
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const PYTHON = existsSync(join(ROOT, ".venv/bin/python")) ? join(ROOT, ".venv/bin/python") : "python3";
const MAX_BODY = 200 * 1024 * 1024;
const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

const jobs = new Map();
let running = null;

const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
const safeName = (s) => basename(String(s)).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 96);

function send(res, code, body) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

function readJson(req) {
  return new Promise((ok, fail) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) { fail(new Error("Upload is over 200 MB.")); req.destroy(); return; }
      chunks.push(c);
    });
    req.on("end", () => {
      try { ok(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch { fail(new Error("The upload was not valid JSON.")); }
    });
    req.on("error", fail);
  });
}

function startJob({ mode, name, files, framesJson, estimateMotion }) {
  const id = slug(name);
  if (!id) throw new Error("Give the scene a name with at least one letter or digit.");
  if (mode !== "image" && mode !== "raw") throw new Error("Unknown mode.");
  if (!Array.isArray(files) || files.length === 0) throw new Error("Choose at least one file.");
  for (const f of files) if (!IMAGE_EXT.test(f.name || "")) throw new Error(`${f.name} is not a PNG, JPEG or WebP image.`);
  if (mode === "image" && files.length !== 1) throw new Error("Game image mode takes exactly one screenshot.");

  let meta = null;
  if (mode === "raw") {
    if (!framesJson) throw new Error("Jittered frames need a frames.json with the jitter of each frame.");
    try { meta = JSON.parse(framesJson); } catch { throw new Error("frames.json is not valid JSON."); }
    if (!Array.isArray(meta.jitter) || meta.jitter.length !== files.length) {
      throw new Error(`frames.json lists ${meta.jitter?.length ?? 0} jitter entries for ${files.length} frames.`);
    }
    if (!meta.motion && !estimateMotion) {
      throw new Error("frames.json has no motion. Add the camera shift per frame, or tick the estimate option.");
    }
  }

  // everything is validated; only now touch the disk
  const dir = join(ROOT, "uploads", mode === "raw" ? `raw-${id}` : id);
  mkdirSync(dir, { recursive: true });
  let source;
  if (mode === "image") {
    source = join(dir, safeName(files[0].name));
    writeFileSync(source, Buffer.from(files[0].data, "base64"));
  } else {
    // number the frames so their order is the order they were chosen in
    files.forEach((f, k) => writeFileSync(join(dir, `${String(k).padStart(4, "0")}_${safeName(f.name)}`), Buffer.from(f.data, "base64")));
    writeFileSync(join(dir, "frames.json"), JSON.stringify(meta));
    source = dir;
  }

  const args = ["-u", join(ROOT, "scripts/import_scene.py"), mode, source, "--name", String(name).slice(0, 80), "--id", id];
  if (mode === "raw" && estimateMotion) args.push("--estimate-motion");
  const job = { id: `${id}-${Date.now()}`, status: "running", log: [], scene: null, started: Date.now(), progress: [0, 5] };
  const child = spawn(PYTHON, args, { cwd: ROOT });
  const onLine = (buf) => {
    for (const line of buf.toString().split("\n").filter(Boolean)) {
      const p = /^progress (\d+)\/(\d+)/.exec(line);
      const s = /^scene (\S+)/.exec(line);
      if (p) job.progress = [Number(p[1]), Number(p[2])];
      else if (s) job.scene = s[1];
      else job.log.push(line);
      if (job.log.length > 200) job.log.shift();
    }
  };
  child.stdout.on("data", onLine);
  child.stderr.on("data", onLine);
  child.on("close", (code) => {
    job.status = code === 0 && job.scene ? "done" : "error";
    job.finished = Date.now();
    running = null;
  });
  jobs.set(job.id, job);
  running = job.id;
  return job;
}

export default function importer() {
  return {
    name: "scene-importer",
    configureServer(server) {
      server.middlewares.use("/api/health", (req, res) => send(res, 200, { ok: true }));
      server.middlewares.use("/api/import", async (req, res) => {
        if (req.method === "GET") {
          const job = jobs.get(decodeURIComponent(req.url.replace(/^\//, "")));
          return job ? send(res, 200, job) : send(res, 404, { error: "No such import." });
        }
        if (req.method !== "POST") return send(res, 405, { error: "Use POST." });
        if (running) return send(res, 409, { error: "Another import is still running. Wait for it to finish." });
        try {
          const job = startJob(await readJson(req));
          return send(res, 202, { job: job.id });
        } catch (e) {
          return send(res, 400, { error: e.message });
        }
      });
    },
  };
}
