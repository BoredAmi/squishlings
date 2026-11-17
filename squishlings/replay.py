"""Watch evolved creatures: record their runs, then a GIF (matplotlib) or a self-contained HTML player (three.js).

    squishlings replay runs/crawler --best 3 --gif crawler.gif --html crawler.html

A recording is the creature scenario run with vertex output, cut to its boundary triangles, each triangle coloured
by the label of the voxel it bounds (tissues in greys/blues, muscle groups in warm colours). The HTML page tints a
muscle face by its stress at that instant (brighter = contracting) and needs no server: open it in a browser.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
from softsim.config.loader import load_yaml

from squishlings.build import MUSCLE_CHARS, TISSUE_CHARS

FRAME_DT = 1.0 / 30.0
_TISSUE_RGB = [(0.80, 0.80, 0.74), (0.36, 0.55, 0.80), (0.55, 0.70, 0.45), (0.70, 0.62, 0.80),
               (0.85, 0.75, 0.50), (0.50, 0.75, 0.75), (0.75, 0.55, 0.55), (0.60, 0.60, 0.60)]
_MUSCLE_RGB = [(0.90, 0.30, 0.25), (0.95, 0.55, 0.15), (0.85, 0.25, 0.55), (0.75, 0.20, 0.20),
               (0.95, 0.40, 0.40), (0.80, 0.45, 0.10), (0.90, 0.20, 0.40), (0.65, 0.30, 0.15), (1.00, 0.60, 0.40)]


def label_colour(label: str) -> tuple[float, float, float]:
    if label in TISSUE_CHARS:
        return _TISSUE_RGB[TISSUE_CHARS.index(label)]
    if label in MUSCLE_CHARS:
        return _MUSCLE_RGB[MUSCLE_CHARS.index(label)]
    return (0.7, 0.7, 0.7)


def _triangle_labels(mesh, volumes: dict[str, np.ndarray], tris: np.ndarray) -> list[str]:
    """Label of the tet each boundary triangle belongs to."""
    tet_label = np.full(mesh.n_tets, "", dtype=object)
    for lab, elements in volumes.items():
        tet_label[np.asarray(elements)] = lab
    owner = {}
    for e, tet in enumerate(mesh.tets):
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            owner.setdefault(tuple(sorted(tet[list(f)])), e)
    return [tet_label[owner[tuple(sorted(t))]] for t in tris]


def record(config: str | Path | dict[str, Any], name: str = "", base_dir: str | Path | None = None,
           overrides: list[str] = ()) -> dict[str, Any]:
    """Run a creature scenario and keep what the players need (positions on the output samples)."""
    from softsim.runner import Simulation

    sim = Simulation(config, ["output.save_model=false", *overrides], base_dir=base_dir)
    try:
        result = sim.run(record_vertices=True)
    finally:
        sim.close()
    ts = result.timeseries
    body = sim.scene.soft_bodies[0]
    geo = sim.scenario.config.body.geometry
    tris = body.mesh.surface_triangles()
    labels = _triangle_labels(body.mesh, geo.volume_regions(body.mesh) if hasattr(geo, "volume_regions") else {}, tris)
    t = ts["t"].to_numpy()
    frames = np.arange(t[0], t[-1] + 1e-9, FRAME_DT)
    idx = np.array([int(np.argmin(np.abs(t - f))) for f in frames])
    stress = {lab: ts[f"muscle_{lab}_stress"].to_numpy()[idx] for lab in set(labels)
              if f"muscle_{lab}_stress" in ts}
    return {"name": name or sim.config.name, "t": t[idx], "x": result.vertices[body.name][idx], "tris": tris,
            "labels": labels, "stress": stress, "summary": result.summary}


def record_best(run_dir: str | Path, n: int = 3, overrides: list[str] = ()) -> list[dict[str, Any]]:
    report = load_yaml(Path(run_dir) / "report.yaml")
    base_dir = _base_dir(report)  # the creature files' relative paths are the job scenario's
    return [record(load_yaml(Path(run_dir) / b["file"]), f"#{rank + 1}  {b['key']}  {report['metric']} "
                   f"{b['fitness']:.4g}", base_dir, overrides) for rank, b in enumerate(report["best"][:n])]


def _base_dir(report: dict) -> Path:
    job = Path(report["job"])
    return (job.parent / load_yaml(job)["scenario"]).resolve().parent


# -- GIF -------------------------------------------------------------------------------------------------------------
def gif(records: list[dict[str, Any]], path: str | Path, elev: float = 25.0, azim: float = -60.0,
        window: float | None = None) -> Path:
    """Side-by-side animation, the camera following each creature's centre, a ground grid for scale (1 cm)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from PIL import Image

    e, a = np.radians(elev), np.radians(azim)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    rot = np.array([right, np.cross(toward, right), toward])
    light = np.array([-0.4, 0.5, 0.8]) / np.linalg.norm([-0.4, 0.5, 0.8])
    if window is None:
        window = max(float(np.ptp(r["x"][0][:, :2], axis=0).max()) for r in records) * 0.9
    n_frames = max(len(r["t"]) for r in records)
    images = []
    for k in range(n_frames):
        fig, axes = plt.subplots(1, len(records), figsize=(3.6 * len(records), 3.6), dpi=80, squeeze=False)
        for ax, r in zip(axes[0], records):
            i = min(k, len(r["t"]) - 1)
            x = r["x"][i]
            c = x.mean(axis=0)
            g0 = np.floor((c[:2] - window) / 0.01) * 0.01
            lines = []
            for v in np.arange(g0[0], c[0] + window, 0.01):
                lines.append([[v, c[1] - window, 0], [v, c[1] + window, 0]])
            for v in np.arange(g0[1], c[1] + window, 0.01):
                lines.append([[c[0] - window, v, 0], [c[0] + window, v, 0]])
            lines = np.array(lines) @ rot.T
            ax.add_collection(LineCollection(lines[:, :, :2], colors=(0.6, 0.6, 0.6, 0.5), linewidths=0.5))
            tri = x[r["tris"]]
            p = tri @ rot.T
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            shade = 0.35 + 0.65 * np.clip((n @ rot.T) @ light, 0, 1)
            base = np.array([label_colour(lab) for lab in r["labels"]])
            for lab, s in r["stress"].items():  # contracting muscle: brighter
                amp = max(np.abs(s).max(), 1e-12)
                mask = np.array([q == lab for q in r["labels"]])
                base[mask] = np.clip(base[mask] * (0.75 + 0.45 * max(s[i], 0) / amp), 0, 1)
            colours = np.clip(base * shade[:, None], 0, 1)
            keep = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0
            order = np.argsort(p[keep][:, :, 2].mean(axis=1))
            ax.add_collection(PolyCollection(p[keep][order][:, :, :2], facecolors=colours[keep][order],
                                             edgecolors="none"))
            cc = rot @ c
            ax.set_xlim(cc[0] - window, cc[0] + window)
            ax.set_ylim(cc[1] - 0.7 * window, cc[1] + 0.7 * window)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(f"{r['name']}\nt = {r['t'][i]:.2f} s", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=96))
        plt.close(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 * FRAME_DT), loop=0,
                   optimize=True)
    return path


# -- HTML ------------------------------------------------------------------------------------------------------------
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def html(records: list[dict[str, Any]], path: str | Path, title: str = "Soft creatures", fragment: bool = False) -> Path:
    """A self-contained page (three.js from a CDN) that plays the recordings; positions as float32, base64."""
    creatures = []
    for r in records:
        x = r["x"].astype(np.float32)
        labels = sorted(set(r["labels"]))
        creatures.append({
            "name": r["name"], "t": [round(float(v), 4) for v in r["t"]], "n_points": int(x.shape[1]),
            "x": _b64(x), "tris": _b64(r["tris"].astype(np.uint32)),
            "label_index": _b64(np.array([labels.index(q) for q in r["labels"]], dtype=np.uint8)),
            "labels": labels, "colours": [label_colour(q) for q in labels],
            "stress": {q: [round(float(v), 2) for v in s] for q, s in r["stress"].items()},
            "summary": {k: float(v) for k, v in r["summary"].items() if k in
                        ("distance", "mean_speed", "max_tilt", "turn")}})
    data = json.dumps(creatures)
    page = _HTML.replace("__TITLE__", title).replace("__DATA__", data)
    if fragment:  # the page's content only, for hosts that wrap it in their own document (no doctype/html/head/body)
        for tag in ("<!doctype html>\n", '<html lang="en"><head>', '<meta charset="utf-8"><meta name="viewport" '
                    'content="width=device-width, initial-scale=1">\n', "</head><body>\n", "</body></html>\n"):
            page = page.replace(tag, "")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: full-bleed 3D stage, one floating control strip at the bottom, legend top right */
:root{--bg:#e9eef1;--fg:#16212a;--muted:#5b6872;--panel:#f7fafbd9;--accent:#b8322a;--line:#c9d3d9;--grid:#8a99a3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121619;--fg:#e2e8ec;--muted:#93a1ab;--panel:#1c2226d9;--accent:#e0675c;--line:#2e383e;--grid:#56636b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121619;--fg:#e2e8ec;--muted:#93a1ab;--panel:#1c2226d9;--accent:#e0675c;--line:#2e383e;--grid:#56636b;color-scheme:dark}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;overflow:hidden}
#view{position:fixed;inset:0}
#hud{position:fixed;left:max(16px,env(safe-area-inset-left,0px));right:max(16px,env(safe-area-inset-right,0px));bottom:calc(16px + env(safe-area-inset-bottom,0px));max-width:720px;margin:0 auto;background:var(--panel);
backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:grid;gap:8px}
#row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);border-radius:8px;padding:4px 10px}
button{cursor:pointer;min-width:64px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{flex:1;min-width:120px;accent-color:var(--accent)}
#stats{color:var(--muted);font-variant-numeric:tabular-nums}
h1{position:fixed;left:max(16px,env(safe-area-inset-left,0px));top:calc(12px + env(safe-area-inset-top,0px));margin:0;font-size:16px;font-weight:600}
#legend{position:fixed;right:max(16px,env(safe-area-inset-right,0px));top:calc(40px + env(safe-area-inset-top,0px));display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;max-width:60vw;color:var(--muted);font-size:12px}
#legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;background:var(--c);vertical-align:-1px}
</style></head><body>
<div id="view"></div><h1>__TITLE__</h1><div id="legend"></div>
<div id="hud"><div id="row"><select id="pick" aria-label="creature"></select><button id="play">Pause</button>
<input id="time" type="range" min="0" max="1" step="1" value="0" aria-label="time"><span id="clock"></span></div>
<div id="stats"></div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const DATA = __DATA__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
for (const c of DATA) { c.X = dec(c.x, Float32Array); c.T = dec(c.tris, Uint32Array); c.L = dec(c.label_index, Uint8Array); }
const css = getComputedStyle(document.documentElement);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio); document.getElementById("view").appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(css.getPropertyValue("--bg").trim());
const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 10); camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-0.3, -0.5, 1); scene.add(sun);
const gridColour = new THREE.Color(css.getPropertyValue("--grid").trim());
const grid = new THREE.GridHelper(1, 100, gridColour, gridColour); grid.rotation.x = Math.PI / 2;
grid.material.transparent = true; grid.material.opacity = 0.35; scene.add(grid);
let mesh = null, cur = null, frame = 0, last = 0, acc = 0;
let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
document.getElementById("play").textContent = playing ? "Pause" : "Play";
const pick = document.getElementById("pick"), slider = document.getElementById("time");
DATA.forEach((c, i) => pick.add(new Option(c.name, i)));
function load(i) {
  cur = DATA[i]; if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
  const nt = cur.T.length / 3, g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nt * 9), 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nt * 9), 3));
  mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors: true, flatShading: true, roughness: 0.7}));
  scene.add(mesh); frame = 0; slider.max = cur.t.length - 1; setFrame(0, true);
  const s = cur.summary, f = v => v === undefined ? "–" : v;
  document.getElementById("stats").textContent =
    `distance ${(1000 * f(s.distance)).toFixed(1)} mm · speed ${(1000 * f(s.mean_speed)).toFixed(1)} mm/s · max tilt ${(f(s.max_tilt) * 57.2958).toFixed(0)}° · turn ${(f(s.turn) * 57.2958).toFixed(0)}°`;
  const lg = document.getElementById("legend"); lg.innerHTML = "";
  cur.labels.forEach((l, k) => { const e = document.createElement("span"); const c = cur.colours[k];
    e.style.setProperty("--c", `rgb(${c.map(v => 255 * v).join(",")})`); e.textContent = /[0-9]/.test(l) ? `muscle ${l}` : `tissue ${l}`; lg.appendChild(e); });
}
function setFrame(k, recentre) {
  frame = k; slider.value = k; const n = cur.n_points, X = cur.X, T = cur.T, off = k * n * 3;
  const P = mesh.geometry.attributes.position.array, C = mesh.geometry.attributes.color.array;
  const amp = {}; for (const [l, s] of Object.entries(cur.stress)) amp[l] = Math.max(1e-9, ...s.map(Math.abs));
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += X[off + 3 * i]; cy += X[off + 3 * i + 1]; cz += X[off + 3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;
  for (let t = 0; t < T.length / 3; t++) {
    const l = cur.L[t], lab = cur.labels[l], col = cur.colours[l];
    const s = cur.stress[lab] ? Math.max(0, cur.stress[lab][k]) / amp[lab] : 0, gain = cur.stress[lab] ? 0.7 + 0.5 * s : 1;
    for (let v = 0; v < 3; v++) { const p = T[3 * t + v];
      for (let d = 0; d < 3; d++) { P[9 * t + 3 * v + d] = X[off + 3 * p + d]; C[9 * t + 3 * v + d] = Math.min(1, col[d] * gain); } }
  }
  mesh.geometry.attributes.position.needsUpdate = true; mesh.geometry.attributes.color.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
  const shift = new THREE.Vector3(cx, cy, cz).sub(controls.target);
  if (recentre) { camera.position.set(cx + 0.09, cy - 0.12, cz + 0.08); controls.target.set(cx, cy, cz); }
  else { controls.target.add(shift); camera.position.add(shift); }
  grid.position.set(Math.round(cx * 100) / 100, Math.round(cy * 100) / 100, 0);
  document.getElementById("clock").textContent = `${cur.t[k].toFixed(2)} s`;
}
pick.onchange = () => load(+pick.value);
slider.oninput = () => setFrame(+slider.value, false);
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
function resize() { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize(); load(0);
renderer.setAnimationLoop(now => {
  const dt = cur.t.length > 1 ? cur.t[1] - cur.t[0] : 0.033;
  if (playing) { acc += (now - last) / 1000; while (acc >= dt) { acc -= dt; setFrame((frame + 1) % cur.t.length, frame + 1 >= cur.t.length); } }
  last = now; controls.update(); renderer.render(scene, camera);
});
</script></body></html>
"""
