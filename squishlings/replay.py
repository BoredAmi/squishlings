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
    if label[:1] == "m" and label[1:].rstrip("ef").isdigit():  # soft-stick muscle: segment i, bend halves e/f
        rgb = _MUSCLE_RGB[int(label[1:].rstrip("ef")) % len(_MUSCLE_RGB)]
        return tuple(0.8 * v for v in rgb) if label.endswith("f") else rgb
    return (0.7, 0.7, 0.7)


def _triangle_labels(mesh, volumes: dict[str, np.ndarray], tris: np.ndarray) -> list[str]:
    """Label of the tet each boundary triangle belongs to."""
    tet_label = np.full(mesh.n_tets, "", dtype=object)
    # largest first, so a region covering everything (a mesh_file's ``body``) never hides the zones inside it
    for lab, elements in sorted(volumes.items(), key=lambda kv: -len(kv[1])):
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
            "labels": labels, "stress": stress, "summary": result.summary, "ground": _ground(sim.scene)}


def _ground(scene) -> dict[str, Any]:
    """How to draw the ground: ``rot`` turns world into display coordinates (gravity straight down: a tilted
    gravity is a slope), ``boxes`` the obstacles (centre, half size), ``sand`` a compliant ground."""
    g = np.asarray(scene.simulation.gravity, float)
    a, b = -g / (np.linalg.norm(g) or 1.0), np.array([0.0, 0.0, 1.0])
    v, cos = np.cross(a, b), float(a @ b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + k + k @ k / (1 + cos) if cos > -1 + 1e-9 else np.diag([1.0, -1.0, -1.0])
    ground = scene.ground
    return {"rot": rot, "boxes": np.array([[*o.center, *o.half_size] for o in ground.obstacles]).reshape(-1, 6),
            "sand": ground.compliance is not None}


def on_terrain(raw: dict[str, Any], base_dir: str | Path, terrain: str, params: dict | None = None) -> dict[str, Any]:
    """The creature's scenario on a terrain (:mod:`squishlings.terrain`; its flat run gives heading and footprint)."""
    from squishlings._jobs import run_summary
    from squishlings.terrain import apply, parse

    flat = run_summary((raw, str(base_dir)))
    if "error" in flat:
        raise RuntimeError(f"flat run failed: {flat['error']}")
    return apply(raw, terrain, parse({terrain: params})[terrain], flat)


def record_best(run_dir: str | Path, n: int = 3, overrides: list[str] = (), terrain: str | None = None,
                niche: str | None = None) -> list[dict[str, Any]]:
    """The ``n`` best of a run (of its ``niche``'s archive: a ground's specialists, when given)."""
    report = load_yaml(Path(run_dir) / "report.yaml")
    base_dir = _base_dir(report)  # the creature files' relative paths are the job scenario's
    best = report["best"]
    if niche:
        if niche not in report.get("niches", {}):
            raise ValueError(f"no niche {niche!r} in this run; have {sorted(report.get('niches', {}))}")
        best = report["niches"][niche]["best"]
    out = []
    for rank, b in enumerate(best[:n]):
        raw = load_yaml(Path(run_dir) / b["file"])
        if terrain:
            raw = on_terrain(raw, base_dir, terrain, (report.get("terrains") or {}).get(terrain))
        out.append(record(raw, f"#{rank + 1}  {b['key']}  {report['metric']} {b['fitness']:.4g}"
                          + (f"  on {terrain}" if terrain else ""), base_dir, overrides))
    return out


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
    flat_ground = {"rot": np.eye(3), "boxes": np.zeros((0, 6)), "sand": False}
    n_frames = max(len(r["t"]) for r in records)
    images = []
    for k in range(n_frames):
        fig, axes = plt.subplots(1, len(records), figsize=(3.6 * len(records), 3.6), dpi=80, squeeze=False)
        for ax, r in zip(axes[0], records):
            i = min(k, len(r["t"]) - 1)
            ground = r.get("ground") or flat_ground
            view = rot @ ground["rot"]  # world -> display (gravity down) -> camera
            xw = r["x"][i]
            cw = xw.mean(axis=0)
            g0 = np.floor((cw[:2] - window) / 0.01) * 0.01
            if ground["sand"]:
                quad = np.array([[cw[0] - window, cw[1] - window, 0], [cw[0] + window, cw[1] - window, 0],
                                 [cw[0] + window, cw[1] + window, 0], [cw[0] - window, cw[1] + window, 0]]) @ view.T
                ax.add_collection(PolyCollection([quad[:, :2]], facecolors=[(0.86, 0.76, 0.55)], edgecolors="none"))
            lines = []
            for v in np.arange(g0[0], cw[0] + window, 0.01):
                lines.append([[v, cw[1] - window, 0], [v, cw[1] + window, 0]])
            for v in np.arange(g0[1], cw[1] + window, 0.01):
                lines.append([[cw[0] - window, v, 0], [cw[0] + window, v, 0]])
            lines = np.array(lines) @ view.T
            ax.add_collection(LineCollection(lines[:, :, :2], colors=(0.6, 0.6, 0.6, 0.5), linewidths=0.5))
            tri = xw[r["tris"]]
            p = tri @ view.T
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            shade = 0.35 + 0.65 * np.clip((n @ view.T) @ light, 0, 1)
            base = np.array([label_colour(lab) for lab in r["labels"]])
            for lab, s in r["stress"].items():  # contracting muscle: brighter
                amp = max(np.abs(s).max(), 1e-12)
                mask = np.array([q == lab for q in r["labels"]])
                base[mask] = np.clip(base[mask] * (0.75 + 0.45 * max(s[i], 0) / amp), 0, 1)
            colours = np.clip(base * shade[:, None], 0, 1)
            bq, bc = _box_faces(ground["boxes"], cw, window)
            if len(bq):
                bn = np.cross(bq[:, 1] - bq[:, 0], bq[:, 2] - bq[:, 0])
                bn /= np.linalg.norm(bn, axis=1, keepdims=True) + 1e-30
                bshade = 0.45 + 0.55 * np.clip((bn @ view.T) @ light, 0, 1)
                bp = bq @ view.T
                bkeep = np.cross(bp[:, 1] - bp[:, 0], bp[:, 2] - bp[:, 0])[:, 2] > 0
                polys = list(p) + list(bp[bkeep])
                cols = list(colours) + list(np.clip(bc[bkeep] * bshade[bkeep, None], 0, 1))
                keep_all = np.concatenate([np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0,
                                           np.ones(int(bkeep.sum()), bool)])
                depth = np.array([q[:, 2].mean() for q in polys])
                order = [j for j in np.argsort(depth) if keep_all[j]]
                ax.add_collection(PolyCollection([polys[j][:, :2] for j in order], facecolors=[cols[j] for j in order],
                                                 edgecolors="none"))
            else:
                keep = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0
                order = np.argsort(p[keep][:, :, 2].mean(axis=1))
                ax.add_collection(PolyCollection(p[keep][order][:, :, :2], facecolors=colours[keep][order],
                                                 edgecolors="none"))
            cc = view @ cw
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


def _box_faces(boxes: np.ndarray, centre: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """The obstacle boxes' faces (quads, outward winding) cut to the square around ``centre``, and their colour."""
    quads = []
    for cx, cy, cz, hx, hy, hz in boxes:
        lo = np.array([max(cx - hx, centre[0] - window), max(cy - hy, centre[1] - window), cz - hz])
        hi = np.array([min(cx + hx, centre[0] + window), min(cy + hy, centre[1] + window), cz + hz])
        if (hi[:2] <= lo[:2]).any():
            continue
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        quads += [[[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],   # top
                  [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],   # -y
                  [[x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]],   # +y
                  [[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]],   # -x
                  [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]]   # +x
    q = np.array(quads, float).reshape(-1, 4, 3)
    return q, np.tile([0.55, 0.6, 0.55], (len(q), 1))


# -- HTML ------------------------------------------------------------------------------------------------------------
def _ground_json(ground: dict[str, Any] | None) -> dict[str, Any]:
    if ground is None:
        return {"rot": np.eye(3).ravel().tolist(), "boxes": [], "sand": False}
    return {"rot": [round(float(v), 6) for v in np.ravel(ground["rot"])],
            "boxes": [[round(float(v), 5) for v in b] for b in ground["boxes"]], "sand": bool(ground["sand"])}


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
                        ("distance", "mean_speed", "max_tilt", "turn")},
            "ground": _ground_json(r.get("ground"))})
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
grid.material.transparent = true; grid.material.opacity = 0.35;
const world = new THREE.Group(); world.matrixAutoUpdate = false; scene.add(world); world.add(grid);
const sand = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({color: 0xdcc28c, roughness: 1}));
sand.position.z = -1e-4; world.add(sand);
const ridgeMat = new THREE.MeshStandardMaterial({color: 0x8c9a8c, roughness: 0.9}); let ridges = new THREE.Group(); world.add(ridges);
let mesh = null, cur = null, frame = 0, last = 0, acc = 0;
let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
document.getElementById("play").textContent = playing ? "Pause" : "Play";
const pick = document.getElementById("pick"), slider = document.getElementById("time");
DATA.forEach((c, i) => pick.add(new Option(c.name, i)));
function load(i) {
  cur = DATA[i]; if (mesh) { world.remove(mesh); mesh.geometry.dispose(); }
  const R = cur.ground.rot; world.matrix.set(R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, 0, 0, 0, 1);
  world.matrixWorldNeedsUpdate = true; sand.visible = cur.ground.sand;
  world.remove(ridges); ridges.traverse(o => o.geometry && o.geometry.dispose()); ridges = new THREE.Group(); world.add(ridges);
  for (const [x, y, z, hx, hy, hz] of cur.ground.boxes) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz), ridgeMat); b.position.set(x, y, z); ridges.add(b); }
  const nt = cur.T.length / 3, g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nt * 9), 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nt * 9), 3));
  mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors: true, flatShading: true, roughness: 0.7}));
  world.add(mesh); frame = 0; slider.max = cur.t.length - 1; setFrame(0, true);
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
  const v = new THREE.Vector3(cx, cy, cz).applyMatrix4(world.matrix); const shift = v.clone().sub(controls.target);
  if (recentre) { camera.position.set(v.x + 0.09, v.y - 0.12, v.z + 0.08); controls.target.copy(v); }
  else { controls.target.add(shift); camera.position.add(shift); }
  grid.position.set(Math.round(cx * 100) / 100, Math.round(cy * 100) / 100, 0); sand.position.x = grid.position.x; sand.position.y = grid.position.y;
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
