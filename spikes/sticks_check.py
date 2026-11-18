"""Exploit check of a soft-sticks run's best creatures (the voxel ``--check`` for sticks).

Re-runs each of the top creatures with a finer mesh, half the time step, and friction x0.8 and x1.2 (set on the
ground, the body and every zone: MuJoCo takes the larger coefficient of a contact), and reports each distance
against the evolved one.

    python spikes/sticks_check.py runs/sticks_long --top 3 --jobs 8 --fine 2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from squishlings.sticks import StickSpace, from_text, place
from squishlings.sticks_evolve import TISSUES
from squishlings.sticks_mesh import scenario


def with_friction(raw: dict, mu: float) -> dict:
    raw = copy.deepcopy(raw)
    raw.setdefault("ground", {}).setdefault("friction", {})["sliding"] = mu

    def card(m):
        ref = dict(m) if isinstance(m, dict) else {"preset": m}
        return {**ref, "friction": {**ref.get("friction", {}), "sliding": mu}}

    raw["body"]["material"] = card(raw["body"]["material"])
    for z in raw["body"].get("zones", []):
        z["material"] = card(z["material"])
    return raw


def variant(job: tuple) -> dict:
    run_dir, key, base_raw, text, label, mesh_size = job
    from softsim.runner import Simulation

    run_dir = Path(run_dir)
    raw = copy.deepcopy(base_raw)
    t0 = time.perf_counter()
    if label == "fine":
        g = from_text(text, StickSpace())
        gtxt, msh = run_dir / "meshes" / f"{key}.txt", run_dir / "meshes" / f"{key}_fine.msh"
        p = subprocess.run([sys.executable, "-m", "squishlings.sticks_mesh", str(gtxt), str(msh), str(mesh_size)],
                           capture_output=True, text=True, timeout=120)
        meshed = json.loads(p.stdout.strip().splitlines()[-1])
        if "tets" not in meshed:
            return {"key": key, "label": label, "error": meshed.get("error")}
        start = raw["actuators"][0]["oscillator"]["start"] if raw["actuators"] else 0.3
        raw = scenario(g, Path("meshes") / msh.name, place(g), set(meshed["tets"]), raw, TISSUES, start=start)
    elif label == "dt/2":
        raw["simulation"]["timestep"] = 0.0005
    elif label.startswith("friction"):
        raw = with_friction(raw, float(label.split("x")[1]))
    try:
        res = Simulation(raw, base_dir=run_dir).run()
        s = res.summary
        out = {"distance": s["distance"], "distance_x": s["distance_x"], "max_tilt": s["max_tilt"],
               "flipped": s["flipped"], "tets": int(len(Simulation(raw, base_dir=run_dir).scene.soft_bodies[0].mesh.tets))
               if label == "fine" else None}
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {"key": key, "label": label, "wall_s": round(time.perf_counter() - t0), **out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--fine", type=float, default=2.0, help="mesh size of the fine variant, mm (evolved at 3)")
    a = ap.parse_args()
    run_dir = Path(a.run).resolve()
    recs = {}
    for line in (run_dir / "evaluations.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[r["key"]] = r
    best = sorted((r for r in recs.values() if "scenario" in r), key=lambda r: -r["fitness"])[: a.top]
    jobs = [(str(run_dir), r["key"], r["scenario"], r["text"], label, a.fine)
            for r in best for label in ("fine", "dt/2", "friction x0.8", "friction x1.2")]
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(variant, jobs))
    report = []
    for r in best:
        row = {"key": r["key"], "evolved_mm": round(1e3 * r["fitness"], 1), "tets": r.get("tets")}
        for v in results:
            if v["key"] == r["key"]:
                if "error" in v:
                    row[v["label"]] = v["error"]
                else:
                    row[v["label"]] = f"{1e3 * v['distance']:.1f} mm ({100 * (v['distance'] / r['fitness'] - 1):+.0f}%)"
                    if v["label"] == "fine":
                        row["fine_tets"] = v["tets"]
        report.append(row)
        print(json.dumps(row), flush=True)
    (run_dir / "check.json").write_text(json.dumps({"report": report, "raw": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
