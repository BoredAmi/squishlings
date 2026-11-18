"""Spike for the soft-sticks encoding (docs/design/soft-sticks.md, "Check in softsim first").

Two fused capsules on the ground: a passive torso (Ecoflex 00-30) and a bending muscle segment split lengthwise
into a bottom (flexor) and a top (extensor) half, driven in antiphase. gmsh's OpenCASCADE kernel builds the
cylinders and joint spheres, BooleanFragments fuses them into one conformal solid, and every piece goes to a 3D
physical group: ``a`` (passive tissue), ``m1_flex``, ``m1_ext``. softsim loads the .msh as a ``mesh_file`` body
and the locomotion scenario runs it on MuJoCo.

Checks:
  1. the .msh loads, and its physical groups work as zone and muscle regions;
  2. the muscle segment bends (tip height oscillates) and does nothing with the muscles off;
  3. cost: tets and wall time against ~15 s for a 2^3 voxel creature (3 s of simulation);
  4. convergence: tip excursion and distance at element size r/2 and r/4.

    python spikes/sticks_spike.py            # outputs in runs/spike_sticks/
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from softsim.runner import Simulation

OUT = Path(__file__).resolve().parent.parent / "runs" / "spike_sticks"
MM = 1e-3


@dataclass
class Stick:
    name: str
    start: tuple[float, float, float]   # m
    end: tuple[float, float, float]
    radius: float                       # m
    tissue: str = "a"
    bend: bool = False                  # split along a horizontal plane through the axis (bends in the vertical plane)


def build_msh(sticks: list[Stick], size: float, path: Path) -> dict[str, int]:
    """Fuse the sticks (cylinders + spheres at both ends) into one conformal tet mesh with named volume groups."""
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        gmsh.model.add("sticks")
        owners: list[tuple[int, str]] = []          # (volume tag, region), in priority order
        for s in sticks:
            p, q = np.array(s.start), np.array(s.end)
            axis = q - p
            cyl = occ.addCylinder(*p, *axis, s.radius)
            if s.bend:
                # half-space box above the axis plane: its intersection is the top (extensor) half
                L = float(np.linalg.norm(axis)) + 4 * s.radius
                box = occ.addBox(p[0] - 2 * s.radius - L, p[1] - L, p[2], 3 * L, 2 * L, 2 * s.radius + L)
                cyl2 = occ.copy([(3, cyl)])[0][1]
                top, _ = occ.intersect([(3, cyl2)], [(3, box)], removeObject=True, removeTool=False)
                bottom, _ = occ.cut([(3, cyl)], [(3, box)], removeObject=True, removeTool=True)
                owners += [(t, f"{s.name}_flex") for _, t in bottom] + [(t, f"{s.name}_ext") for _, t in top]
            else:
                owners.append((cyl, s.tissue))
        for s in sticks:                            # rounded joints and ends: passive, the stick's own tissue
            for c in (s.start, s.end):
                owners.append((occ.addSphere(*c, s.radius), s.tissue))
        occ.synchronize()
        _, children = occ.fragment([(3, t) for t, _ in owners], [])
        occ.synchronize()
        region_of: dict[int, str] = {}
        for (tag, region), kids in zip(owners, children):  # first owner in priority order claims a piece
            for _, k in kids:
                region_of.setdefault(k, region)
        groups: dict[str, list[int]] = {}
        for k, region in region_of.items():
            groups.setdefault(region, []).append(k)
        for region, tags in groups.items():
            gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, tags), region)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 2)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
        n_tets = {r: sum(len(gmsh.model.mesh.getElements(3, t)[1][0]) for t in tags) for r, tags in groups.items()}
    finally:
        gmsh.finalize()
    return n_tets


def creature(r: float) -> list[Stick]:
    """Torso 30 mm (radius r), then a bending segment 25 mm (radius 0.8 r), lying along +x."""
    z = r
    return [Stick("torso", (0.0, 0.0, z), (30 * MM, 0.0, z), r),
            Stick("m1", (30 * MM, 0.0, z), (55 * MM, 0.0, z), 0.8 * r, bend=True)]


def scenario(msh: Path, amplitude: float, frequency: float = 4.0, duration: float = 2.0) -> dict:
    osc = {"frequency": frequency, "amplitude": amplitude, "start": 0.3}
    return {
        "scenario": "locomotion", "backend": "mujoco", "settle": 0.3,
        "simulation": {"timestep": 0.001, "duration": duration, "output_rate": 100},
        "ground": {"friction": {"sliding": 1.0}},
        "body": {"name": "creature", "material": "ecoflex_00_30",
                 "geometry": {"type": "mesh_file", "path": str(msh), "center": "base"},
                 "zones": [{"name": "muscle_flex", "material": "ecoflex_00_30", "region": "m1_flex"}]},
        "actuators": [
            {"name": "m1_flex", "type": "muscle", "region": "m1_flex", "direction": [1.0, 0.0, 0.0],
             "oscillator": {**osc, "phase": 0.0}},
            {"name": "m1_ext", "type": "muscle", "region": "m1_ext", "direction": [1.0, 0.0, 0.0],
             "oscillator": {**osc, "phase": math.pi}},
        ],
    }


def run(msh: Path, amplitude: float) -> dict:
    sim = Simulation(scenario(msh, amplitude))
    rest = sim.scene.soft_bodies[0].mesh.points
    tip = int(np.argmax(rest[:, 0]))                       # the far end of the muscle segment
    root = int(np.argmin(rest[:, 0]))
    trace = []

    def record(s: Simulation):
        x = s.state().soft["creature"].vertices
        trace.append((s.time, x[tip, 2] - x[root, 2], x[tip, 0] - x[root, 0]))

    t0 = time.perf_counter()
    res = sim.run(controller=record)
    wall = time.perf_counter() - t0
    tr = np.array(trace)
    gait = tr[tr[:, 0] >= 0.3 + 0.25]                       # after settling and the one-period ramp
    return {"amplitude_kPa": amplitude / 1e3, "wall_s": round(wall, 2),
            "tets": int(len(sim.scene.soft_bodies[0].mesh.tets)),
            "points": int(len(rest)),
            "tip_rel_z_range_mm": round(1e3 * float(np.ptp(gait[:, 1])), 3),
            "length_range_mm": round(1e3 * float(np.ptp(gait[:, 2])), 3),
            "distance_mm": round(1e3 * res.summary["distance"], 2),
            "distance_x_mm": round(1e3 * res.summary["distance_x"], 2),
            "distance_y_mm": round(1e3 * res.summary["distance_y"], 2),
            "max_tilt_deg": round(math.degrees(res.summary["max_tilt"]), 2),
            "findings": [str(f) for f in sim.findings]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # python sticks_spike.py [kPa,kPa,... at r/2] [kPa,... at r/4]
    amps_coarse = [float(a) * 1e3 for a in (sys.argv[1] if len(sys.argv) > 1 else "0,8").split(",") if a]
    amps_fine = [float(a) * 1e3 for a in (sys.argv[2] if len(sys.argv) > 2 else "8").split(",") if a]
    r = 5 * MM
    results = {}
    for label, size, amps in (("r/2", r / 2, amps_coarse), ("r/4", r / 4, amps_fine)):
        if not amps:
            continue
        msh = OUT / f"two_sticks_{label.replace('/', '_')}.msh"
        t0 = time.perf_counter()
        tets = build_msh(creature(r), size, msh)
        mesh_s = time.perf_counter() - t0
        print(f"[{label}] meshed in {mesh_s:.2f} s, tets per region {tets}", flush=True)
        for amp in amps:
            out = run(msh, amp)
            out["mesh_s"] = round(mesh_s, 2)
            results[f"{label} {amp / 1e3:g} kPa"] = out
            print(f"[{label}, {amp / 1e3:g} kPa] {json.dumps({k: v for k, v in out.items() if k != 'findings'})}",
                  flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
