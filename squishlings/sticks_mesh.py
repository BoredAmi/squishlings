"""Soft sticks → one tet mesh with named regions (gmsh, OpenCASCADE) → a softsim ``locomotion`` scenario.
Prototype.

Each segment is a cylinder with a sphere at both ends (rounded joints). Everything is fused into one solid and
the part below ``zmin + FLAT`` is cut off, so every creature stands on flat feet: round bodies otherwise roll
onto their side (spike, 2026-11-18). The solid is meshed as one volume, and each tet is then labelled by the
segment whose capsule its centroid is deepest in (smallest distance to the axis minus radius):

- ``a``, ``b``...: passive tissue;
- ``m<i>`` (axial muscle) or ``m<i>f`` / ``m<i>e`` (bend muscle: the halves below and above the bend plane):
  segment ``i``'s muscle.

Labelling tets instead of fragmenting the CAD keeps gmsh away from the slivers that overlapping capsules make
(fragmenting failed on 16 of 20 random creatures). The labelled mesh is written as one discrete volume entity
and 3D physical group per region, in millimetres; the scenario loads it with ``scale: 0.001``.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from squishlings.sticks import Placed, StickGenome, place

FLAT = 1.5     # mm trimmed off the lowest point
JOINT = 1.1    # joint sphere radius / segment radius: 1.0 makes tangent seams (meshed 39 of 42 test creatures,
               # 1.1 all 42); main() retries with RETRY_JOINTS if gmsh still fails
RETRY_JOINTS = (1.05, 1.2)


class MeshError(RuntimeError):
    pass


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def label_tets(centroids: np.ndarray, segs: list[Placed]) -> list[str]:
    """Region of each tet centroid (see module docstring)."""
    score = np.empty((len(centroids), len(segs)))
    for j, s in enumerate(segs):
        v = centroids - s.start
        t = np.clip(v @ s.axis, 0.0, s.length)
        score[:, j] = np.linalg.norm(v - t[:, None] * s.axis, axis=1) - s.radius
    owner = np.argmin(score, axis=1)
    labels = []
    for c, j in zip(centroids, owner):
        s = segs[j]
        if not s.muscle:
            labels.append(s.mat)
        elif s.bend:
            labels.append(f"m{s.index}" + ("e" if (c - s.start) @ s.R[:, 2] >= 0 else "f"))
        else:
            labels.append(f"m{s.index}")
    return labels


def build_mesh(g: StickGenome, path: Path, size: float = 3.0) -> dict[str, Any]:
    """Mesh the creature to ``path`` (.msh, mm). Returns the regions' tet counts and the placed segments.
    Raises :class:`MeshError` if gmsh cannot build or mesh it."""
    import gmsh

    segs = place(g)
    zmin = min(min(s.start[2], s.end[2]) - JOINT * s.radius for s in segs)
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        gmsh.model.add("creature")
        solids = []
        for s in segs:
            solids.append(occ.addCylinder(*s.start, *(s.length * s.axis), s.radius))
            solids += [occ.addSphere(*c, JOINT * s.radius) for c in (s.start, s.end)]
        body = [(3, solids[0])]
        if len(solids) > 1:
            body, _ = occ.fuse(body, [(3, t) for t in solids[1:]])
        lo = np.min([np.minimum(s.start, s.end) for s in segs], axis=0) - 50
        hi = np.max([np.maximum(s.start, s.end) for s in segs], axis=0) + 50
        ground = occ.addBox(lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], zmin + FLAT - lo[2])
        body, _ = occ.cut(body, [(3, ground)])
        occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if len(vols) > 1:  # the ground cut can split off a sliver: keep the largest solid
            keep = max(vols, key=lambda v: occ.getMass(*v))
            occ.remove([v for v in vols if v != keep], recursive=True)
            occ.synchronize()
            vols = [keep]
        if not vols:
            raise MeshError("nothing left above the ground cut")
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, _, tet_nodes = gmsh.model.mesh.getElements(3, vols[0][1])
        if not tet_nodes or not len(tet_nodes[0]):
            raise MeshError("gmsh produced no tetrahedra")
        tet_nodes = tet_nodes[0].astype(np.int64).reshape(-1, 4)
        index = np.full(int(tags.max()) + 1, -1)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        pts = xyz.reshape(-1, 3)
        labels = np.array(label_tets(pts[index[tet_nodes]].mean(axis=1), segs))

        gmsh.model.add("labelled")
        regions = sorted(set(labels))
        ents = {r: gmsh.model.addDiscreteEntity(3) for r in regions}
        gmsh.model.mesh.addNodes(3, ents[regions[0]], tags, xyz)
        tets = {}
        for r in regions:
            sel = tet_nodes[labels == r]
            gmsh.model.mesh.addElementsByType(ents[r], 4, [], sel.ravel())
            gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [ents[r]]), r)
            tets[r] = int(len(sel))
        path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(path))
    except MeshError:
        raise
    except Exception as exc:  # gmsh raises bare Exceptions
        raise MeshError(str(exc)) from None
    finally:
        gmsh.finalize()
    return {"tets": tets, "segments": segs}


def scenario(g: StickGenome, msh: Path, segs: list[Placed], regions: set[str], base: dict[str, Any],
             tissues: dict[str, Any], start: float) -> dict[str, Any]:
    """The locomotion scenario of a meshed creature: ``base`` with the creature's body, zones and muscles."""
    raw = copy.deepcopy(base)
    zones, actuators = [], []
    for lab, card in tissues.items():
        if lab != "a" and lab in regions:
            zones.append({"name": f"tissue_{lab}", "material": card, "region": lab})
    osc = {"frequency": g.frequency, "amplitude": g.amplitude, "start": start}
    for s in segs:
        if not s.muscle:
            continue
        phase = g.phases[s.muscle - 1]
        parts = [(f"m{s.index}f", phase), (f"m{s.index}e", phase + math.pi)] if s.bend else [(f"m{s.index}", phase)]
        for region, ph in parts:
            if region not in regions:
                continue
            if s.mat != "a":
                zones.append({"name": f"tissue_{region}", "material": tissues[s.mat], "region": region})
            actuators.append({"name": region, "type": "muscle", "region": region,
                              "direction": [float(v) for v in s.axis],
                              "oscillator": {**osc, "phase": float(ph % (2 * math.pi))}})
    raw["body"] = {**raw.get("body", {}), "name": "creature", "material": tissues["a"],
                   "geometry": {"type": "mesh_file", "path": str(msh), "scale": 0.001, "center": "base"},
                   "zones": zones}
    raw["actuators"] = actuators
    return raw


def space_from_dict(d: dict) -> "StickSpace":
    from squishlings.sticks import StickSpace

    return StickSpace(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def run_sticks(task: dict[str, Any]) -> dict[str, Any]:
    """Worker of ``squishlings evolve`` for ``encoding: sticks``: mesh the creature in a child process (killed
    after ``task["timeout"]`` s: gmsh can hang on awkward overlaps; a mesh already on disk is reused), then
    simulate it. Returns the scenario summary plus ``tets``, or ``{"error": ...}``."""
    import json
    import subprocess
    import sys

    from squishlings._jobs import run_summary
    from squishlings.sticks import from_text

    g = from_text(task["text"], space_from_dict(task["space"]))
    msh = Path(task["mesh"])
    msh.parent.mkdir(parents=True, exist_ok=True)
    regions = msh.with_suffix(".regions.json")
    if not (regions.is_file() and msh.is_file()):
        txt = msh.with_suffix(".txt")
        txt.write_text(g.text)
        try:
            p = subprocess.run([sys.executable, "-m", "squishlings.sticks_mesh", str(txt), str(msh),
                                str(task["mesh_size"]), "--space", json.dumps(task["space"])],
                               capture_output=True, text=True, timeout=task["timeout"])
        except subprocess.TimeoutExpired:
            return {"error": f"mesh: gmsh timed out after {task['timeout']:g} s"}
        lines = p.stdout.strip().splitlines()
        out = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {"error": p.stderr[-300:]}
        if "tets" not in out:
            return {"error": f"mesh: {out.get('error', 'unknown')}"}
        regions.write_text(json.dumps(out["tets"]))
    tets = json.loads(regions.read_text())
    raw = scenario(g, msh, place(g), set(tets), task["base"], task["tissues"], task["start"])
    raw["name"] = task["name"]
    summary = run_summary((raw, task["base_dir"]))
    if "error" not in summary:
        summary["tets"] = float(sum(tets.values()))
    return summary


def main(argv: list[str] | None = None) -> int:
    """``python -m squishlings.sticks_mesh genome.txt out.msh [size] [--space JSON]``: mesh in a separate
    process, so the caller can kill a gmsh that hangs. Prints the regions' tet counts as JSON."""
    import argparse
    import json

    from squishlings.sticks import StickSpace, from_text

    ap = argparse.ArgumentParser()
    ap.add_argument("genome")
    ap.add_argument("msh")
    ap.add_argument("size", type=float, nargs="?", default=3.0)
    ap.add_argument("--space", help="the StickSpace as JSON (default: StickSpace())")
    a = ap.parse_args(argv)
    space = space_from_dict(json.loads(a.space)) if a.space else StickSpace()
    g = from_text(Path(a.genome).read_text(), space)
    global JOINT
    errors = []
    for joint in (JOINT, *RETRY_JOINTS):
        JOINT = joint
        try:
            out = build_mesh(g, Path(a.msh), a.size)
        except MeshError as exc:
            errors.append(f"joint {joint}: {exc}")
            continue
        print(json.dumps({"tets": out["tets"], "joint": joint}))
        return 0
    print(json.dumps({"error": "; ".join(errors)}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
