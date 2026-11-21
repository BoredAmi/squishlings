"""Terrains: score a creature on harder ground than the flat plane it was built on (README, "Terrain").

A creature scenario (a ``locomotion`` softsim config) is run on flat ground first, then on each terrain:

- ``slope``: the ground tilted by ``angle`` (gravity tilted, the ground stays at z = 0) so that uphill is the
  creature's own heading on flat ground (its net displacement there; +x if it barely moved). Score: the
  displacement uphill, negative when it slides down.
- ``ridges``: a square lattice of low static boxes (``height``, ``width``, every ``spacing``) out to ``extent``
  from the start, both ways in x and y, with a square around the creature's rest footprint left clear (a body
  can't start inside a ridge). Score: ``distance``.
- ``sand``: a soft, damped ground (softsim ``ground.compliance``: contact ``timeconst``, ``dampratio``,
  ``solimp``): the body sinks a few mm and loses energy at every push. A continuum, not grains: no displaced
  material, no footprints. Score: ``distance``.

:func:`evaluate` returns the flat summary with, per terrain, ``<terrain>/distance``, ``/final_max_z``,
``/flipped`` (or ``/error``), plus ``terrain_mean`` and ``terrain_min`` over flat and every terrain (-inf if any run
failed). A job uses them as its fitness (``fitness: {metric: terrain_mean}``).
"""

from __future__ import annotations

import copy
import math
from typing import Any

from softsim.config.units import parse_quantity

from squishlings._jobs import run_summary

DEFAULTS: dict[str, dict[str, Any]] = {
    "slope": {"angle": "8 deg"},
    "ridges": {"height": "3 mm", "width": "6 mm", "spacing": "40 mm", "extent": "1.5 m"},
    "sand": {"timeconst": "20 ms", "dampratio": 1.0, "solimp": [0.5, 0.95, 0.005]},
}
_DIMENSIONS = {"angle": "angle", "height": "length", "width": "length", "spacing": "length", "extent": "length",
               "timeconst": "time"}
_KEEP = ("distance", "distance_x", "distance_y", "final_max_z", "flipped")


class TerrainError(ValueError):
    pass


def parse(spec: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{name: params or None}`` -> ``{name: SI params}``, defaults filled in, in the order given."""
    out: dict[str, dict[str, Any]] = {}
    for name, params in (spec or {}).items():
        if name not in DEFAULTS:
            raise TerrainError(f"terrains: unknown terrain {name!r}; known: {', '.join(DEFAULTS)}")
        params = dict(params or {})
        unknown = set(params) - set(DEFAULTS[name])
        if unknown:
            raise TerrainError(f"terrains.{name}: unknown keys {sorted(unknown)}; known: {sorted(DEFAULTS[name])}")
        merged = {**DEFAULTS[name], **params}
        si: dict[str, Any] = {}
        for k, v in merged.items():
            if k == "solimp":
                si[k] = [float(x) for x in v]
                if len(si[k]) != 3:
                    raise TerrainError(f"terrains.{name}.solimp: three numbers")
            else:
                si[k] = float(parse_quantity(v, _DIMENSIONS.get(k, "dimensionless")) if k in _DIMENSIONS else v)
                if si[k] <= 0:
                    raise TerrainError(f"terrains.{name}.{k}: must be positive, got {v!r}")
        out[name] = si
    return out


def describe(terrains: dict[str, dict[str, Any]]) -> str:
    """One line for people (and the LLM prompt): what the creature is scored on."""
    parts = ["flat ground"]
    for name, p in terrains.items():
        if name == "slope":
            parts.append(f"a slope of {math.degrees(p['angle']):.0f} degrees, climbed uphill in its own direction")
        elif name == "ridges":
            parts.append(f"ground with {1e3 * p['height']:.0f} mm high ridges every {1e3 * p['spacing']:.0f} mm "
                         f"in both directions")
        elif name == "sand":
            parts.append("soft sand-like ground that the body sinks into")
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def heading(flat: dict[str, float]) -> tuple[float, float]:
    """Unit vector of the flat-ground displacement; +x when the creature moved less than 1 mm."""
    dx, dy = flat.get("distance_x", 0.0), flat.get("distance_y", 0.0)
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-3 and math.isfinite(n) else (1.0, 0.0)


def ridge_boxes(p: dict[str, Any], center: tuple[float, float], clear: float) -> list[dict[str, list[float]]]:
    """The ridge lattice as softsim ground boxes, around ``center`` and out to ``extent``, with the square of
    half-size ``clear`` around ``center`` left free."""
    h, w, s, e = p["height"], p["width"], p["spacing"], p["extent"]
    n = math.ceil(e / s)
    boxes = []
    for axis in (0, 1):  # axis 0: ridges along y at x = const; axis 1: along x at y = const
        for k in range(-n, n):
            off = (k + 0.5) * s
            if abs(off) < clear:  # crosses the clear square: two pieces, one each side
                pieces = [(-e, -clear), (clear, e)]
            else:
                pieces = [(-e, e)]
            for a, b in pieces:
                mid, half = 0.5 * (a + b), 0.5 * (b - a)
                if half <= 0:
                    continue
                c = [center[0] + off, center[1] + mid] if axis == 0 else [center[0] + mid, center[1] + off]
                hs = [w / 2, half] if axis == 0 else [half, w / 2]
                boxes.append({"center": [c[0], c[1], h / 2], "half_size": [hs[0], hs[1], h / 2]})
    return boxes


def apply(raw: dict[str, Any], name: str, p: dict[str, Any], flat: dict[str, float]) -> dict[str, Any]:
    """The creature's scenario on terrain ``name``; ``flat`` is its flat-ground summary (heading, footprint)."""
    raw = copy.deepcopy(raw)
    ground = raw.setdefault("ground", {})
    if name == "slope":
        ux, uy = heading(flat)
        g = 9.81
        sim = raw.setdefault("simulation", {})
        gz = sim.get("gravity", [0.0, 0.0, -g])
        g = math.sqrt(sum(float(x) ** 2 for x in gz)) or g
        sa, ca = math.sin(p["angle"]), math.cos(p["angle"])
        sim["gravity"] = [-g * sa * ux, -g * sa * uy, -g * ca]
    elif name == "ridges":
        center = (flat.get("rest_com_x", 0.0), flat.get("rest_com_y", 0.0))
        clear = flat.get("footprint", 0.05) + p["width"] + 2e-3
        ground["obstacles"] = ridge_boxes(p, center, clear)
    elif name == "sand":
        ground["compliance"] = {"timeconst": p["timeconst"], "dampratio": p["dampratio"], "solimp": list(p["solimp"])}
    else:
        raise TerrainError(f"unknown terrain {name!r}")
    raw["name"] = f"{raw.get('name', 'creature')}_{name}"
    return raw


def score(name: str, summary: dict[str, float], flat: dict[str, float]) -> float:
    """Distance on the terrain; on the slope, the displacement along the uphill heading."""
    if name == "slope":
        ux, uy = heading(flat)
        return summary["distance_x"] * ux + summary["distance_y"] * uy
    return summary["distance"]


def evaluate(raw: dict[str, Any], base_dir: str, terrains: dict[str, dict[str, Any]],
             flat: dict[str, float] | None = None) -> dict[str, Any]:
    """Flat run (unless ``flat`` is given), then one run per terrain; see the module docstring."""
    if flat is None:
        flat = run_summary((raw, base_dir))
    if "error" in flat:
        return flat
    out: dict[str, Any] = dict(flat)
    values = [flat["distance"]]
    for name, p in terrains.items():
        s = run_summary((apply(raw, name, p, flat), base_dir))
        if "error" in s:
            out[f"{name}/error"] = s["error"]
            values.append(-math.inf)
            continue
        for k in _KEEP:
            if k in s:
                out[f"{name}/{k}"] = s[k]
        out[f"{name}/distance"] = score(name, s, flat)
        values.append(out[f"{name}/distance"])
    values = [v if math.isfinite(v) else -math.inf for v in values]
    out["terrain_mean"] = sum(values) / len(values) if all(math.isfinite(v) for v in values) else -math.inf
    out["terrain_min"] = min(values)
    return out


def max_height(summary: dict[str, Any]) -> float:
    """The highest ``final_max_z`` over flat and every terrain run (a blow-up launches the body)."""
    return max([v for k, v in summary.items() if (k == "final_max_z" or k.endswith("/final_max_z"))
                and isinstance(v, (int, float))] or [0.0])


def run_task(task: tuple[dict[str, Any], str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Worker: ``(raw, base_dir, terrains)`` -> :func:`evaluate` (flat only when ``terrains`` is empty)."""
    raw, base_dir, terrains = task
    return evaluate(raw, base_dir, terrains) if terrains else run_summary((raw, base_dir))


def transfer(run_dirs: list[str], top: int = 3, terrains: dict[str, dict[str, Any]] | None = None,
             jobs: int = 1) -> dict[str, list[dict[str, Any]]]:
    """Re-run the ``top`` best creatures of each finished run on flat ground and every terrain (all of them when
    ``terrains`` is None); writes ``terrain.yaml`` in each run directory and returns ``{run_dir: rows}``."""
    from pathlib import Path

    import yaml
    from softsim.config.loader import load_yaml

    from squishlings._jobs import process_pool
    from squishlings.replay import _base_dir

    terrains = parse({k: None for k in DEFAULTS}) if terrains is None else terrains
    index, tasks = [], []
    for run in run_dirs:
        report = load_yaml(Path(run) / "report.yaml")
        base_dir = str(_base_dir(report))
        for rank, b in enumerate(report["best"][:top]):
            index.append((run, rank, b))
            tasks.append((load_yaml(Path(run) / b["file"]), base_dir, terrains))
    if jobs > 1:
        with process_pool(jobs) as pool:
            results = list(pool.map(run_task, tasks))
    else:
        results = [run_task(t) for t in tasks]
    out: dict[str, list[dict[str, Any]]] = {}
    for (run, rank, b), s in zip(index, results):
        row: dict[str, Any] = {"rank": rank + 1, "key": b["key"], "file": b["file"]}
        if "error" in s:
            row["error"] = s["error"]
        else:
            row["flat"] = float(s["distance"])
            for name in terrains:
                row[name] = float(s.get(f"{name}/distance", float("nan")))
                if f"{name}/error" in s:
                    row[f"{name}_error"] = s[f"{name}/error"]
            row["mean"], row["min"] = float(s["terrain_mean"]), float(s["terrain_min"])
        out.setdefault(run, []).append(row)
    for run, rows in out.items():
        (Path(run) / "terrain.yaml").write_text(yaml.safe_dump({"terrains": terrains, "creatures": rows},
                                                               sort_keys=False))
    return out
