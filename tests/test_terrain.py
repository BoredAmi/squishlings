"""Terrains: parsing, the slope's gravity, the ridge lattice's clear start, evaluation and a tiny terrain job."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from squishlings import terrain

CREATURES = Path(__file__).resolve().parents[1] / "squishlings" / "creatures"


def test_parse_fills_defaults_and_rejects_unknown():
    t = terrain.parse({"slope": {"angle": "10 deg"}, "ridges": None})
    assert t["slope"]["angle"] == pytest.approx(math.radians(10))
    assert t["ridges"]["spacing"] == pytest.approx(0.04) and t["ridges"]["height"] == pytest.approx(0.003)
    with pytest.raises(terrain.TerrainError, match="unknown terrain"):
        terrain.parse({"ice": None})
    with pytest.raises(terrain.TerrainError, match="unknown keys"):
        terrain.parse({"slope": {"tilt": 3}})
    assert "slope" in terrain.describe(t) and terrain.describe({}) == "flat ground"


def test_slope_tilts_gravity_against_the_heading():
    p = terrain.parse({"slope": {"angle": "30 deg"}})["slope"]
    raw = terrain.apply({"simulation": {}}, "slope", p, {"distance_x": 0.0, "distance_y": -0.2})
    g = np.array(raw["simulation"]["gravity"])
    assert np.linalg.norm(g) == pytest.approx(9.81)
    np.testing.assert_allclose(g, [0.0, 9.81 * 0.5, -9.81 * math.cos(math.radians(30))], atol=1e-9)  # pulls back
    # uphill score: displacement along the flat heading
    assert terrain.score("slope", {"distance_x": 0.0, "distance_y": -0.05}, {"distance_x": 0, "distance_y": -1}) \
        == pytest.approx(0.05)
    assert terrain.heading({"distance_x": 1e-5, "distance_y": 0.0}) == (1.0, 0.0)


def test_ridges_leave_the_start_clear():
    p = terrain.parse({"ridges": {"extent": "0.3 m"}})["ridges"]
    boxes = np.array([[*b["center"], *b["half_size"]] for b in terrain.ridge_boxes(p, (0.1, 0.0), 0.05)])
    lo, hi = boxes[:, :3] - boxes[:, 3:], boxes[:, :3] + boxes[:, 3:]
    e = 1e-9  # touching the clear square's edge is fine
    inside = (hi[:, 0] > 0.05 + e) & (lo[:, 0] < 0.15 - e) & (hi[:, 1] > -0.05 + e) & (lo[:, 1] < 0.05 - e)
    assert not inside.any()
    assert np.allclose(boxes[:, 2], 0.0015) and np.allclose(boxes[:, 5], 0.0015)
    assert len(boxes) == 2 * (2 * math.ceil(0.3 / 0.04) + 2)  # lines through the clear square come in two pieces


def test_evaluate_on_a_resting_block():
    pytest.importorskip("mujoco")
    from softsim.config.loader import apply_overrides, load_yaml

    raw = apply_overrides(load_yaml(CREATURES / "base.yaml"), ["simulation.duration=0.3 s", "settle=0.1 s",
                                                               "body.geometry.cells=[[aaa, aaa]]"])
    s = terrain.evaluate(raw, str(CREATURES), terrain.parse({"slope": None, "ridges": None, "sand": None}))
    assert "error" not in s, s
    for name in ("slope", "ridges", "sand"):
        assert f"{name}/distance" in s and f"{name}/error" not in s
    assert s["footprint"] > 0.01
    assert s["sand/final_max_z"] < s["final_max_z"]  # sinks into the soft ground
    assert math.isfinite(s["terrain_mean"]) and s["terrain_min"] <= s["terrain_mean"]
    assert terrain.max_height(s) >= s["final_max_z"]


def test_tiny_sticks_terrain_job(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("gmsh")
    import yaml

    from squishlings import Evolution

    spec = {"scenario": str(CREATURES / "base.yaml"), "overrides": ["simulation.duration=0.1 s", "settle=0.05 s"],
            "genome": {"encoding": "sticks", "tissues": ["ecoflex_00_30", "dragon_skin_10"],
                       "muscle": {"groups": 2, "amplitude": ["2 kPa", "20 kPa"], "start": "0.05 s"},
                       "max_segments": 4, "mesh_size": "4 mm"},
            "evolution": {"population": 2, "offspring": 1, "generations": 1, "seed": 0},
            "terrains": {"slope": None, "ridges": {"extent": "0.3 m"}, "sand": None},
            "fitness": {"metric": "terrain_mean"},
            "seeds": [str(CREATURES / "sticks" / "walker.txt")], "output": str(tmp_path / "out")}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
    report = Evolution(tmp_path / "job.yaml").run()
    assert report["failed"] == 0 and set(report["terrains"]) == {"slope", "ridges", "sand"}
    line = yaml.safe_load((tmp_path / "out" / "evaluations.jsonl").read_text().splitlines()[0])
    assert {"slope/distance", "ridges/distance", "sand/distance", "terrain_mean"} <= set(line["summary"])
    check = Evolution(tmp_path / "job.yaml").check(top=1)
    assert all("error" not in v for v in check["creatures"][0]["variants"].values())
    out = terrain.transfer([str(tmp_path / "out")], top=1, terrains=terrain.parse({"sand": None}))
    assert "sand" in out[str(tmp_path / "out")][0]
    spec["terrains"] = {"lava": None}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(Exception, match="unknown terrain"):
        Evolution(tmp_path / "job.yaml")


def test_tiny_map_elites_with_terrain_niches(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("gmsh")
    import yaml

    from squishlings import Evolution

    spec = {"scenario": str(CREATURES / "base.yaml"), "overrides": ["simulation.duration=0.1 s", "settle=0.05 s"],
            "genome": {"encoding": "sticks", "tissues": ["ecoflex_00_30", "dragon_skin_10"],
                       "muscle": {"groups": 2, "amplitude": ["2 kPa", "20 kPa"], "start": "0.05 s"},
                       "max_segments": 4, "mesh_size": "4 mm"},
            "evolution": {"method": "map_elites", "population": 3, "offspring": 2, "generations": 1, "seed": 0},
            "archive": {"descriptors": ["limbs", "elongation"], "per_terrain": True},
            "terrains": {"slope": None, "sand": None}, "fitness": {"metric": "terrain_mean"},
            "seeds": [str(CREATURES / "sticks" / "walker.txt"), str(CREATURES / "sticks" / "inchworm.txt")],
            "output": str(tmp_path / "out")}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
    report = Evolution(tmp_path / "job.yaml").run()
    assert set(report["niches"]) == {"flat", "slope", "sand"}
    for n, niche in report["niches"].items():
        assert niche["metric"] == ("distance" if n == "flat" else f"{n}/distance")
        assert len(niche["qd_history"]) == 2 and niche["best"]
        assert (tmp_path / "out" / f"archive_{n}.png").is_file()
        assert (tmp_path / "out" / niche["best"][0]["file"]).is_file()
    spec["terrains"] = {}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(Exception, match="per_terrain"):
        Evolution(tmp_path / "job.yaml")
