"""The locomotion scenario: orientation from the vertices, a resting body stays put, the inchworm seed crawls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squishlings.locomotion import _kabsch

LOCO = Path(__file__).resolve().parents[1] / "squishlings" / "creatures"


def test_kabsch_recovers_a_rotation():
    rng = np.random.default_rng(0)
    rest = rng.normal(size=(30, 3))
    w = rng.random(30)
    w /= w.sum()
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tilt = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    Q = R @ tilt
    np.testing.assert_allclose(_kabsch(rest, rest @ Q.T + [1, 2, 3], w), Q, atol=1e-12)


def _run(name: str, *overrides: str):
    pytest.importorskip("mujoco")
    from softsim.runner import Simulation

    sim = Simulation(LOCO / name, ["output.save_model=false", *overrides])
    try:
        return sim.run().summary
    finally:
        sim.close()


def test_resting_voxel_body_does_not_move():
    s = _run("base.yaml", "simulation.duration=0.6 s", "body.geometry.cells=[[aaa, aaa]]")
    assert s["distance"] < 1e-4 and s["max_tilt"] < 0.01 and s["flipped"] == 0


def test_inchworm_crawls_forward():
    """Seed gait (10 Hz, 40 kPa): ~23 mm along x in its first 0.7 s of gait at the default resolution. It also drifts
    ~5 mm sideways although the body is mirror-symmetric in y: the 6-tet split of each cube is not, so the mesh
    breaks the symmetry (docs/creatures.md)."""
    s = _run("inchworm.yaml", "simulation.duration=1 s")
    assert s["distance_x"] > 0.01
    assert abs(s["distance_y"]) < 0.35 * s["distance_x"]
    assert s["flipped"] == 0
