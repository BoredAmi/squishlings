"""Soft creatures: genome operators, genome <-> scenario, and a tiny resumable evolution run (MuJoCo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import ndimage
from softsim.config.loader import load_yaml
from softsim.geometry import SoftVoxels

from squishlings.build import CreatureBuilder, genome_from_scenario
from squishlings.genome import GenomeSpace, crop, crossover, mutate, random_genome

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
LOCO = ROOT / "squishlings" / "creatures"
FACE = ndimage.generate_binary_structure(3, 1)


def _space(**kw) -> GenomeSpace:
    return GenomeSpace(**{"shape": (5, 5, 3), "n_tissues": 2, "n_muscles": 3, "min_cells": 4, "max_cells": 30, **kw})


def test_operators_keep_one_grounded_body_within_limits():
    space, rng = _space(), np.random.default_rng(1)
    pop = [random_genome(space, rng) for _ in range(10)]
    for _ in range(300):
        a, b = pop[rng.integers(10)], pop[rng.integers(10)]
        child = mutate(crossover(a, b, space, rng), space, rng)
        assert ndimage.label(child.cells > 0, structure=FACE)[1] == 1
        assert child.cells[:, :, 0].any()
        assert space.min_cells <= child.n_cells <= space.max_cells
        assert child.cells.max() < space.n_codes
        assert np.all((0 <= child.phases) & (child.phases < 2 * np.pi))
        pop[rng.integers(10)] = child


def test_determinism_and_key():
    space = _space()
    a = [random_genome(space, np.random.default_rng(7)).key for _ in range(2)]
    assert a[0] == a[1]
    g = random_genome(space, np.random.default_rng(7))
    assert mutate(g, space, np.random.default_rng(3)).key == mutate(g, space, np.random.default_rng(3)).key
    assert mutate(g, space, np.random.default_rng(3)).key != g.key


def test_mutation_changes_body_or_signal():
    space, rng = _space(), np.random.default_rng(0)
    g = random_genome(space, rng)
    assert all(mutate(g, space, rng).key != g.key for _ in range(20))


def _builder(space: GenomeSpace) -> CreatureBuilder:
    return CreatureBuilder(load_yaml(LOCO / "base.yaml"), 0.01, ("ecoflex_00_30", "dragon_skin_10"), "ecoflex_00_30",
                           0.3, space)


def test_builder_makes_a_valid_voxel_body_with_zones_and_muscles():
    space = _space()
    g = random_genome(space, np.random.default_rng(4))
    raw = _builder(space).scenario(g)
    geo = SoftVoxels(**raw["body"]["geometry"])
    assert geo.n_cells == g.n_cells
    labels = {c for layer in raw["body"]["geometry"]["cells"] for row in layer for c in row} - {"."}
    assert {z["region"] for z in raw["body"]["zones"]} == labels & {"b"}
    assert {a["region"] for a in raw["actuators"]} == labels & set("123")
    assert np.array_equal(geo.to_array() != ".", crop(g.cells) > 0)


@pytest.mark.parametrize("name", ["inchworm", "quad"])
def test_seed_round_trip(name):
    space = GenomeSpace(shape=(6, 6, 4), n_tissues=2, n_muscles=4, frequency=(1.0, 12.0), amplitude=(5e3, 40e3))
    builder = _builder(space)
    seed = load_yaml(LOCO / f"{name}.yaml")
    g = genome_from_scenario(seed, builder)
    raw = builder.scenario(g)
    assert raw["body"]["geometry"]["cells"] == seed["body"]["geometry"]["cells"]
    phases = {a["region"]: a["oscillator"]["phase"] for a in raw["actuators"]}
    assert phases["2"] - phases["1"] == pytest.approx({"inchworm": np.pi, "quad": np.pi / 2}[name], abs=1e-4)
    assert g.frequency == pytest.approx(10.0) and g.amplitude == pytest.approx(40e3)


def _job(tmp_path, generations: int):
    job = {"scenario": str(LOCO / "base.yaml"), "overrides": ["simulation.duration=0.25 s", "settle=0.05 s"],
           "genome": {"shape": [3, 3, 2], "voxel": "10 mm", "tissues": ["ecoflex_00_30"],
                      "muscle": {"groups": 2, "amplitude": "20 kPa", "start": "0.05 s"}, "min_cells": 3,
                      "max_cells": 8, "fill": 0.5},
           "evolution": {"population": 3, "offspring": 2, "generations": generations, "seed": 0},
           "output": str(tmp_path / "out")}
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job))
    return path


def test_tiny_evolution_is_resumable(tmp_path):
    pytest.importorskip("mujoco")
    from squishlings import Evolution

    first = Evolution(_job(tmp_path, 1))
    report = first.run()
    assert first.n_simulated >= 3 and report["failed"] == 0
    hist = pd.read_parquet(tmp_path / "out" / "history.parquet")
    best = hist[hist.generation == hist.generation.max()].sort_values("rank")
    assert best.fitness.is_monotonic_decreasing
    # the elitist loop never loses its best creature
    per_gen = hist.groupby("generation").fitness.max()
    assert per_gen.is_monotonic_increasing
    # rerun with one more generation: generation 0 and 1 are replayed from the log, not simulated
    again = Evolution(_job(tmp_path, 2))
    again.run()
    assert again.n_simulated <= 2
    h2 = pd.read_parquet(tmp_path / "out" / "history.parquet")
    pd.testing.assert_frame_equal(h2[h2.generation <= 1].reset_index(drop=True), hist.reset_index(drop=True))
    f = tmp_path / "out" / report["best"][0]["file"]
    assert report["best"][0]["key"] in yaml.safe_load(f.read_text())["description"]
    from softsim.runner import Simulation  # a best creature is a runnable scenario

    Simulation(f, ["simulation.duration=0.01 s", "output.save_model=false"]).close()
