"""Soft-stick genomes: text round trip, repair limits, operators, placement, tet labels and meshing."""

import math

import numpy as np
import pytest

from squishlings.sticks import (ParseError, StickSpace, crossover, from_text, mutate, place, random_genome)

SPACE = StickSpace()
WALKER = """seg len=30 r=6 mat=a
  seg len=14 r=4 mat=a yaw=-60 pitch=-40 muscle=1 bend roll=90 sym
    seg len=9 r=3 mat=b pitch=-30
---
freq=4 amp=8000 phases=0,3.142
"""


def test_text_round_trip_and_key():
    g = from_text(WALKER, SPACE)
    assert from_text(g.text, SPACE).text == g.text
    assert from_text(g.text, SPACE).key == g.key
    assert g.n_segments == 5  # root + mirrored (leg + foot) pair
    assert len(g.phases) == SPACE.n_muscles


def test_parse_errors():
    with pytest.raises(ParseError):
        from_text("seg len=10 wings=2\n---\n", SPACE)
    with pytest.raises(ParseError):
        from_text("---\nfreq=2\n", SPACE)
    with pytest.raises(ParseError):
        from_text("seg len=10\nseg len=10\n---\n", SPACE)


def test_repair_clamps_and_cleans():
    g = from_text("seg len=500 r=0.1 mat=z yaw=40 sym\n  seg len=10 roll=30 muscle=9 bend\n---\namp=1e9 freq=0\n",
                  SPACE)
    root, child = g.root, g.root.children[0]
    assert root.length == SPACE.length[1] and root.radius == SPACE.radius[0] and root.mat == "a"
    assert root.yaw == 0 and not root.sym                       # the root's direction is fixed
    assert child.muscle == 0 and not child.bend and child.roll == 0  # invalid group: passive, no bend
    assert g.amplitude == SPACE.amplitude[1] and g.frequency == SPACE.frequency[0]


def test_repair_prunes_to_segment_limit():
    lines = ["seg len=10 r=4"] + ["  seg len=10 r=3 yaw=90 sym"] * 12
    g = from_text("\n".join(lines) + "\n---\n", SPACE)
    assert g.n_segments <= SPACE.max_segments


def test_operators_return_valid_canonical_genomes():
    rng = np.random.default_rng(0)
    pop = [random_genome(SPACE, rng) for _ in range(10)]
    for _ in range(100):
        a, b = pop[rng.integers(len(pop))], pop[rng.integers(len(pop))]
        child = mutate(crossover(a, b, SPACE, rng), SPACE, rng) if rng.random() < 0.5 else mutate(a, SPACE, rng)
        assert 1 <= child.n_segments <= SPACE.max_segments
        assert from_text(child.text, SPACE).text == child.text
        pop.append(child)
    assert len({g.key for g in pop}) > 50  # operators explore


def test_operators_are_deterministic_for_a_seed():
    def run(seed):
        rng = np.random.default_rng(seed)
        g = random_genome(SPACE, rng)
        return mutate(crossover(g, random_genome(SPACE, rng), SPACE, rng), SPACE, rng).text
    assert run(3) == run(3)


def test_placement_mirrors_sym_subtrees():
    segs = place(from_text(WALKER, SPACE))
    root, leg, foot, leg_m, foot_m = segs
    np.testing.assert_allclose(root.end, [30, 0, 0], atol=1e-9)
    np.testing.assert_allclose(leg.start, root.end)
    np.testing.assert_allclose(foot.start, leg.end)
    for a, b in ((leg, leg_m), (foot, foot_m)):  # mirror image across the x-z plane
        np.testing.assert_allclose(a.end * [1, -1, 1], b.end, atol=1e-9)
    assert leg.end[2] < 0 and foot.end[2] < leg.end[2]  # both pitch down
    for s in segs:
        np.testing.assert_allclose(s.R.T @ s.R, np.eye(3), atol=1e-12)


def test_tet_labels_split_bend_muscles():
    from squishlings.sticks_mesh import label_tets

    g = from_text("seg len=20 r=5\n  seg len=20 r=4 muscle=1 bend\n---\n", SPACE)
    segs = place(g)
    pts = np.array([[10, 0, 0], [30, 0, 2], [30, 0, -2], [41, 0, 0.5]])
    assert label_tets(pts, segs) == ["a", "m1e", "m1f", "m1e"]
    rolled = place(from_text("seg len=20 r=5\n  seg len=20 r=4 muscle=1 bend roll=90\n---\n", SPACE))
    assert label_tets(np.array([[30, 2, 0], [30, -2, 0]]), rolled) == ["m1f", "m1e"]


def test_mesh_regions_and_flat_base(tmp_path):
    pytest.importorskip("gmsh")
    from squishlings.sticks_mesh import FLAT, build_mesh

    g = from_text(WALKER, SPACE)
    out = build_mesh(g, tmp_path / "walker.msh")
    assert set(out["tets"]) == {"a", "b", "m1e", "m1f", "m3e", "m3f"}
    assert all(n > 50 for n in out["tets"].values())

    import gmsh
    gmsh.initialize()
    try:
        gmsh.open(str(tmp_path / "walker.msh"))
        z = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)[:, 2]
    finally:
        gmsh.finalize()
    lowest = min(min(s.start[2], s.end[2]) - 1.1 * s.radius for s in place(g))
    assert z.min() == pytest.approx(lowest + FLAT, abs=1e-6)  # cut flat at the feet
    assert math.isfinite(z.max())


def test_tiny_sticks_evolution_resumes_and_checks(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("gmsh")
    import yaml

    from squishlings import Evolution

    creatures = __import__("pathlib").Path(__file__).resolve().parents[1] / "squishlings" / "creatures"

    def job(generations):
        spec = {"scenario": str(creatures / "base.yaml"), "overrides": ["simulation.duration=0.1 s", "settle=0.05 s"],
                "genome": {"encoding": "sticks", "tissues": ["ecoflex_00_30", "dragon_skin_10"],
                           "muscle": {"groups": 2, "amplitude": ["2 kPa", "20 kPa"], "start": "0.05 s"},
                           "max_segments": 4, "mesh_size": "4 mm"},
                "evolution": {"population": 2, "offspring": 1, "generations": generations, "seed": 0},
                "seeds": [str(creatures / "sticks" / "walker.txt")], "output": str(tmp_path / "out")}
        path = tmp_path / "job.yaml"
        path.write_text(yaml.safe_dump(spec))
        return path

    first = Evolution(job(1))
    report = first.run()
    assert report["failed"] == 0 and first.n_simulated == 3
    assert "seg " in report["best"][0]["genome"]["text"]
    assert list((tmp_path / "out" / "meshes").glob("*.msh"))
    again = Evolution(job(1))
    again.run()
    assert again.n_simulated == 0  # replayed from the log
    check = again.check(top=1)
    assert set(check["creatures"][0]["variants"]) == {"mesh_2.67mm", "half_timestep", "friction_x0.8",
                                                       "friction_x1.2"}
    assert all("error" not in v for v in check["creatures"][0]["variants"].values())


def test_map_elites_descriptors_and_archive():
    from squishlings.map_elites import Archive, Axis, stick_descriptors

    creatures = __import__("pathlib").Path(__file__).resolve().parents[1] / "squishlings" / "creatures" / "sticks"
    d = {n: stick_descriptors(from_text((creatures / f"{n}.txt").read_text(), SPACE))
         for n in ("inchworm", "walker", "champion_1")}
    assert d["inchworm"]["limbs"] == 1 and d["walker"]["limbs"] == 3 and d["champion_1"]["limbs"] == 4
    assert d["inchworm"]["elongation"] > 2 > d["champion_1"]["elongation"]  # a chain is long and thin
    assert stick_descriptors(from_text("seg len=20 r=5\n---\n", SPACE))["limbs"] == 0

    axes = [Axis("limbs", 0, 6, 6), Axis("elongation", -1, 3, 5)]
    assert axes[0].index(5) == 5 and axes[0].index(9) == 5 and axes[1].index(-7) == 0
    arch = Archive(axes, stick_descriptors)
    walker = from_text((creatures / "walker.txt").read_text(), SPACE)
    assert arch.add(walker, 0.1) and not arch.add(walker, 0.05) and arch.add(walker, 0.2)
    assert not arch.add(walker, float("-inf"))
    assert len(arch.cells) == 1 and arch.qd_score == pytest.approx(0.2) and arch.coverage == 1 / 30


def test_tiny_map_elites_run(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("gmsh")
    import yaml

    from squishlings import Evolution

    creatures = __import__("pathlib").Path(__file__).resolve().parents[1] / "squishlings" / "creatures"
    spec = {"scenario": str(creatures / "base.yaml"), "overrides": ["simulation.duration=0.1 s", "settle=0.05 s"],
            "genome": {"encoding": "sticks", "tissues": ["ecoflex_00_30", "dragon_skin_10"],
                       "muscle": {"groups": 2, "amplitude": ["2 kPa", "20 kPa"], "start": "0.05 s"},
                       "max_segments": 4, "mesh_size": "4 mm"},
            "evolution": {"method": "map_elites", "population": 3, "offspring": 2, "generations": 1, "seed": 0},
            "archive": {"descriptors": ["limbs", "elongation"]},
            "seeds": [str(creatures / "sticks" / "walker.txt"), str(creatures / "sticks" / "inchworm.txt")],
            "output": str(tmp_path / "out")}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
    report = Evolution(tmp_path / "job.yaml").run()
    arch = report["archive"]
    assert report["method"] == "map_elites" and report["failed"] == 0
    assert len(arch["cells"]) >= 2  # the walker and the inchworm are different kinds
    assert len({tuple(c["cell"]) for c in arch["cells"]}) == len(arch["cells"])
    assert (tmp_path / "out" / "archive.png").is_file()
    assert len(list((tmp_path / "out" / "elites").glob("cell*.yaml"))) == len(arch["cells"])
    with pytest.raises(Exception, match="archive.descriptors"):
        spec["archive"] = {"descriptors": ["cells"]}  # a voxel descriptor
        (tmp_path / "job.yaml").write_text(yaml.safe_dump(spec))
        Evolution(tmp_path / "job.yaml")
