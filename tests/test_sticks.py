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
