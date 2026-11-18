"""A small (mu+lambda) evolution of soft-stick creatures. Prototype: not yet behind ``squishlings evolve``.

    python -m squishlings.sticks_evolve --generations 4 --population 8 --offspring 8 --jobs 6 --out runs/sticks_test

Each creature is meshed in a child process (killed after ``--mesh-timeout`` s: gmsh can hang on awkward
overlaps) and run in the ``locomotion`` scenario of ``squishlings/creatures/base.yaml`` on MuJoCo. Fitness is the
planar distance the centre of mass travels after settling. A creature that fails to mesh or diverges scores 0
and keeps its error in the log.

Outputs in ``--out``: ``evaluations.jsonl`` (one line per creature, the cache: a stopped run resumes without
re-simulating), ``history.json`` (best and median per generation), ``meshes/<key>.msh``, and for the best
creatures ``best_<rank>.txt`` (genome) and ``best_<rank>.yaml`` (``softsim run`` it).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from squishlings.sticks import StickGenome, StickSpace, crossover, from_text, mutate, place, random_genome
from squishlings.sticks_mesh import scenario

BASE = Path(__file__).resolve().parent / "creatures" / "base.yaml"
TISSUES = {"a": "ecoflex_00_30", "b": "dragon_skin_10"}
SEEDS = {
    "inchworm": """seg len=20 r=5 mat=a
  seg len=20 r=4.5 mat=a muscle=1 bend
    seg len=20 r=4.5 mat=a muscle=2 bend
---
freq=3 amp=8000 phases=0,1.571,0,0
""",
    "walker": """seg len=30 r=6 mat=a
  seg len=14 r=4 mat=a yaw=-60 pitch=-40 muscle=1 bend roll=90 sym
    seg len=9 r=3 mat=b pitch=-30
  seg len=20 r=4.5 mat=a yaw=180 pitch=-20 muscle=2 bend
---
freq=4 amp=8000 phases=0,3.142,0,0
""",
}


def _base(duration: float) -> dict[str, Any]:
    raw = yaml.safe_load(BASE.read_text())
    raw["simulation"]["duration"] = duration
    return raw


def evaluate(text: str, out: str, duration: float, mesh_size: float, mesh_timeout: float) -> dict[str, Any]:
    """Mesh + simulate one creature (runs in a worker process)."""
    from softsim.config.units import parse_quantity
    from softsim.runner import Simulation

    space = StickSpace()
    g = from_text(text, space)
    out_dir = Path(out)
    msh = out_dir / "meshes" / f"{g.key}.msh"
    gtxt = out_dir / "meshes" / f"{g.key}.txt"
    msh.parent.mkdir(parents=True, exist_ok=True)
    gtxt.write_text(text)
    rec: dict[str, Any] = {"key": g.key, "text": text, "fitness": 0.0}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-m", "squishlings.sticks_mesh", str(gtxt), str(msh), str(mesh_size)],
                              capture_output=True, text=True, timeout=mesh_timeout)
        last = (proc.stdout.strip().splitlines() or ["{}"])[-1]
        meshed = json.loads(last) if last.startswith("{") else {"error": proc.stderr[-300:]}
    except subprocess.TimeoutExpired:
        meshed = {"error": f"meshing timed out after {mesh_timeout:g} s"}
    rec["mesh_s"] = round(time.perf_counter() - t0, 2)
    if "tets" not in meshed:
        rec["error"] = "mesh: " + str(meshed.get("error", "unknown"))
        return rec
    rec["tets"] = int(sum(meshed["tets"].values()))
    base = _base(duration)
    settle = float(parse_quantity(base.get("settle", 0.3), "time"))
    raw = scenario(g, Path("meshes") / msh.name, place(g), set(meshed["tets"]), base, TISSUES, start=settle)
    rec["scenario"] = raw
    t0 = time.perf_counter()
    try:
        res = Simulation(raw, base_dir=out_dir).run()
        s = res.summary
        rec.update(fitness=float(s["distance"]), distance_x=float(s["distance_x"]),
                   distance_y=float(s["distance_y"]), max_tilt=float(s["max_tilt"]), flipped=float(s["flipped"]))
    except Exception as exc:  # diverged, refused by the pre-run checks...
        rec["error"] = f"sim: {type(exc).__name__}: {str(exc)[:300]}"
    rec["sim_s"] = round(time.perf_counter() - t0, 1)
    return rec


class Run:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.space = StickSpace()
        self.rng = np.random.default_rng(args.seed)
        self.log = self.out / "evaluations.jsonl"
        self.done: dict[str, dict[str, Any]] = {}
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                r = json.loads(line)
                self.done[r["key"]] = r
        self.pool = ProcessPoolExecutor(args.jobs)
        self.history: list[dict[str, Any]] = []

    def ensure(self, genomes: list[StickGenome]) -> None:
        todo = {g.key: g for g in genomes if g.key not in self.done}
        if not todo:
            return
        futs = [self.pool.submit(evaluate, g.text, str(self.out), self.a.duration, self.a.mesh_size,
                                 self.a.mesh_timeout) for g in todo.values()]
        with self.log.open("a") as f:
            for fut in futs:
                r = fut.result()
                self.done[r["key"]] = r
                f.write(json.dumps(r) + "\n")
                f.flush()

    def fitness(self, g: StickGenome) -> float:
        return self.done[g.key]["fitness"]

    def rank(self, genomes: list[StickGenome]) -> list[StickGenome]:
        uniq = list({g.key: g for g in genomes}.values())
        return sorted(uniq, key=lambda g: (-self.fitness(g), g.key))

    def select(self, ranked: list[StickGenome]) -> StickGenome:
        idx = self.rng.integers(0, len(ranked), self.a.tournament)
        return ranked[int(idx.min())]

    def report(self, gen: int, ranked: list[StickGenome], t0: float) -> None:
        fits = [self.fitness(g) for g in ranked]
        failed = sum("error" in self.done[g.key] for g in ranked)
        entry = {"generation": gen, "best_mm": round(1e3 * fits[0], 2), "median_mm": round(1e3 * float(np.median(fits)), 2),
                 "best_key": ranked[0].key, "evaluated": len(self.done), "failed_in_population": failed,
                 "wall_min": round((time.perf_counter() - t0) / 60, 1)}
        self.history.append(entry)
        print(json.dumps(entry), flush=True)
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2))

    def write_best(self, ranked: list[StickGenome], top: int = 3) -> None:
        for i, g in enumerate(ranked[:top]):
            r = self.done[g.key]
            (self.out / f"best_{i + 1}.txt").write_text(f"# distance {1e3 * r['fitness']:.1f} mm, key {g.key}\n" + g.text)
            if "scenario" in r:
                (self.out / f"best_{i + 1}.yaml").write_text(
                    f"# Soft-stick creature {g.key}: {1e3 * r['fitness']:.1f} mm. Run: softsim run best_{i + 1}.yaml\n"
                    + yaml.safe_dump(r["scenario"], sort_keys=False))

    def run(self) -> None:
        t0 = time.perf_counter()
        texts = list(SEEDS.values()) + [Path(f).read_text() for f in self.a.seeds]
        pop = list({g.key: g for g in (from_text(t, self.space) for t in texts)}.values())
        while len(pop) < self.a.population:  # (seeds beyond the population size all compete in generation 0)
            pop.append(random_genome(self.space, self.rng))
        self.ensure(pop)
        ranked = self.rank(pop)
        self.report(0, ranked, t0)
        for gen in range(1, self.a.generations + 1):
            keys = {g.key for g in ranked}
            children: list[StickGenome] = []
            tries = 0
            while len(children) < self.a.offspring and tries < 20 * self.a.offspring:
                tries += 1
                a = self.select(ranked)
                if self.rng.random() < self.a.crossover:
                    child = mutate(crossover(a, self.select(ranked), self.space, self.rng), self.space, self.rng)
                else:
                    child = mutate(a, self.space, self.rng)
                if child.key not in keys:
                    keys.add(child.key)
                    children.append(child)
            self.ensure(children)
            ranked = self.rank(ranked + children)[: self.a.population]
            self.report(gen, ranked, t0)
            self.write_best(ranked)
        self.pool.shutdown()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="runs/sticks_test")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--offspring", type=int, default=8)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--crossover", type=float, default=0.3)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=2.0, help="simulated seconds (0.3 s of them settling)")
    p.add_argument("--mesh-size", type=float, default=3.0, help="mm")
    p.add_argument("--seeds", nargs="*", default=[], help="genome text files added to the first population")
    p.add_argument("--mesh-timeout", type=float, default=30.0, help="s")
    Run(p.parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
