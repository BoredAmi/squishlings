"""Evolve soft voxel creatures: ``squishlings evolve jobs/crawler.yaml``.

A job file::

    scenario: ../squishlings/creatures/base.yaml   # a locomotion scenario; its body geometry, material, zones
                                                  # and actuators are replaced by each creature's
    backend: mujoco                               # optional
    overrides: [simulation.duration=3 s]
    genome:
      shape: [6, 6, 4]                            # the box creatures grow in (cells)
      voxel: 10 mm
      tissues: [ecoflex_00_30, dragon_skin_30_marechal2021]   # passive cards (labels a, b, ...)
      muscle: {material: ecoflex_00_30, groups: 4, frequency: 2 Hz, amplitude: [4 kPa, 12 kPa], start: 0.3 s}
      min_cells: 4
      max_cells: 40
      fill: 0.5
    evolution: {population: 24, offspring: 24, generations: 40, tournament: 3, crossover: 0.5, seed: 0}
    fitness: {metric: distance, size_penalty: 0}  # per-cell penalty in the metric's unit
    seeds: [../squishlings/creatures/inchworm.yaml]   # optional hand-made creatures in the first population
    output: ../runs/crawler

Method: a (mu + lambda) evolution strategy. Each generation draws ``offspring`` children by tournament selection,
uniform choice between crossover (probability ``crossover``) + mutation and mutation alone
(:mod:`squishlings.genome`); the next population is the best ``population`` of parents and children together
(elitist: the best creature is never lost). Children identical to a creature already evaluated are not simulated
again. A creature whose simulation fails (unstable, crash) gets the worst fitness and is counted in the report.

Every evaluation is appended to ``<output>/evaluations.jsonl``, keyed by the job's scenario, code
(softsim's and squishlings' source, :func:`squishlings._jobs.code_hash`), overrides and genome settings plus the creature's content hash; rerunning the job
replays them, and the run is deterministic for a seed, so a stopped run resumes where it stopped and raising
``generations`` continues it.

Outputs: ``history.parquet`` (every creature of every generation: generation, key, parents, fitness, summary),
``best/rank<k>_<key>.yaml`` (the best creatures as runnable scenarios), ``report.yaml``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.io import save_table

from squishlings._jobs import code_hash
from squishlings._jobs import content_hash as _hash
from squishlings._jobs import process_pool as _process_pool
from squishlings._jobs import run_summary as _run_summary
from squishlings.build import MUSCLE_CHARS, TISSUE_CHARS, CreatureBuilder, genome_from_scenario
from squishlings.genome import Genome, GenomeSpace, crossover, mutate, random_genome

_JOB_FIELDS = {"scenario", "backend", "overrides", "genome", "evolution", "fitness", "seeds", "output"}
_GENOME_FIELDS = {"friction", "shape", "voxel", "tissues", "muscle", "min_cells", "max_cells", "fill", "muscle_fraction",
                  "subdivide", "cell_rate", "phase_sigma", "direction_rate", "scalar_sigma"}
_MUSCLE_FIELDS = {"material", "groups", "frequency", "amplitude", "start"}
_EVO_FIELDS = {"population", "offspring", "generations", "tournament", "crossover", "seed"}


def _check_keys(block: dict, known: set[str], where: str) -> None:
    unknown = set(block) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; known: {sorted(known)}")


def _bounds(value: Any, dimension: str) -> tuple[float, float]:
    """A fixed value or ``[low, high]`` (units allowed)."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ConfigError(f"expected a value or [low, high], got {value!r}")
        lo, hi = (float(parse_quantity(v, dimension)) for v in value)
    else:
        lo = hi = float(parse_quantity(value, dimension))
    if not 0 < lo <= hi:
        raise ConfigError(f"need 0 < low <= high, got {value!r}")
    return lo, hi


@dataclass
class Evolution:
    """Runs an evolution job file; see the module docstring."""

    path: Path
    jobs: int = 1
    n_simulated: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path).resolve()
        if not self.path.is_file():
            raise ConfigError(f"job file not found: {self.path}")
        spec = load_yaml(self.path)
        if not isinstance(spec, dict):
            raise ConfigError("job file must be a mapping")
        _check_keys(spec, _JOB_FIELDS, "job")
        if "scenario" not in spec:
            raise ConfigError("job needs a 'scenario' (a locomotion scenario file)")
        scenario = (self.path.parent / spec["scenario"]).resolve()
        if not scenario.is_file():
            raise ConfigError(f"scenario file not found: {scenario}")
        overrides = list(spec.get("overrides", []))
        if spec.get("backend"):
            overrides.append(f"backend={spec['backend']}")
        self.base_dir = str(scenario.parent)
        base = apply_overrides(load_yaml(scenario), overrides)

        gspec = dict(spec.get("genome") or {})
        _check_keys(gspec, _GENOME_FIELDS, "genome")
        mspec = dict(gspec.get("muscle") or {})
        _check_keys(mspec, _MUSCLE_FIELDS, "genome.muscle")
        tissues = tuple(gspec.get("tissues") or [base.get("body", {}).get("material", "ecoflex_00_30")])
        if not 1 <= len(tissues) <= len(TISSUE_CHARS):
            raise ConfigError(f"genome.tissues: 1 to {len(TISSUE_CHARS)} material cards")
        groups = int(mspec.get("groups", 4))
        if not 0 <= groups <= len(MUSCLE_CHARS):
            raise ConfigError(f"genome.muscle.groups: 0 to {len(MUSCLE_CHARS)}")
        shape = tuple(int(n) for n in gspec.get("shape", (6, 6, 4)))
        if len(shape) != 3 or min(shape) < 1:
            raise ConfigError(f"genome.shape: three positive cell counts, got {shape}")
        rates = {k: float(gspec[k]) for k in ("fill", "muscle_fraction", "cell_rate", "phase_sigma",
                                              "direction_rate", "scalar_sigma") if k in gspec}
        self.space = GenomeSpace(shape=shape, n_tissues=len(tissues), n_muscles=groups,
                                 min_cells=int(gspec.get("min_cells", 4)),
                                 max_cells=None if gspec.get("max_cells") is None else int(gspec["max_cells"]),
                                 frequency=_bounds(mspec.get("frequency", "2 Hz"), "frequency"),
                                 amplitude=_bounds(mspec.get("amplitude", "8 kPa"), "pressure"), **rates)
        self.builder = CreatureBuilder(base, float(parse_quantity(gspec.get("voxel", "10 mm"), "length")), tissues,
                                       mspec.get("material", tissues[0]),
                                       float(parse_quantity(mspec.get("start", "0.3 s"), "time")), self.space,
                                       int(gspec.get("subdivide", 1)),
                                       float(gspec.get("friction",
                                                       base.get("ground", {}).get("friction", {}).get("sliding", 0.8))))

        espec = dict(spec.get("evolution") or {})
        _check_keys(espec, _EVO_FIELDS, "evolution")
        self.population = int(espec.get("population", 24))
        self.offspring = int(espec.get("offspring", self.population))
        self.generations = int(espec.get("generations", 20))
        self.tournament = int(espec.get("tournament", 3))
        self.p_crossover = float(espec.get("crossover", 0.5))
        self.seed = int(espec.get("seed", 0))
        if min(self.population, self.offspring, self.tournament) < 1 or self.generations < 0:
            raise ConfigError("evolution: population, offspring and tournament >= 1, generations >= 0")

        fspec = dict(spec.get("fitness") or {})
        _check_keys(fspec, {"metric", "size_penalty"}, "fitness")
        self.metric = str(fspec.get("metric", "distance"))
        self.size_penalty = float(fspec.get("size_penalty", 0.0))
        self.seed_files = [(self.path.parent / s).resolve() for s in spec.get("seeds", [])]
        self.output = (self.path.parent / spec.get("output", "evo_out")).resolve()

        self._key_hash = _hash({"scenario": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                                "resolved": load_yaml(scenario), "code": code_hash(), "overrides": overrides,
                                "genome": {k: v for k, v in gspec.items() if k in
                                           ("voxel", "tissues", "muscle", "subdivide", "friction")}})
        self._log = self.output / "evaluations.jsonl"
        self._done = self._load_log()
        self._pool = None

    # -- evaluation ----------------------------------------------------------------------------------------------
    def _load_log(self) -> dict[str, dict[str, Any]]:
        done = {}
        if self._log.is_file():
            for line in self._log.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("spec") == self._key_hash:
                    done[entry["key"]] = entry["summary"]
        return done

    def _ensure(self, genomes: list[Genome]) -> None:
        todo: dict[str, Genome] = {}
        for g in genomes:
            if g.key not in self._done:
                todo.setdefault(g.key, g)
        if not todo:
            return
        tasks = [(self.builder.scenario(g, f"creature_{k}"), self.base_dir) for k, g in todo.items()]
        if self._pool is None:
            results = [_run_summary(t) for t in tasks]
        else:
            from concurrent.futures.process import BrokenProcessPool

            try:
                results = list(self._pool.map(_run_summary, tasks))
            except BrokenProcessPool:  # a native crash: one fresh process per creature
                self._pool.shutdown(cancel_futures=True)
                self._pool = _process_pool(self.jobs)
                results = []
                for t in tasks:
                    with _process_pool(1) as single:
                        try:
                            results.append(single.submit(_run_summary, t).result())
                        except BrokenProcessPool:
                            results.append({"error": "worker process crashed (native error in the backend)"})
        self.output.mkdir(parents=True, exist_ok=True)
        with self._log.open("a") as f:
            for (k, g), summary in zip(todo.items(), results):
                self.n_simulated += 1
                self._done[k] = summary
                f.write(json.dumps({"spec": self._key_hash, "key": k, "genome": g.to_dict(), "summary": summary},
                                   default=float) + "\n")

    def fitness(self, g: Genome) -> float:
        summary = self._done[g.key]
        if "error" in summary:
            return -math.inf
        if self.metric not in summary:
            raise ConfigError(f"fitness metric {self.metric!r} is not in the scenario's summary; available: "
                              f"{', '.join(sorted(summary))}")
        value = float(summary[self.metric])
        return value - self.size_penalty * g.n_cells if math.isfinite(value) else -math.inf

    # -- the loop ------------------------------------------------------------------------------------------------
    def _seed_genomes(self) -> list[Genome]:
        return [genome_from_scenario(load_yaml(p), self.builder) for p in self.seed_files]

    def _select(self, ranked: list[Genome], rng: np.random.Generator) -> Genome:
        """Tournament on rank (``ranked`` is best first)."""
        return ranked[int(rng.integers(len(ranked), size=self.tournament).min())]

    def _rank(self, genomes: list[Genome]) -> list[Genome]:
        unique = list({g.key: g for g in genomes}.values())
        return sorted(unique, key=lambda g: (-self.fitness(g), g.key))

    def _record(self, generation: int, ranked: list[Genome]) -> None:
        for rank, g in enumerate(ranked):
            s = self._done[g.key]
            self.history.append({"generation": generation, "rank": rank, "key": g.key,
                                 "parents": ",".join(g.parents), "fitness": self.fitness(g), "cells": g.n_cells,
                                 "muscle_groups": len(g.muscle_groups(self.space)), "frequency": g.frequency,
                                 "amplitude": g.amplitude, "error": s.get("error", ""),
                                 **{k: v for k, v in s.items() if k != "error"}})

    def run(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        if self.jobs > 1:
            self._pool = _process_pool(self.jobs)
        try:
            pop = self._seed_genomes()[: self.population]
            while len(pop) < self.population:
                pop.append(random_genome(self.space, rng))
            self._ensure(pop)
            ranked = self._rank(pop)[: self.population]
            self._record(0, ranked)
            self._progress(0, ranked)
            for gen in range(1, self.generations + 1):
                children = []
                for _ in range(self.offspring):
                    a = self._select(ranked, rng)
                    if rng.random() < self.p_crossover and len(ranked) > 1:
                        a = crossover(a, self._select(ranked, rng), self.space, rng)
                    children.append(mutate(a, self.space, rng))
                self._ensure(children)
                ranked = self._rank(ranked + children)[: self.population]
                self._record(gen, ranked)
                self._progress(gen, ranked)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        return self._write(ranked)

    def check(self, top: int = 3, tolerance: float = 0.25) -> dict[str, Any]:
        """Exploit check: re-run the ``top`` creatures of the finished run under perturbed numerics and contact
        (each voxel split in 2^3, half the time step, ground friction x0.8 and x1.2). A gait whose fitness metric
        moves by more than ``tolerance`` (relative) under any of them is flagged: it may live on a numerical
        artefact (contact slip, a stiff explicit force) rather than on the body. Writes ``check.yaml``."""
        from softsim.config.loader import apply_overrides

        report = load_yaml(self.output / "report.yaml")
        base = self.builder.base
        dt = float(parse_quantity(base["simulation"]["timestep"], "time"))
        mu = self.builder.friction
        variants = {"subdivide_2": None, "half_timestep": [f"simulation.timestep={dt / 2}"],
                    "friction_x0.8": 0.8, "friction_x1.2": 1.2}
        tasks, index = [], []
        for b in report["best"][:top]:
            raw = load_yaml(self.output / b["file"])
            for name, ov in variants.items():
                v = copy.deepcopy(raw)
                if ov is None:
                    v["body"]["geometry"]["subdivide"] = 2 * int(v["body"]["geometry"].get("subdivide", 1))
                elif isinstance(ov, float):
                    if mu is None:
                        continue
                    v["ground"]["friction"]["sliding"] = ov * mu  # ground, body and every zone together
                    for m in [v["body"]["material"], *(z["material"] for z in v["body"].get("zones", []))]:
                        m["friction"]["sliding"] = ov * mu
                else:
                    v = apply_overrides(v, ov)
                tasks.append((v, self.base_dir))
                index.append((b, name))
        if self.jobs > 1:
            with _process_pool(self.jobs) as pool:
                results = list(pool.map(_run_summary, tasks))
        else:
            results = [_run_summary(t) for t in tasks]
        out: dict[str, Any] = {"metric": self.metric, "tolerance": tolerance, "creatures": []}
        rows: dict[str, dict] = {}
        for (b, name), summary in zip(index, results):
            row = rows.setdefault(b["key"], {"key": b["key"], "fitness": b["fitness"], "variants": {},
                                             "robust": True})
            value = summary.get(self.metric, float("nan")) if "error" not in summary else float("nan")
            change = (value - b["fitness"]) / abs(b["fitness"]) if b["fitness"] else float("nan")
            row["variants"][name] = {"value": float(value), "change": float(change),
                                     **({"error": summary["error"]} if "error" in summary else {})}
            if not (abs(change) <= tolerance):
                row["robust"] = False
        out["creatures"] = list(rows.values())
        (self.output / "check.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
        return out

    def _progress(self, gen: int, ranked: list[Genome]) -> None:
        fit = [self.fitness(g) for g in ranked]
        finite = [f for f in fit if math.isfinite(f)]
        median = float(np.median(finite)) if finite else float("nan")
        print(f"generation {gen:3d}: best {fit[0]:.4g}  median {median:.4g}  "
              f"({self.n_simulated} simulated, {len(self._done)} known)", flush=True)

    def _write(self, ranked: list[Genome], top: int = 5) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        save_table(pd.DataFrame(self.history), self.output / "history.parquet")
        best_dir = self.output / "best"
        best_dir.mkdir(exist_ok=True)
        for old in best_dir.glob("rank*.yaml"):
            old.unlink()
        best = []
        for rank, g in enumerate(ranked[:top]):
            raw = self.builder.scenario(g, f"creature_{g.key}")
            raw["description"] = (f"Evolved creature {g.key} (rank {rank}, {self.metric} {self.fitness(g):.4g}) "
                                  f"from {self.path.name}")
            f = best_dir / f"rank{rank}_{g.key}.yaml"
            f.write_text(f"# evolved by squishlings evolve {self.path.name}: rank {rank}, {self.metric} "
                         f"{self.fitness(g):.6g}, generation {self.history[-1]['generation']}\n"
                         f"# run with: softsim run {f.name}   (material paths, if any, are relative to {self.base_dir})\n"
                         + yaml.safe_dump(raw, sort_keys=False))
            best.append({"key": g.key, "fitness": self.fitness(g), "file": str(f.relative_to(self.output))})
        failed = sum("error" in s for s in self._done.values())
        report = {"job": str(self.path), "metric": self.metric, "size_penalty": self.size_penalty,
                  "generations": int(self.history[-1]["generation"]) if self.history else 0,
                  "evaluated": len(self._done), "simulated_this_run": self.n_simulated, "failed": failed,
                  "best": best}
        (self.output / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        return report
