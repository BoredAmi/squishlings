"""Evolve soft creatures: ``squishlings evolve jobs/crawler.yaml`` (voxels) or ``jobs/sticks.yaml`` (soft sticks).

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
    fitness: {metric: distance, size_penalty: 0}  # per-cell penalty in the metric's unit; max_height (default
                                                  # 10 cm): ending higher marks a blow-up, the worst fitness
    seeds: [../squishlings/creatures/inchworm.yaml]   # optional hand-made creatures in the first population
    output: ../runs/crawler

Soft sticks (:mod:`squishlings.sticks`): ``genome: {encoding: sticks, ...}`` with ``tissues``, ``muscle`` (as
above, without ``material``: a muscle segment is made of its own tissue), ``max_segments``, ``max_depth``,
``length`` and ``radius`` ([low, high]), ``mesh_size`` (default 3 mm), ``mesh_timeout`` (s), ``muscle_rate``,
``phase_sigma``, ``scalar_sigma``, ``angle_sigma``. Seeds are genome text files; meshes go to
``<output>/meshes/``.

MAP-Elites (``evolution: {method: map_elites, ...}``, :mod:`squishlings.map_elites`): ``population`` creatures
(the seeds, then random) start the archive, then each of ``generations`` batches draws ``offspring`` children
from parents picked uniformly among the archive's elites; a child takes its cell (``archive:`` descriptors and
bins) if the cell is empty or it beats the elite there. ``tournament`` is unused. Outputs also
``archive.png`` (best fitness per cell), ``elites/`` (every elite as a runnable scenario) and the archive in
``report.yaml`` (coverage, QD-score, cells). ``best/`` holds the fittest elites, which are of different kinds.

A language model as mutation operator (soft sticks with MAP-Elites, :mod:`squishlings.llm`)::

    evolution: {method: map_elites, ..., llm: {model: "qwen3.5:9b", rate: 0.5, examples: 3}}

With probability ``rate`` a child is the model's edit of its parent (shown the parent, its distance and kind, and
the ``examples`` fittest other elites) instead of crossover + random mutation; an answer that does not parse or
changes nothing falls back to random mutation. Needs a running ollama (``url``, default http://localhost:11434).
Answers are cached in ``<output>/llm.jsonl``, so a stopped run resumes without asking again. The report adds how
many children came from the model and how many of them entered the archive.

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
from squishlings import genome as voxel_ops
from squishlings import sticks as stick_ops
from squishlings.genome import Genome, GenomeSpace
from squishlings.sticks import StickGenome, StickSpace

_JOB_FIELDS = {"scenario", "backend", "overrides", "genome", "evolution", "fitness", "seeds", "output", "archive"}
_GENOME_FIELDS = {"friction", "shape", "voxel", "tissues", "muscle", "min_cells", "max_cells", "fill", "muscle_fraction",
                  "subdivide", "cell_rate", "phase_sigma", "direction_rate", "scalar_sigma"}
_MUSCLE_FIELDS = {"material", "groups", "frequency", "amplitude", "start"}
_STICK_FIELDS = {"encoding", "tissues", "muscle", "max_segments", "max_depth", "length", "radius", "mesh_size",
                 "mesh_timeout", "muscle_rate", "phase_sigma", "scalar_sigma", "angle_sigma"}
_EVO_FIELDS = {"method", "population", "offspring", "generations", "tournament", "crossover", "seed", "llm"}
_LLM_FIELDS = {"model", "rate", "url", "temperature", "examples", "think", "timeout"}


def _call(pair: tuple) -> Any:
    """``fn(arg)`` in a worker process (the exploit check mixes workers)."""
    fn, arg = pair
    return fn(arg)


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


def _with_friction(raw: dict[str, Any], mu: float) -> dict[str, Any]:
    """One sliding friction on the ground, the body and every zone (MuJoCo takes the larger of a contact's two)."""
    raw = copy.deepcopy(raw)
    raw.setdefault("ground", {}).setdefault("friction", {})["sliding"] = mu

    def card(m: Any) -> dict:
        ref = dict(m) if isinstance(m, dict) else {"preset": m}
        return {**ref, "friction": {**ref.get("friction", {}), "sliding": mu}}

    raw["body"]["material"] = card(raw["body"]["material"])
    for z in raw["body"].get("zones", []):
        z["material"] = card(z["material"])
    return raw


class _Voxels:
    """The voxel encoding (:mod:`squishlings.genome`, :mod:`squishlings.build`)."""

    worker = staticmethod(_run_summary)

    def __init__(self, space: GenomeSpace, builder: CreatureBuilder, base_dir: str):
        self.space, self.builder, self.base_dir = space, builder, base_dir
        self.friction = builder.friction

    def random(self, rng): return voxel_ops.random_genome(self.space, rng)
    def mutate(self, g, rng): return voxel_ops.mutate(g, self.space, rng)
    def crossover(self, a, b, rng): return voxel_ops.crossover(a, b, self.space, rng)
    def size(self, g: Genome) -> int: return g.n_cells
    def groups(self, g: Genome) -> int: return len(g.muscle_groups(self.space))
    def to_dict(self, g: Genome) -> dict: return g.to_dict()
    def seed(self, path: Path) -> Genome: return genome_from_scenario(load_yaml(path), self.builder)
    descriptors = ("cells", "elongation", "height")

    def describe(self, g: Genome) -> dict[str, float]:
        from squishlings.map_elites import voxel_descriptors

        return voxel_descriptors(g)
    def task(self, g: Genome, name: str): return (self.builder.scenario(g, name), self.base_dir)
    def scenario(self, g: Genome, name: str) -> dict: return self.builder.scenario(g, name)

    def finer(self, best: dict, raw: dict):
        v = copy.deepcopy(raw)
        v["body"]["geometry"]["subdivide"] = 2 * int(v["body"]["geometry"].get("subdivide", 1))
        return "subdivide_2", _run_summary, (v, self.base_dir)


class _Sticks:
    """The soft-stick encoding (:mod:`squishlings.sticks`, :mod:`squishlings.sticks_mesh`)."""

    def __init__(self, space: StickSpace, base: dict, base_dir: str, output: Path, tissues: dict[str, Any],
                 start: float, mesh_size: float, mesh_timeout: float):
        from squishlings.sticks_mesh import run_sticks

        self.worker = run_sticks
        self.space, self.base, self.base_dir, self.output = space, base, base_dir, output
        self.tissues, self.start, self.mesh_size, self.mesh_timeout = tissues, start, mesh_size, mesh_timeout
        self.friction = float(base.get("ground", {}).get("friction", {}).get("sliding", 1.0))

    def random(self, rng): return stick_ops.random_genome(self.space, rng)
    def mutate(self, g, rng): return stick_ops.mutate(g, self.space, rng)
    def crossover(self, a, b, rng): return stick_ops.crossover(a, b, self.space, rng)
    def size(self, g: StickGenome) -> int: return g.n_segments
    def groups(self, g: StickGenome) -> int: return len({n.muscle for n, _, _ in g.root.walk() if n.muscle})
    def to_dict(self, g: StickGenome) -> dict: return {"text": g.text, "parents": list(g.parents)}
    def seed(self, path: Path) -> StickGenome: return stick_ops.from_text(path.read_text(), self.space)
    descriptors = ("limbs", "segments", "elongation", "height")

    def describe(self, g: StickGenome) -> dict[str, float]:
        from squishlings.map_elites import stick_descriptors

        return stick_descriptors(g)

    def _msh(self, key: str, size: float) -> Path:
        tag = "" if size == self.mesh_size else f"_{size:g}mm"
        return self.output / "meshes" / f"{key}{tag}.msh"

    def task(self, g: StickGenome, name: str, size: float | None = None) -> dict:
        from dataclasses import asdict

        size = self.mesh_size if size is None else size
        return {"text": g.text, "space": asdict(self.space), "base": self.base, "tissues": self.tissues,
                "start": self.start, "mesh": str(self._msh(g.key, size)), "mesh_size": size,
                "timeout": self.mesh_timeout, "base_dir": self.base_dir, "name": name}

    def scenario(self, g: StickGenome, name: str) -> dict:
        from squishlings.sticks_mesh import scenario

        msh = self._msh(g.key, self.mesh_size)
        regions = json.loads(msh.with_suffix(".regions.json").read_text())
        raw = scenario(g, msh, stick_ops.place(g), set(regions), self.base, self.tissues, self.start)
        raw["name"] = name
        return raw

    def finer(self, best: dict, raw: dict):
        g = stick_ops.from_text(best["genome"]["text"], self.space)
        size = round(self.mesh_size * 2 / 3, 4)
        return f"mesh_{size:.3g}mm", self.worker, self.task(g, f"creature_{g.key}_fine", size)


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
        self.encoding = str(gspec.get("encoding", "voxels"))
        if self.encoding not in ("voxels", "sticks"):
            raise ConfigError(f"genome.encoding: 'voxels' or 'sticks', got {self.encoding!r}")
        _check_keys(gspec, _STICK_FIELDS if self.encoding == "sticks" else _GENOME_FIELDS | {"encoding"}, "genome")
        mspec = dict(gspec.get("muscle") or {})
        _check_keys(mspec, _MUSCLE_FIELDS, "genome.muscle")
        tissues = tuple(gspec.get("tissues") or [base.get("body", {}).get("material", "ecoflex_00_30")])
        if not 1 <= len(tissues) <= len(TISSUE_CHARS):
            raise ConfigError(f"genome.tissues: 1 to {len(TISSUE_CHARS)} material cards")
        groups = int(mspec.get("groups", 4))
        if not 0 <= groups <= len(MUSCLE_CHARS):
            raise ConfigError(f"genome.muscle.groups: 0 to {len(MUSCLE_CHARS)}")
        self.output = (self.path.parent / spec.get("output", "evo_out")).resolve()
        if self.encoding == "sticks":
            self._init_sticks(gspec, mspec, base, tissues, groups)
        else:
            self._init_voxels(gspec, mspec, base, tissues, groups)
        self._init_rest(spec, scenario, overrides, gspec)

    def _init_sticks(self, gspec: dict, mspec: dict, base: dict, tissues: tuple, groups: int) -> None:
        mm = lambda key, default: tuple(1e3 * v for v in _bounds(gspec.get(key, default), "length"))  # noqa: E731
        rates = {k: float(gspec[k]) for k in ("muscle_rate", "phase_sigma", "scalar_sigma", "angle_sigma")
                 if k in gspec}
        self.space = StickSpace(tissues=tuple(TISSUE_CHARS[: len(tissues)]), n_muscles=groups,
                                max_segments=int(gspec.get("max_segments", 10)),
                                max_depth=int(gspec.get("max_depth", 4)),
                                length=mm("length", ["8 mm", "30 mm"]), radius=mm("radius", ["3 mm", "7 mm"]),
                                frequency=_bounds(mspec.get("frequency", ["1 Hz", "8 Hz"]), "frequency"),
                                amplitude=_bounds(mspec.get("amplitude", ["2 kPa", "15 kPa"]), "pressure"), **rates)
        self.enc = _Sticks(self.space, base, self.base_dir, self.output,
                           {TISSUE_CHARS[i]: card for i, card in enumerate(tissues)},
                           float(parse_quantity(mspec.get("start", "0.3 s"), "time")),
                           1e3 * float(parse_quantity(gspec.get("mesh_size", "3 mm"), "length")),
                           float(gspec.get("mesh_timeout", 60)))

    def _init_voxels(self, gspec: dict, mspec: dict, base: dict, tissues: tuple, groups: int) -> None:
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
        self.enc = _Voxels(self.space, self.builder, self.base_dir)

    def _init_rest(self, spec: dict, scenario: Path, overrides: list, gspec: dict) -> None:
        espec = dict(spec.get("evolution") or {})
        _check_keys(espec, _EVO_FIELDS, "evolution")
        self.population = int(espec.get("population", 24))
        self.offspring = int(espec.get("offspring", self.population))
        self.generations = int(espec.get("generations", 20))
        self.tournament = int(espec.get("tournament", 3))
        self.p_crossover = float(espec.get("crossover", 0.5))
        self.seed = int(espec.get("seed", 0))
        self.method = str(espec.get("method", "mu_plus_lambda"))
        if self.method not in ("mu_plus_lambda", "map_elites"):
            raise ConfigError(f"evolution.method: 'mu_plus_lambda' or 'map_elites', got {self.method!r}")
        self.axes = None
        if self.method == "map_elites":
            from squishlings.map_elites import axes_from_spec

            aspec = dict(spec.get("archive") or {})
            _check_keys(aspec, {"descriptors", "bins", "ranges"}, "archive")
            self.axes = axes_from_spec(aspec, set(self.enc.descriptors))
        elif spec.get("archive"):
            raise ConfigError("archive: only with evolution.method: map_elites")
        self.llm = None
        if espec.get("llm"):
            from squishlings.llm import LLMMutator

            lspec = dict(espec["llm"])
            _check_keys(lspec, _LLM_FIELDS, "evolution.llm")
            if self.encoding != "sticks" or self.method != "map_elites":
                raise ConfigError("evolution.llm: only for soft sticks with method: map_elites")
            if "model" not in lspec:
                raise ConfigError("evolution.llm: needs a 'model' (an ollama model, e.g. qwen3.5:9b)")
            self.llm_rate = float(lspec.get("rate", 0.5))
            self.llm_examples = int(lspec.get("examples", 3))
            if not 0 <= self.llm_rate <= 1 or self.llm_examples < 0:
                raise ConfigError("evolution.llm: rate in [0, 1], examples >= 0")
            self.llm = LLMMutator(str(lspec["model"]), self.enc.space, self.output / "llm.jsonl",
                                  url=str(lspec.get("url", "http://localhost:11434")),
                                  temperature=float(lspec.get("temperature", 0.8)),
                                  think=bool(lspec.get("think", False)),
                                  timeout=float(lspec.get("timeout", 300)))
            self.llm_counts = {"proposed": 0, "used": 0, "into_archive": 0}
        if min(self.population, self.offspring, self.tournament) < 1 or self.generations < 0:
            raise ConfigError("evolution: population, offspring and tournament >= 1, generations >= 0")

        fspec = dict(spec.get("fitness") or {})
        _check_keys(fspec, {"metric", "size_penalty", "max_height"}, "fitness")
        self.metric = str(fspec.get("metric", "distance"))
        self.size_penalty = float(fspec.get("size_penalty", 0.0))
        # a creature still this high at the end was launched by a numerical blow-up, not by a gait (2026-11-19:
        # a 5 cm stick creature ended 17.6 cm up, flipped, "2.7 m" away; no other creature in 1,400 above 6 cm)
        self.max_height = parse_quantity(fspec.get("max_height", "10 cm"), "length")
        self.seed_files = [(self.path.parent / s).resolve() for s in spec.get("seeds", [])]

        self._key_hash = _hash({"scenario": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                                "resolved": load_yaml(scenario), "code": code_hash(), "overrides": overrides,
                                "genome": {k: v for k, v in gspec.items() if k in
                                           ("voxel", "tissues", "muscle", "subdivide", "friction")}
                                if self.encoding == "voxels" else
                                {k: v for k, v in gspec.items() if k in ("encoding", "tissues", "muscle", "mesh_size")}})
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
        tasks = [self.enc.task(g, f"creature_{k}") for k, g in todo.items()]
        worker = self.enc.worker
        if self._pool is None:
            results = [worker(t) for t in tasks]
        else:
            from concurrent.futures.process import BrokenProcessPool

            try:
                results = list(self._pool.map(worker, tasks))
            except BrokenProcessPool:  # a native crash: one fresh process per creature
                self._pool.shutdown(cancel_futures=True)
                self._pool = _process_pool(self.jobs)
                results = []
                for t in tasks:
                    with _process_pool(1) as single:
                        try:
                            results.append(single.submit(worker, t).result())
                        except BrokenProcessPool:
                            results.append({"error": "worker process crashed (native error in the backend)"})
        self.output.mkdir(parents=True, exist_ok=True)
        with self._log.open("a") as f:
            for (k, g), summary in zip(todo.items(), results):
                self.n_simulated += 1
                self._done[k] = summary
                f.write(json.dumps({"spec": self._key_hash, "key": k, "genome": self.enc.to_dict(g), "summary": summary},
                                   default=float) + "\n")

    def fitness(self, g: Genome) -> float:
        summary = self._done[g.key]
        if "error" in summary:
            return -math.inf
        if self.metric not in summary:
            raise ConfigError(f"fitness metric {self.metric!r} is not in the scenario's summary; available: "
                              f"{', '.join(sorted(summary))}")
        if summary.get("final_max_z", 0.0) > self.max_height:
            return -math.inf
        value = float(summary[self.metric])
        return value - self.size_penalty * self.enc.size(g) if math.isfinite(value) else -math.inf

    # -- the loop ------------------------------------------------------------------------------------------------
    def _seed_genomes(self) -> list[Genome]:
        return [self.enc.seed(p) for p in self.seed_files]

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
                                 "parents": ",".join(g.parents), "fitness": self.fitness(g), "cells": self.enc.size(g),
                                 "muscle_groups": self.enc.groups(g), "frequency": g.frequency,
                                 "amplitude": g.amplitude, "error": s.get("error", ""),
                                 **{k: v for k, v in s.items() if k != "error"}})

    def run(self) -> dict[str, Any]:
        if self.method == "map_elites":
            return self._run_map_elites()
        rng = np.random.default_rng(self.seed)
        if self.jobs > 1:
            self._pool = _process_pool(self.jobs)
        try:
            pop = self._seed_genomes()[: self.population]
            while len(pop) < self.population:
                pop.append(self.enc.random(rng))
            self._ensure(pop)
            ranked = self._rank(pop)[: self.population]
            self._record(0, ranked)
            self._progress(0, ranked)
            for gen in range(1, self.generations + 1):
                children = []
                for _ in range(self.offspring):
                    a = self._select(ranked, rng)
                    if rng.random() < self.p_crossover and len(ranked) > 1:
                        a = self.enc.crossover(a, self._select(ranked, rng), rng)
                    children.append(self.enc.mutate(a, rng))
                self._ensure(children)
                ranked = self._rank(ranked + children)[: self.population]
                self._record(gen, ranked)
                self._progress(gen, ranked)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        return self._write(ranked)

    def _run_map_elites(self) -> dict[str, Any]:
        from squishlings.map_elites import Archive

        rng = np.random.default_rng(self.seed)
        archive = Archive(self.axes, self.enc.describe)
        if self.jobs > 1:
            self._pool = _process_pool(self.jobs)
        try:
            pop = self._seed_genomes()
            while len(pop) < self.population:
                pop.append(self.enc.random(rng))
            self._ensure(pop)
            for g in pop:
                archive.add(g, self.fitness(g))
            self._record_archive(0, archive)
            for gen in range(1, self.generations + 1):
                elites = [g for _, g, _, _ in archive.elites()]
                children, from_llm = [], set()
                for _ in range(self.offspring):
                    if not elites:  # nothing valid yet: keep sampling at random
                        children.append(self.enc.random(rng))
                        continue
                    a = elites[int(rng.integers(len(elites)))]
                    if self.llm is not None:  # both draws always made: the random stream does not depend on answers
                        ask, seed = rng.random() < self.llm_rate, int(rng.integers(2**31))
                        if ask:
                            child = self._llm_child(a, archive, seed)
                            if child is not None:
                                children.append(child)
                                from_llm.add(child.key)
                                continue
                    if rng.random() < self.p_crossover and len(elites) > 1:
                        a = self.enc.crossover(a, elites[int(rng.integers(len(elites)))], rng)
                    children.append(self.enc.mutate(a, rng))
                self._ensure(children)
                entered = sum(archive.add(g, self.fitness(g)) and g.key in from_llm for g in children)
                self._record_archive(gen, archive)
                if self.llm is not None:
                    self.llm_counts["used"] += len(from_llm)
                    self.llm_counts["into_archive"] += entered
                    print(f"           llm: {len(from_llm)} children, {entered} into the archive "
                          f"({self.llm.stats})", flush=True)
                if gen % 5 == 0:
                    self._write_archive(archive)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        ranked = sorted((g for _, g, _, _ in archive.elites()), key=lambda g: (-self.fitness(g), g.key))
        return self._write(ranked, archive=archive)

    def _llm_child(self, parent, archive, seed: int):
        others = sorted((e for e in archive.elites() if e[1].key != parent.key), key=lambda e: (-e[2], e[1].key))
        self.llm_counts["proposed"] += 1
        child, _ = self.llm.propose(parent, self.fitness(parent), self.enc.describe(parent),
                                    [(g, f, d) for _, g, f, d in others[:self.llm_examples]], seed)
        return child

    def _record_archive(self, generation: int, archive) -> None:
        ranked = sorted(archive.elites(), key=lambda e: (-e[2], e[1].key))
        start = len(self.history)
        self._record(generation, [g for _, g, _, _ in ranked])
        for row, (cell, _, _, d) in zip(self.history[start:], ranked):
            row["cell"] = ",".join(map(str, cell))
            row.update({f"desc_{k}": v for k, v in d.items()})
        print(f"batch {generation:3d}: {len(archive.cells)}/{archive.size} cells, best {ranked[0][2]:.4g}, "
              f"QD-score {archive.qd_score:.4g}  ({self.n_simulated} simulated, {len(self._done)} known)"
              if ranked else f"batch {generation:3d}: archive empty", flush=True)

    def _write_archive(self, archive) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        archive.heatmap(self.output / "archive.png", self.metric)
        elite_dir = self.output / "elites"
        elite_dir.mkdir(exist_ok=True)
        for old in elite_dir.glob("cell*.yaml"):
            old.unlink()
        for cell, g, f, d in archive.elites():
            raw = self.enc.scenario(g, f"creature_{g.key}")
            raw["description"] = (f"MAP-Elites elite {g.key} of cell {cell} "
                                  f"({', '.join(f'{k} {v:.3g}' for k, v in d.items())}), {self.metric} {f:.4g}")
            name = "cell" + "_".join(map(str, cell)) + f"_{g.key}.yaml"
            (elite_dir / name).write_text(f"# run with: softsim run {name}\n" + yaml.safe_dump(raw, sort_keys=False))
        return {"descriptors": [{"name": a.name, "range": [a.lo, a.hi], "bins": a.bins} for a in archive.axes],
                "coverage": archive.coverage, "qd_score": archive.qd_score, "cells": archive.table()}

    def check(self, top: int = 3, tolerance: float = 0.25) -> dict[str, Any]:
        """Exploit check: re-run the ``top`` creatures of the finished run under perturbed numerics and contact
        (each voxel split in 2^3, or soft sticks meshed at 2/3 the size; half the time step; friction x0.8 and x1.2
        on the ground, the body and every zone). A gait whose fitness metric
        moves by more than ``tolerance`` (relative) under any of them is flagged: it may live on a numerical
        artefact (contact slip, a stiff explicit force) rather than on the body. Writes ``check.yaml``."""
        from softsim.config.loader import apply_overrides

        report = load_yaml(self.output / "report.yaml")
        mu = self.enc.friction
        tasks, index = [], []
        for b in report["best"][:top]:
            raw = load_yaml(self.output / b["file"])
            dt = float(parse_quantity(raw["simulation"]["timestep"], "time"))
            name, fn, task = self.enc.finer(b, raw)
            variants = [(name, fn, task),
                        ("half_timestep", _run_summary,
                         (apply_overrides(raw, [f"simulation.timestep={dt / 2}"]), self.base_dir))]
            if mu is not None:
                variants += [(f"friction_x{k}", _run_summary, (_with_friction(raw, k * mu), self.base_dir))
                             for k in (0.8, 1.2)]
            for name, fn, task in variants:
                tasks.append((fn, task))
                index.append((b, name))
        if self.jobs > 1:
            with _process_pool(self.jobs) as pool:
                results = list(pool.map(_call, tasks))
        else:
            results = [_call(t) for t in tasks]
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

    def _write(self, ranked: list[Genome], top: int = 5, archive=None) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        save_table(pd.DataFrame(self.history), self.output / "history.parquet")
        best_dir = self.output / "best"
        best_dir.mkdir(exist_ok=True)
        for old in best_dir.glob("rank*.yaml"):
            old.unlink()
        best = []
        for rank, g in enumerate(ranked[:top]):
            raw = self.enc.scenario(g, f"creature_{g.key}")
            raw["description"] = (f"Evolved creature {g.key} (rank {rank}, {self.metric} {self.fitness(g):.4g}) "
                                  f"from {self.path.name}")
            f = best_dir / f"rank{rank}_{g.key}.yaml"
            f.write_text(f"# evolved by squishlings evolve {self.path.name}: rank {rank}, {self.metric} "
                         f"{self.fitness(g):.6g}, generation {self.history[-1]['generation']}\n"
                         f"# run with: softsim run {f.name}   (material paths, if any, are relative to {self.base_dir})\n"
                         + yaml.safe_dump(raw, sort_keys=False))
            best.append({"key": g.key, "fitness": self.fitness(g), "file": str(f.relative_to(self.output)),
                         **({"genome": self.enc.to_dict(g)} if self.encoding == "sticks" else {})})
        failed = sum("error" in s for s in self._done.values())
        report = {"job": str(self.path), "metric": self.metric, "size_penalty": self.size_penalty,
                  "generations": int(self.history[-1]["generation"]) if self.history else 0,
                  "evaluated": len(self._done), "simulated_this_run": self.n_simulated, "failed": failed,
                  "rejected_too_high": sum(v.get("final_max_z", 0.0) > self.max_height for v in self._done.values()),
                  "best": best}
        if archive is not None:
            report = {"method": "map_elites", **report, "archive": self._write_archive(archive)}
        if getattr(self, "llm", None) is not None:
            report["llm"] = {"model": self.llm.model, "rate": self.llm_rate, **self.llm_counts,
                             "answers": dict(self.llm.stats)}
        (self.output / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        return report
