"""Voxel-creature genomes: a labelled grid plus one oscillator per muscle group.

A genome is a direct encoding on a fixed box of ``shape = (nx, ny, nz)`` cells. Each cell holds a code:
``0`` empty, ``1..n_tissues`` a passive tissue (a material card), then ``n_tissues+1 ..`` a muscle group (the
muscle material, actuated). Muscle group ``g`` has its own phase (rad) and fibre direction (an index into
:data:`DIRECTIONS`); the frequency and stress amplitude are shared by the creature and may also evolve.

Every genome the operators return is **repaired**: only the largest face-connected component is kept, shifted down
so it stands on the ground (z layer 0), and grown back to ``min_cells`` if it shrank below. So the body always
meshes into one bonded solid. Everything random comes from the ``numpy.random.Generator`` passed in: a run is
deterministic for a seed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

DIRECTIONS = ("x", "y", "z", "isotropic")
_FACE = ndimage.generate_binary_structure(3, 1)  # 6-connectivity: cubes bonded only through faces


@dataclass(frozen=True)
class GenomeSpace:
    """What a genome may contain and how it mutates (the evolution job's ``genome:`` block)."""

    shape: tuple[int, int, int] = (6, 6, 4)
    n_tissues: int = 2
    n_muscles: int = 4
    min_cells: int = 4
    max_cells: int | None = None
    fill: float = 0.5                          # occupancy of random genomes
    muscle_fraction: float = 0.4               # share of occupied random cells that are muscle
    frequency: tuple[float, float] = (2.0, 2.0)   # [low, high] Hz; equal = fixed
    amplitude: tuple[float, float] = (8e3, 8e3)   # [low, high] Pa; equal = fixed
    cell_rate: float = 0.05                    # per-cell mutation probability (at least one cell changes)
    phase_sigma: float = 0.6                   # rad, Gaussian phase step
    direction_rate: float = 0.1
    scalar_sigma: float = 0.1                  # relative step of frequency / amplitude

    @property
    def n_codes(self) -> int:
        return 1 + self.n_tissues + self.n_muscles

    def is_muscle(self, code: np.ndarray | int) -> np.ndarray | bool:
        return np.asarray(code) > self.n_tissues

    def group(self, code: int) -> int:
        """Muscle group index (0-based) of a muscle code."""
        return int(code) - self.n_tissues - 1


@dataclass(frozen=True, eq=False)
class Genome:
    cells: np.ndarray                 # (nx, ny, nz) uint8 codes
    phases: np.ndarray                # (n_muscles,) rad in [0, 2pi)
    directions: np.ndarray            # (n_muscles,) indices into DIRECTIONS
    frequency: float
    amplitude: float
    parents: tuple[str, ...] = field(default=())

    @property
    def key(self) -> str:
        """Content hash (the cache key): equal bodies and signals give the same key, whatever their history."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.cells, dtype=np.uint8).tobytes())
        h.update(str(self.cells.shape).encode())
        h.update(np.round(self.phases, 9).tobytes())
        h.update(np.asarray(self.directions, dtype=np.int64).tobytes())
        h.update(f"{self.frequency:.9g},{self.amplitude:.9g}".encode())
        return h.hexdigest()[:16]

    @property
    def n_cells(self) -> int:
        return int(np.count_nonzero(self.cells))

    def muscle_groups(self, space: GenomeSpace) -> list[int]:
        """Muscle groups present in the body (others have no effect on the creature)."""
        codes = np.unique(self.cells[space.is_muscle(self.cells)])
        return [space.group(c) for c in codes]

    def to_dict(self) -> dict:
        return {"cells": self.cells.tolist(), "phases": [float(p) for p in self.phases],
                "directions": [DIRECTIONS[int(d)] for d in self.directions],
                "frequency": float(self.frequency), "amplitude": float(self.amplitude),
                "parents": list(self.parents)}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(np.asarray(d["cells"], dtype=np.uint8), np.asarray(d["phases"], dtype=float),
                   np.array([DIRECTIONS.index(x) for x in d["directions"]]), float(d["frequency"]),
                   float(d["amplitude"]), tuple(d.get("parents", ())))


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


def random_genome(space: GenomeSpace, rng: np.random.Generator) -> Genome:
    occupied = rng.random(space.shape) < space.fill
    muscle = rng.random(space.shape) < space.muscle_fraction
    tissue = rng.integers(1, space.n_tissues + 1, space.shape)
    group = rng.integers(space.n_tissues + 1, space.n_codes, space.shape)
    cells = np.where(occupied, np.where(muscle, group, tissue), 0).astype(np.uint8)
    g = Genome(cells, rng.uniform(0, 2 * math.pi, space.n_muscles), rng.integers(0, len(DIRECTIONS), space.n_muscles),
               _uniform(rng, space.frequency), _uniform(rng, space.amplitude))
    return repair(g, space, rng)


def repair(g: Genome, space: GenomeSpace, rng: np.random.Generator) -> Genome:
    """Keep the largest face-connected component, put it on the ground, and enforce the cell-count limits."""
    cells = g.cells.copy()
    labels, n = ndimage.label(cells > 0, structure=_FACE)
    if n > 1:
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        cells[labels != 1 + int(np.argmax(sizes))] = 0
    if not cells.any():  # nothing left: seed one tissue cell in the middle of the floor
        cells[space.shape[0] // 2, space.shape[1] // 2, 0] = 1
    z = np.nonzero(cells.any(axis=(0, 1)))[0]
    if z[0] > 0:
        cells = np.roll(cells, -int(z[0]), axis=2)
    if space.max_cells is not None:
        while np.count_nonzero(cells) > space.max_cells:
            cells = _remove_cell(cells, rng)
    while np.count_nonzero(cells) < min(space.min_cells, cells.size):
        cells = _add_cell(cells, space, rng)
    return replace(g, cells=cells.astype(np.uint8))


def _neighbours_empty(cells: np.ndarray) -> np.ndarray:
    """Empty cells sharing a face with the body (where a new cell stays connected)."""
    return ndimage.binary_dilation(cells > 0, structure=_FACE) & (cells == 0)


def _add_cell(cells: np.ndarray, space: GenomeSpace, rng: np.random.Generator) -> np.ndarray:
    free = np.argwhere(_neighbours_empty(cells))
    if len(free) == 0:
        return cells
    cells = cells.copy()
    cells[tuple(free[rng.integers(len(free))])] = rng.integers(1, space.n_codes)
    return cells


def _remove_cell(cells: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove a random surface cell whose removal keeps the body connected (any cell if none does)."""
    occupied = np.argwhere(cells > 0)
    for i in rng.permutation(len(occupied)):
        trial = cells.copy()
        trial[tuple(occupied[i])] = 0
        if trial.any() and ndimage.label(trial > 0, structure=_FACE)[1] == 1:
            return trial
    trial = cells.copy()
    trial[tuple(occupied[rng.integers(len(occupied))])] = 0
    return trial


def mutate(g: Genome, space: GenomeSpace, rng: np.random.Generator) -> Genome:
    """Per-cell code changes (at least one), plus phase, direction and scalar steps; returns a repaired child."""
    cells = g.cells.copy()
    candidates = (cells > 0) | _neighbours_empty(cells)       # mutate the body and its skin, not the far void
    idx = np.argwhere(candidates)
    hit = rng.random(len(idx)) < space.cell_rate
    if not hit.any():
        hit[rng.integers(len(idx))] = True
    for i, j, k in idx[hit]:
        current = cells[i, j, k]
        choices = [c for c in range(space.n_codes) if c != current]
        cells[i, j, k] = choices[rng.integers(len(choices))]
    phases = np.mod(g.phases + rng.normal(0.0, space.phase_sigma, g.phases.shape), 2 * math.pi)
    flip = rng.random(g.directions.shape) < space.direction_rate
    directions = np.where(flip, rng.integers(0, len(DIRECTIONS), g.directions.shape), g.directions)
    frequency = _step(g.frequency, space.frequency, space.scalar_sigma, rng)
    amplitude = _step(g.amplitude, space.amplitude, space.scalar_sigma, rng)
    child = Genome(cells, phases, directions, frequency, amplitude, (g.key,))
    return repair(child, space, rng)


def _step(value: float, bounds: tuple[float, float], sigma: float, rng: np.random.Generator) -> float:
    lo, hi = bounds
    if lo == hi:
        return float(lo)
    return float(np.clip(value * math.exp(rng.normal(0.0, sigma)), lo, hi))


def crossover(a: Genome, b: Genome, space: GenomeSpace, rng: np.random.Generator) -> Genome:
    """One cut plane across a random axis: cells below it from ``a``, above from ``b``; each muscle group's
    oscillator from either parent at random; scalars averaged. Returns a repaired child."""
    axis = int(rng.integers(3))
    cut = int(rng.integers(1, space.shape[axis])) if space.shape[axis] > 1 else 0
    take_a = (np.arange(space.shape[axis]) < cut).reshape([-1 if d == axis else 1 for d in range(3)])
    cells = np.where(take_a, a.cells, b.cells)
    pick = rng.random(space.n_muscles) < 0.5
    child = Genome(cells, np.where(pick, a.phases, b.phases), np.where(pick, a.directions, b.directions),
                   0.5 * (a.frequency + b.frequency), 0.5 * (a.amplitude + b.amplitude), (a.key, b.key))
    return repair(child, space, rng)


def crop(cells: np.ndarray) -> np.ndarray:
    """The grid cut to the body's bounding box (z keeps its ground layer)."""
    occ = np.argwhere(cells > 0)
    lo, hi = occ.min(axis=0), occ.max(axis=0) + 1
    lo[2] = 0
    return cells[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
