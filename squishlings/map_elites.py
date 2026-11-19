"""MAP-Elites (Mouret & Clune 2015): keep the best creature of each *kind* instead of one ranked population.

Each creature gets descriptors (numbers that say what kind of body it is, not how good it is). They place it in
a cell of a grid, the archive, and it takes the cell if the cell is empty or it beats the elite there. Parents
are drawn uniformly from the occupied cells, so every kind of body keeps being improved, and no body plan can
take over the run.

Descriptors come from the genome (exact, no simulation needed):

- sticks: ``limbs`` (terminal segments other than the root, mirrored copies counted), ``segments``,
  ``elongation`` (log2 of body length / width, from the placed capsules: > 0 long, < 0 wide), ``height``
  (vertical extent / horizontal size);
- voxels: ``cells``, ``elongation``, ``height`` (the same, from the cropped grid).

An archive is configured in the job file (``archive:``), each descriptor with a range and a number of bins;
values outside the range go to the end bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# name: (default range, default bins)
DEFAULTS: dict[str, tuple[tuple[float, float], int]] = {
    "limbs": ((0.0, 6.0), 6),        # 0, 1, ..., 4, 5+
    "segments": ((1.0, 11.0), 5),    # 1-2, 3-4, ..., 9-10
    "cells": ((0.0, 40.0), 5),
    "elongation": ((-1.0, 3.0), 5),  # wide ... long (log2 length / width)
    "height": ((0.0, 1.5), 5),
}


def _leaves(node, is_root: bool = True) -> int:
    if not node.children:
        return 0 if is_root else 1
    return sum(_leaves(c, False) * (2 if c.sym else 1) for c in node.children)


def stick_descriptors(g) -> dict[str, float]:
    from squishlings.sticks import place

    segs = place(g)
    pts = np.array([p for s in segs for p in (s.start, s.end)])
    r = np.array([s.radius for s in segs for _ in range(2)])
    lo, hi = (pts - r[:, None]).min(axis=0), (pts + r[:, None]).max(axis=0)
    ext = hi - lo
    return {"limbs": float(_leaves(g.root)), "segments": float(g.n_segments),
            "elongation": math.log2(ext[0] / ext[1]), "height": float(ext[2] / max(ext[0], ext[1]))}


def voxel_descriptors(g) -> dict[str, float]:
    from squishlings.genome import crop

    c = crop(g.cells)
    nx, ny, nz = c.shape
    return {"cells": float(g.n_cells), "elongation": math.log2(nx / ny), "height": float(nz / max(nx, ny))}


@dataclass
class Axis:
    name: str
    lo: float
    hi: float
    bins: int

    def index(self, v: float) -> int:
        return int(np.clip(math.floor((v - self.lo) / (self.hi - self.lo) * self.bins), 0, self.bins - 1))

    def edges(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.bins + 1)


def axes_from_spec(spec: dict[str, Any], available: set[str]) -> list[Axis]:
    """``archive: {descriptors: [limbs, elongation], bins: [6, 5], ranges: [[0, 6], [-1, 3]]}`` (bins and
    ranges optional: :data:`DEFAULTS`)."""
    from softsim.errors import ConfigError

    names = list(spec.get("descriptors", ["limbs", "elongation"]))
    if not 1 <= len(names) <= 3:
        raise ConfigError("archive.descriptors: 1 to 3 names")
    for n in names:
        if n not in available:
            raise ConfigError(f"archive.descriptors: {n!r} is not a descriptor of this encoding; have "
                              f"{sorted(available)}")
    bins = spec.get("bins") or [DEFAULTS[n][1] for n in names]
    ranges = spec.get("ranges") or [DEFAULTS[n][0] for n in names]
    if len(bins) != len(names) or len(ranges) != len(names):
        raise ConfigError("archive: one bins and one range per descriptor")
    axes = [Axis(n, float(r[0]), float(r[1]), int(b)) for n, r, b in zip(names, ranges, bins)]
    if any(a.bins < 1 or not a.hi > a.lo for a in axes):
        raise ConfigError("archive: bins >= 1 and range [low, high] with high > low")
    return axes


class Archive:
    """Cells -> (genome, fitness, descriptors). Insertion is deterministic: ties keep the incumbent."""

    def __init__(self, axes: list[Axis], describe: Callable[[Any], dict[str, float]]):
        self.axes, self.describe = axes, describe
        self.cells: dict[tuple[int, ...], tuple[Any, float, dict[str, float]]] = {}

    def cell(self, d: dict[str, float]) -> tuple[int, ...]:
        return tuple(a.index(d[a.name]) for a in self.axes)

    def add(self, g, fitness: float) -> bool:
        if not math.isfinite(fitness):
            return False
        d = self.describe(g)
        c = self.cell(d)
        if c not in self.cells or fitness > self.cells[c][1]:
            self.cells[c] = (g, fitness, d)
            return True
        return False

    def elites(self) -> list[tuple[tuple[int, ...], Any, float, dict[str, float]]]:
        """Occupied cells in cell order (deterministic for the parent draw)."""
        return [(c, *self.cells[c]) for c in sorted(self.cells)]

    @property
    def size(self) -> int:
        return int(np.prod([a.bins for a in self.axes]))

    @property
    def coverage(self) -> float:
        return len(self.cells) / self.size

    @property
    def qd_score(self) -> float:
        """Sum of the elites' fitness (non-negative fitness: distance)."""
        return float(sum(max(f, 0.0) for _, f, _ in self.cells.values()))

    def table(self) -> list[dict[str, Any]]:
        return [{"cell": list(c), **{a.name: d[a.name] for a in self.axes}, "key": g.key, "fitness": f}
                for c, g, f, d in self.elites()]

    def heatmap(self, path, metric: str = "distance", scale: float = 1e3, unit: str = "mm") -> None:
        """2D archives: best fitness per cell (empty cells blank). 1D: a single row."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax0 = self.axes[0]
        ax1 = self.axes[1] if len(self.axes) > 1 else Axis("", 0, 1, 1)
        grid = np.full((ax0.bins, ax1.bins), np.nan)
        for c, (_, f, _) in self.cells.items():  # a third axis is collapsed to its best cell
            i, j = c[0], c[1] if len(c) > 1 else 0
            grid[i, j] = f * scale if np.isnan(grid[i, j]) else max(grid[i, j], f * scale)
        fig, ax = plt.subplots(figsize=(1.3 * ax1.bins + 2.2, 0.9 * ax0.bins + 1.4), dpi=110)
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        for i in range(ax0.bins):
            for j in range(ax1.bins):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if grid[i, j] < np.nanmax(grid) * 0.6 else "black")

        def labels(a: Axis) -> list[str]:
            e = a.edges()
            if a.name in ("limbs",):
                return [f"{e[k]:.0f}" + ("+" if k == a.bins - 1 else "") for k in range(a.bins)]
            return [f"{e[k]:.1f} … {e[k + 1]:.1f}" for k in range(a.bins)]

        ax.set_yticks(range(ax0.bins), labels(ax0))
        ax.set_ylabel(ax0.name)
        if ax1.name:
            ax.set_xticks(range(ax1.bins), labels(ax1), rotation=30)
            ax.set_xlabel(ax1.name + (" (log2 length / width: wide ← → long)" if ax1.name == "elongation" else ""))
        else:
            ax.set_xticks([])
        fig.colorbar(im, ax=ax, label=f"best {metric} [{unit}]")
        ax.set_title(f"MAP-Elites archive: {len(self.cells)}/{self.size} cells, QD-score "
                     f"{self.qd_score * scale:.0f} {unit}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
