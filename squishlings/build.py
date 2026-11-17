"""Genome <-> scenario: a creature as a ``voxels`` body with tissue zones and muscle actuators, and back.

Labels: tissue ``k`` (1-based) is the character ``"abcdefgh"[k-1]``, muscle group ``g`` (0-based) is
``"123456789"[g]``; ``.`` is empty. Tissue ``a`` is the body's own material, the other labels are material zones
(``region: <label>``), and each muscle group present is one ``type: muscle`` actuator on its label with the
creature's oscillator (``start`` delays it so the body settles first).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError

from squishlings.genome import DIRECTIONS, Genome, GenomeSpace, crop, repair

TISSUE_CHARS = "abcdefgh"
MUSCLE_CHARS = "123456789"
_DIRECTION_VECTORS = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


@dataclass(frozen=True)
class CreatureBuilder:
    """Turns a genome into a scenario config (a plain mapping, ready for :class:`softsim.runner.Simulation`)."""

    base: dict[str, Any]
    voxel: float
    tissues: tuple[Any, ...]
    muscle_material: Any
    start: float
    space: GenomeSpace
    subdivide: int = 1
    friction: float | None = None   # one sliding friction for every tissue and the ground (MuJoCo takes the max)

    def material(self, card: Any) -> Any:
        if self.friction is None:
            return card
        ref = dict(card) if isinstance(card, dict) else {"preset": card}
        return {**ref, "friction": {**ref.get("friction", {}), "sliding": self.friction}}

    def label(self, code: int) -> str:
        if code == 0:
            return "."
        if code <= self.space.n_tissues:
            return TISSUE_CHARS[code - 1]
        return MUSCLE_CHARS[self.space.group(code)]

    def cells(self, g: Genome) -> list[list[str]]:
        """``cells`` of the voxels shape: z layers bottom-up, rows indexed by y, characters by x."""
        grid = crop(g.cells)
        return [["".join(self.label(int(grid[x, y, z])) for x in range(grid.shape[0])) for y in range(grid.shape[1])]
                for z in range(grid.shape[2])]

    def scenario(self, g: Genome, name: str | None = None) -> dict[str, Any]:
        raw = copy.deepcopy(self.base)
        body = raw.setdefault("body", {})
        body["geometry"] = {"type": "voxels", "voxel": self.voxel, "cells": self.cells(g), "subdivide": self.subdivide}
        body["material"] = self.material(self.tissues[0])
        if self.friction is not None:
            raw.setdefault("ground", {}).setdefault("friction", {})["sliding"] = self.friction
        body.pop("fixed", None)
        present = sorted({int(c) for c in np.unique(g.cells) if c})
        zones, actuators = [], []
        for code in present:
            lab = self.label(code)
            if code <= self.space.n_tissues:
                if code > 1:
                    zones.append({"name": f"tissue_{lab}", "material": self.material(self.tissues[code - 1]),
                                  "region": lab})
                continue
            grp = self.space.group(code)
            if self.muscle_material != self.tissues[0]:
                zones.append({"name": f"muscle_{lab}", "material": self.material(self.muscle_material),
                                  "region": lab})
            d = DIRECTIONS[int(g.directions[grp])]
            actuators.append({"name": f"muscle_{lab}", "type": "muscle", "region": lab,
                              "direction": "isotropic" if d == "isotropic" else _DIRECTION_VECTORS[d],
                              "oscillator": {"frequency": float(g.frequency), "amplitude": float(g.amplitude),
                                             "phase": float(g.phases[grp]), "start": self.start}})
        body["zones"] = zones
        raw["actuators"] = actuators
        if name:
            raw["name"] = name
        return raw


def genome_from_scenario(raw: dict[str, Any], builder: CreatureBuilder) -> Genome:
    """The genome of a hand-made creature scenario (a ``voxels`` body using the job's labels), placed at the centre
    of the genome box. Muscle groups without an actuator keep phase 0 and direction x; the oscillator's frequency
    and amplitude come from the first muscle (clipped to the job's ranges)."""
    space = builder.space
    geo = raw.get("body", {}).get("geometry", {})
    if geo.get("type") != "voxels":
        raise ConfigError("a seed creature needs a body with geometry type 'voxels'")
    layers = geo["cells"]
    nz, ny, nx = len(layers), len(layers[0]), len(layers[0][0])
    if nx > space.shape[0] or ny > space.shape[1] or nz > space.shape[2]:
        raise ConfigError(f"seed creature {nx}x{ny}x{nz} does not fit the genome box {space.shape}")
    codes = {".": 0, **{c: i + 1 for i, c in enumerate(TISSUE_CHARS[:space.n_tissues])},
             **{c: space.n_tissues + 1 + g for g, c in enumerate(MUSCLE_CHARS[:space.n_muscles])}}
    cells = np.zeros(space.shape, dtype=np.uint8)
    ox, oy = (space.shape[0] - nx) // 2, (space.shape[1] - ny) // 2
    for z, layer in enumerate(layers):
        for y, row in enumerate(layer):
            for x, ch in enumerate(row):
                if ch not in codes:
                    raise ConfigError(f"seed creature label {ch!r} is not one of the job's labels {sorted(codes)}")
                cells[ox + x, oy + y, z] = codes[ch]
    phases = np.zeros(space.n_muscles)
    directions = np.zeros(space.n_muscles, dtype=int)
    frequency, amplitude = space.frequency[0], space.amplitude[0]
    first = True
    for a in raw.get("actuators", []):
        if a.get("type") != "muscle" or a.get("region") not in MUSCLE_CHARS[:space.n_muscles]:
            continue
        g = MUSCLE_CHARS.index(a["region"])
        osc = a.get("oscillator", {})
        phases[g] = math.fmod(float(parse_quantity(osc.get("phase", 0.0), "angle")), 2 * math.pi) % (2 * math.pi)
        d = a.get("direction", "isotropic")
        directions[g] = DIRECTIONS.index("isotropic") if d == "isotropic" else int(np.argmax(np.abs(d)))
        if first:
            frequency = float(np.clip(parse_quantity(osc.get("frequency", frequency), "frequency"), *space.frequency))
            amplitude = float(np.clip(parse_quantity(osc.get("amplitude", amplitude), "pressure"), *space.amplitude))
            first = False
    g = Genome(cells, phases, directions, frequency, amplitude, ("seed",))
    rng = np.random.default_rng(0)
    return repair(g, space, rng)
