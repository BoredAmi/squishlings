"""Locomotion: a free soft body on the ground that moves itself (e.g. a muscle-driven voxel creature).

The config is the ``actuation`` scenario's (any shape, material zones, actuators), without the pressure actuator
default and with ``settle`` [s]: the time the body is given to come to rest on the ground; the gait is measured from
there (a muscle's ``oscillator.start`` normally equals it). Gravity and a frictional ground are needed to move.

Metrics, on top of the actuation scenario's: the body's centre of mass ``com_x/y/z`` (volume-weighted vertices:
lumped tet volumes, so material zones of another density are weighted as the body's), and its rigid orientation
(the best-fit rotation of the vertices from their rest shape, Kabsch): ``tilt`` (angle of the body's z axis from
vertical) and ``yaw`` (heading of its x axis).

Summary: ``distance`` (planar displacement of the centre of mass from ``settle`` to the end) and its ``distance_x``,
``distance_y``; ``mean_speed`` (distance per second of gait); ``path_length`` (planar path of the centre of mass,
from the output samples); ``max_tilt``, ``final_tilt``, ``turn`` (yaw change over the gait); ``flipped`` (1 when
the body turned over, tilt > 90 deg, at any time); ``footprint`` (the largest horizontal distance of a rest vertex
from the rest centre of mass at ``rest_com_x/y``: terrain keeps a square of that half-size clear, :mod:`terrain`).
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from softsim.backends.base import SimState
from softsim.config.units import Time
from softsim.scenarios.inflation import InflatedBodyConfig, InflationConfig, InflationScenario
from softsim.scene import Scene

_ACTUATORS = InflationConfig.model_fields["actuators"].annotation


class LocomotionBodyConfig(InflatedBodyConfig):
    """The body; free on the ground (``fixed`` is allowed but would anchor it)."""


class LocomotionConfig(InflationConfig):
    scenario: Literal["locomotion"] = "locomotion"  # type: ignore[assignment]
    body: LocomotionBodyConfig
    actuators: _ACTUATORS = []  # type: ignore[valid-type]
    settle: Annotated[Time, Field(ge=0)] = 0.3


def _kabsch(rest: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum w |R (rest - c0) - (x - c)|^2."""
    a = (rest - w @ rest) * w[:, None]
    b = x - w @ x
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return (u @ np.diag([1.0, 1.0, d]) @ vt).T


class LocomotionScenario(InflationScenario):
    """A free soft body moving on the ground; see module docstring."""

    name: ClassVar[str] = "locomotion"
    config_model: ClassVar[type] = LocomotionConfig

    config: LocomotionConfig

    def build_scene(self) -> Scene:
        scene = super().build_scene()
        body = scene.soft_bodies[0]
        vol = np.zeros(body.mesh.n_points)
        np.add.at(vol, body.mesh.tets.ravel(), np.repeat(body.mesh.tet_volumes() / 4.0, 4))
        self._weights = vol / vol.sum()
        self._rest = body.mesh.points.copy()
        return scene

    def metrics(self, state: SimState) -> dict[str, float]:
        out = super().metrics(state)
        x = state.soft[self.config.body.name].vertices
        com = self._weights @ x
        R = _kabsch(self._rest, x, self._weights)
        out.update(com_x=float(com[0]), com_y=float(com[1]), com_z=float(com[2]),
                   tilt=float(np.arccos(np.clip(R[2, 2], -1.0, 1.0))), yaw=float(np.arctan2(R[1, 0], R[0, 0])))
        return out

    def summarize(self, ts: pd.DataFrame) -> dict[str, float]:
        out = super().summarize(ts)
        gait = ts[ts["t"] >= self.config.settle - 1e-12]
        if len(gait) < 2:
            gait = ts.iloc[-2:]
        xy = gait[["com_x", "com_y"]].to_numpy()
        d = xy[-1] - xy[0]
        duration = float(gait["t"].iloc[-1] - gait["t"].iloc[0])
        yaw = np.unwrap(gait["yaw"].to_numpy())
        out.update(distance=float(np.hypot(*d)), distance_x=float(d[0]), distance_y=float(d[1]),
                   mean_speed=float(np.hypot(*d) / duration) if duration > 0 else 0.0,
                   path_length=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                   max_tilt=float(ts["tilt"].max()), final_tilt=float(ts["tilt"].iloc[-1]),
                   turn=float(yaw[-1] - yaw[0]), flipped=float(ts["tilt"].max() > np.pi / 2))
        c0 = self._weights @ self._rest
        out.update(rest_com_x=float(c0[0]), rest_com_y=float(c0[1]),
                   footprint=float(np.linalg.norm(self._rest[:, :2] - c0[:2], axis=1).max()))
        return out

