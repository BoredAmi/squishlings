"""Small job helpers: one simulation per worker process, and content hashes for the evaluation cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_CODE_HASH: str | None = None


def run_summary(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Simulate one creature scenario (in a worker process) and return its summary, or ``{"error": ...}``."""
    from softsim.runner import Simulation

    raw, base_dir = task
    try:
        sim = Simulation(raw, base_dir=base_dir)
        try:
            summary = sim.run().summary
        finally:
            sim.close()
        return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.floating))}
    except Exception as exc:  # noqa: BLE001 - an unstable or invalid creature gets the worst fitness
        return {"error": f"{type(exc).__name__}: {exc}"}


def process_pool(workers: int):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_hash() -> str:
    """softsim's code hash combined with squishlings' own source: either changing makes cached results stale."""
    global _CODE_HASH
    if _CODE_HASH is None:
        from softsim.bench import code_hash as softsim_hash

        h = hashlib.sha256(softsim_hash().encode())
        root = Path(__file__).resolve().parent
        for f in sorted(root.rglob("*")):
            if f.suffix in (".py", ".yaml") and "__pycache__" not in f.parts:
                h.update(str(f.relative_to(root)).encode())
                h.update(f.read_bytes())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH
