"""Soft sticks: a Framsticks-style body encoding (docs/design/soft-sticks.md). Prototype.

A genome is a tree of segments written as indented text, one segment per line, plus an oscillator footer::

    seg len=24 r=6 mat=a
      seg len=14 r=4 mat=a yaw=60 pitch=-35 muscle=1 bend roll=0 sym
        seg len=9 r=3 mat=b pitch=-30
    ---
    freq=4 amp=8000 phases=0,3.142,0,0

Lengths and radii in mm, angles in degrees, amplitude in Pa. The root lies along +x. A child grows from its
parent's far end, turned by ``yaw`` (about the parent's up) and then ``pitch``. ``sym`` adds the mirror image of
the node's subtree across the parent's vertical plane. ``muscle=g`` makes the segment muscle group ``g``: ``bend``
splits it lengthwise into two halves driven in antiphase (it curls in the plane set by ``roll``), otherwise it
contracts along its axis.

The text is the canonical form: :func:`repair` clamps every value to the :class:`StickSpace`, rounds it, and prunes
the tree to the size limits, and the key is the hash of the printed text. Everything random comes from the
``numpy.random.Generator`` passed in.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StickSpace:
    tissues: tuple[str, ...] = ("a", "b")
    n_muscles: int = 4
    max_segments: int = 10                 # after sym expansion
    max_depth: int = 4                     # root = depth 0
    length: tuple[float, float] = (8.0, 30.0)     # mm
    radius: tuple[float, float] = (3.0, 7.0)      # mm
    pitch: tuple[float, float] = (-80.0, 80.0)    # deg
    frequency: tuple[float, float] = (1.0, 8.0)   # Hz
    amplitude: tuple[float, float] = (2e3, 15e3)  # Pa (sticks bend far more than voxels per Pa: spike)
    phase_sigma: float = 0.6
    scalar_sigma: float = 0.15
    angle_sigma: float = 25.0
    muscle_rate: float = 0.5               # share of random segments that are muscle


@dataclass
class Node:
    length: float = 15.0
    radius: float = 4.0
    mat: str = "a"
    yaw: float = 0.0
    pitch: float = 0.0
    muscle: int = 0                        # 0: passive, 1..n_muscles: group
    bend: bool = False
    roll: float = 0.0
    sym: bool = False
    children: list["Node"] = field(default_factory=list)

    def walk(self, depth: int = 0, parent: "Node | None" = None):
        """Pre-order (node, depth, parent)."""
        yield self, depth, parent
        for c in self.children:
            yield from c.walk(depth + 1, self)


def _count(node: Node, is_root: bool = True) -> int:
    n = 1 + sum(_count(c, False) for c in node.children)
    return n * (2 if node.sym and not is_root else 1)


@dataclass(frozen=True, eq=False)
class StickGenome:
    root: Node
    phases: tuple[float, ...]
    frequency: float
    amplitude: float
    parents: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return to_text(self)

    @property
    def key(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    @property
    def n_segments(self) -> int:
        return _count(self.root)


# -- text --------------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.4g}" if abs(x) < 1e4 else f"{x:.0f}"


def to_text(g: StickGenome) -> str:
    lines = []
    for n, depth, _ in g.root.walk():
        parts = [f"seg len={_fmt(n.length)} r={_fmt(n.radius)} mat={n.mat}"]
        if depth:
            if n.yaw:
                parts.append(f"yaw={_fmt(n.yaw)}")
            if n.pitch:
                parts.append(f"pitch={_fmt(n.pitch)}")
        if n.muscle:
            parts.append(f"muscle={n.muscle}")
            if n.bend:
                parts.append("bend")
                if n.roll:
                    parts.append(f"roll={_fmt(n.roll)}")
        if n.sym and depth:
            parts.append("sym")
        lines.append("  " * depth + " ".join(parts))
    lines.append("---")
    lines.append(f"freq={_fmt(g.frequency)} amp={_fmt(g.amplitude)} phases=" + ",".join(f"{p:.3f}" for p in g.phases))
    return "\n".join(lines) + "\n"


class ParseError(ValueError):
    pass


def from_text(text: str, space: StickSpace) -> StickGenome:
    """Parse (lenient: unknown words are errors, missing fields take defaults), then repair."""
    body, _, footer = text.partition("---")
    stack: list[tuple[int, Node]] = []
    root = None
    for raw in body.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        words = raw.split()
        if words[0] != "seg":
            raise ParseError(f"line must start with 'seg': {raw!r}")
        n = Node()
        for w in words[1:]:
            k, eq, v = w.partition("=")
            try:
                if k in ("bend", "sym") and not eq:
                    setattr(n, k, True)
                elif k == "len":
                    n.length = float(v)
                elif k == "r":
                    n.radius = float(v)
                elif k == "mat":
                    n.mat = v
                elif k in ("yaw", "pitch", "roll"):
                    setattr(n, k, float(v))
                elif k == "muscle":
                    n.muscle = int(v)
                else:
                    raise ParseError(f"unknown field {w!r}")
            except ValueError as exc:
                raise ParseError(f"bad value in {w!r}: {exc}") from None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            if root is not None:
                raise ParseError("more than one root segment")
            root = n
        else:
            stack[-1][1].children.append(n)
        stack.append((indent, n))
    if root is None:
        raise ParseError("no segments")
    vals = dict(w.partition("=")[::2] for w in footer.split())
    try:
        phases = tuple(float(p) for p in vals.get("phases", "").split(",") if p)
        freq = float(vals.get("freq", space.frequency[0]))
        amp = float(vals.get("amp", space.amplitude[0]))
    except ValueError as exc:
        raise ParseError(f"bad footer: {exc}") from None
    phases = (phases + (0.0,) * space.n_muscles)[:space.n_muscles]
    return repair(StickGenome(root, phases, freq, amp), space)


# -- repair ------------------------------------------------------------------------------------------------------
def _clip(x: float, b: tuple[float, float]) -> float:
    return float(min(max(x, b[0]), b[1]))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def repair(g: StickGenome, space: StickSpace) -> StickGenome:
    root = copy.deepcopy(g.root)
    for n, depth, _ in list(root.walk()):
        n.length = round(_clip(n.length, space.length), 1)
        n.radius = round(_clip(n.radius, space.radius), 1)
        n.mat = n.mat if n.mat in space.tissues else space.tissues[0]
        n.yaw = 0.0 if depth == 0 else float(round(_wrap(n.yaw)))
        n.pitch = 0.0 if depth == 0 else float(round(_clip(n.pitch, space.pitch)))
        n.muscle = n.muscle if 0 <= n.muscle <= space.n_muscles else 0
        n.bend = n.bend and n.muscle > 0
        n.roll = float(round(_wrap(n.roll))) if n.bend else 0.0
        n.sym = n.sym and depth > 0
        if depth >= space.max_depth:
            n.children = []
    while _count(root) > space.max_segments:          # prune the last deepest leaf (or un-sym it)
        leaves = [(d, i, n, p) for i, (n, d, p) in enumerate(root.walk()) if not n.children and p is not None]
        if not leaves:
            break
        _, _, leaf, parent = max(leaves, key=lambda t: (t[0], t[1]))
        if leaf.sym:
            leaf.sym = False
        else:
            parent.children.remove(leaf)
    phases = tuple(round(float(p) % (2 * math.pi), 3) for p in g.phases)
    return StickGenome(root, phases, round(_clip(g.frequency, space.frequency), 2),
                       float(round(_clip(g.amplitude, space.amplitude), -1)), g.parents)


# -- operators ---------------------------------------------------------------------------------------------------
def _random_node(space: StickSpace, rng: np.random.Generator) -> Node:
    muscle = int(rng.integers(1, space.n_muscles + 1)) if rng.random() < space.muscle_rate else 0
    return Node(length=float(rng.uniform(*space.length)), radius=float(rng.uniform(space.radius[0], 5.0)),
                mat=space.tissues[int(rng.integers(len(space.tissues)))], yaw=float(rng.uniform(-150, 150)),
                pitch=float(rng.uniform(-60, 20)), muscle=muscle, bend=bool(muscle and rng.random() < 0.7),
                roll=float(rng.choice([0.0, 90.0, -90.0])), sym=bool(rng.random() < 0.5))


def random_genome(space: StickSpace, rng: np.random.Generator) -> StickGenome:
    root = Node(length=float(rng.uniform(15, space.length[1])), radius=float(rng.uniform(4.0, space.radius[1])),
                mat=space.tissues[0])
    for _ in range(int(rng.integers(1, 5))):
        nodes = [n for n, d, _ in root.walk() if d < space.max_depth]
        nodes[int(rng.integers(len(nodes)))].children.append(_random_node(space, rng))
    g = StickGenome(root, tuple(rng.uniform(0, 2 * math.pi, space.n_muscles)),
                    float(rng.uniform(*space.frequency)), float(rng.uniform(*space.amplitude)))
    return repair(g, space)


def _lognormal(x: float, sigma: float, rng: np.random.Generator) -> float:
    return x * math.exp(rng.normal(0.0, sigma))


def mutate(g: StickGenome, space: StickSpace, rng: np.random.Generator) -> StickGenome:
    """One or two structural or parameter changes, plus small oscillator steps; returns a repaired child."""
    root = copy.deepcopy(g.root)
    for _ in range(int(rng.integers(1, 3))):
        nodes = list(root.walk())
        n, depth, parent = nodes[int(rng.integers(len(nodes)))]
        op = rng.choice(["length", "radius", "angle", "mat", "muscle", "bend", "sym", "add", "delete", "copy"])
        if op == "length":
            n.length = _lognormal(n.length, 0.25, rng)
        elif op == "radius":
            n.radius = _lognormal(n.radius, 0.2, rng)
        elif op == "angle":
            attr = rng.choice(["yaw", "pitch", "roll"])
            setattr(n, attr, getattr(n, attr) + rng.normal(0.0, space.angle_sigma))
        elif op == "mat":
            n.mat = space.tissues[int(rng.integers(len(space.tissues)))]
        elif op == "muscle":
            n.muscle = int(rng.integers(0, space.n_muscles + 1))
            n.bend = n.bend or bool(rng.random() < 0.5)
        elif op == "bend":
            n.bend = not n.bend
        elif op == "sym":
            n.sym = not n.sym
        elif op == "add":
            n.children.append(_random_node(space, rng))
        elif op == "delete" and parent is not None:
            parent.children.remove(n)
        elif op == "copy" and parent is not None:
            others = [m for m, d, _ in root.walk() if m is not n and d < space.max_depth]
            if others:
                others[int(rng.integers(len(others)))].children.append(copy.deepcopy(n))
    phases = tuple(p + rng.normal(0.0, space.phase_sigma) for p in g.phases)
    child = StickGenome(root, phases, _lognormal(g.frequency, space.scalar_sigma, rng),
                        _lognormal(g.amplitude, space.scalar_sigma, rng), (g.key,))
    return repair(child, space)


def crossover(a: StickGenome, b: StickGenome, space: StickSpace, rng: np.random.Generator) -> StickGenome:
    """Replace a random non-root subtree of ``a`` by a copy of a random non-root subtree of ``b`` (or graft it on
    ``a``'s root if ``a`` has none); each phase from either parent, scalars averaged."""
    root = copy.deepcopy(a.root)
    donors = [n for n, d, _ in b.root.walk() if d > 0]
    if donors:
        graft = copy.deepcopy(donors[int(rng.integers(len(donors)))])
        targets = [(n, p) for n, d, p in root.walk() if p is not None]
        if targets:
            n, p = targets[int(rng.integers(len(targets)))]
            p.children[p.children.index(n)] = graft
        else:
            root.children.append(graft)
    pick = rng.random(space.n_muscles) < 0.5
    phases = tuple(pa if k else pb for pa, pb, k in zip(a.phases, b.phases, pick))
    child = StickGenome(root, phases, 0.5 * (a.frequency + b.frequency), 0.5 * (a.amplitude + b.amplitude),
                        (a.key, b.key))
    return repair(child, space)


# -- placement ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Placed:
    """One segment in space (mm): starts at ``start``, axis ``R[:, 0]``; ``R`` columns (axis, side, up) with the
    bend plane's roll already applied."""

    index: int
    start: np.ndarray
    R: np.ndarray
    length: float
    radius: float
    mat: str
    muscle: int
    bend: bool

    @property
    def end(self) -> np.ndarray:
        return self.start + self.length * self.R[:, 0]

    @property
    def axis(self) -> np.ndarray:
        return self.R[:, 0]


def _child_frame(F: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    d, l, u = F[:, 0], F[:, 1], F[:, 2]
    y, p = math.radians(yaw), math.radians(pitch)
    d1, l1 = math.cos(y) * d + math.sin(y) * l, -math.sin(y) * d + math.cos(y) * l
    d2, u2 = math.cos(p) * d1 + math.sin(p) * u, -math.sin(p) * d1 + math.cos(p) * u
    return np.column_stack([d2, l1, u2])


def place(g: StickGenome) -> list[Placed]:
    """Forward kinematics of the expanded tree (sym subtrees mirrored: yaw and roll negated)."""
    out: list[Placed] = []

    def visit(n: Node, start: np.ndarray, F_parent: np.ndarray, mirror: bool, is_root: bool) -> None:
        s = -1.0 if mirror else 1.0
        F = F_parent if is_root else _child_frame(F_parent, s * n.yaw, n.pitch)
        r = math.radians(s * n.roll)
        d, l, u = F[:, 0], F[:, 1], F[:, 2]
        R = np.column_stack([d, math.cos(r) * l + math.sin(r) * u, -math.sin(r) * l + math.cos(r) * u])
        p = Placed(len(out), start, R, n.length, n.radius, n.mat, n.muscle, n.bend)
        out.append(p)
        for c in n.children:
            visit(c, p.end, F, mirror, False)
            if c.sym:
                visit(c, p.end, F, not mirror, False)

    visit(g.root, np.zeros(3), np.eye(3), False, True)
    return out
