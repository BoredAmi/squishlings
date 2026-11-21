"""A language model as a mutation operator for soft sticks (Evolution through Large Models, Lehman et al. 2022).

The stick genome is short text, so a local model (ollama) can read a parent, see how far it and a few other
elites travelled, and write an edited genome. :func:`squishlings.sticks.from_text` and ``repair`` make every
answer valid; an answer that does not parse, or that changes nothing, returns ``None`` and the caller falls back
to random mutation.

Answers are cached in a JSON-lines file keyed by (model, prompt, seed), so a stopped run resumes without asking
again and a run is reproducible from its cache even though sampling is not.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from squishlings.sticks import ParseError, StickGenome, StickSpace, from_text

FLAT_TASK = "crawl on flat ground. Fitness is the\ndistance the centre of mass travels in 2.7 s."

_RULES = """You design soft robots made of silicone capsules (sticks) that {task} Each robot is a tree of segments, written one per line:

  seg len=<mm> r=<mm> mat=<a|b> [yaw=<deg>] [pitch=<deg>] [muscle=<1..{n_muscles}> [bend [roll=<deg>]]] [sym]

- Indentation (2 spaces per level) makes a segment the child of the line above it with less indentation. The
  first line is the root; it lies along +x. A child grows from its parent's far end.
- len {len_lo:g}-{len_hi:g} mm, r {r_lo:g}-{r_hi:g} mm. mat=a: soft Ecoflex 00-30, mat=b: stiffer Dragon Skin 10.
- yaw turns left/right about the parent's up axis, pitch tilts up (+) or down (-), {p_lo:g} to {p_hi:g} deg.
- muscle=g makes the segment muscle group g; it contracts along its length. With bend it is split lengthwise into
  two halves driven in antiphase, so it curls; roll rotates the plane it curls in.
- sym adds a mirrored copy of the segment and its whole subtree (a left/right pair). Do not use sym on a segment
  that points straight ahead or back (yaw 0 or 180): its mirror lands on itself.
- At most {max_segments} segments in total (a sym copy counts), at most {max_depth} levels below the root.
- The footer, after a line "---", drives the muscles: group g contracts with amp * sin(2 pi freq t + phase_g).
  freq {f_lo:g}-{f_hi:g} Hz, amp {a_lo:g}-{a_hi:g} Pa, phases in radians, one per group ({n_muscles} groups).
- The body is cut flat 1.5 mm above its lowest point, so the lowest segments are its feet.
- Write plain decimal numbers only (no expressions such as pi or 1.2+3.14)."""

_ASK = """Propose ONE modified robot that you expect to travel further than the parent. Change the body, not only
the numbers: add, remove or move limbs, change what is a muscle, or change the symmetry, and adjust the footer to
match. Keep what probably makes the parent work.

Answer with one line starting with "Idea:" (what you changed and why), then the complete robot in a ```text
block, in exactly the format above."""

_BLOCK = re.compile(r"```(?:text)?\s*\n(.*?)```", re.S)
_IDEA = re.compile(r"^\s*Idea:\s*(.+)$", re.M)


def _describe(d: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.2g}" for k, v in d.items())


@dataclass
class LLMMutator:
    """``propose(parent, ...)`` -> an edited genome, or ``None`` (fall back to random mutation)."""

    model: str
    space: StickSpace
    cache: Path
    url: str = "http://localhost:11434"
    temperature: float = 0.8
    timeout: float = 300.0
    think: bool = False
    task: str = FLAT_TASK  # what the robot is scored on: the start of the system prompt (terrain jobs change it)
    stats: dict[str, int] = field(default_factory=lambda: {"ok": 0, "unparsed": 0, "unchanged": 0, "error": 0,
                                                            "cached": 0})

    def __post_init__(self) -> None:
        self.cache = Path(self.cache)
        self._answers: dict[str, dict[str, Any]] = {}
        if self.cache.is_file():
            for line in self.cache.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._answers[row["id"]] = row
        s = self.space
        self._system = _RULES.format(
            task=self.task, n_muscles=s.n_muscles, len_lo=s.length[0], len_hi=s.length[1], r_lo=s.radius[0], r_hi=s.radius[1],
            p_lo=s.pitch[0], p_hi=s.pitch[1], max_segments=s.max_segments, max_depth=s.max_depth,
            f_lo=s.frequency[0], f_hi=s.frequency[1], a_lo=s.amplitude[0], a_hi=s.amplitude[1])

    def prompt(self, parent: StickGenome, fitness: float, descriptors: dict[str, float],
               others: list[tuple[StickGenome, float, dict[str, float]]]) -> str:
        parts = []
        if others:
            parts.append("Other robots found so far (different body types), with their distances:")
            for g, f, d in others:
                parts.append(f"\n# {f * 1000:.0f} mm ({_describe(d)})\n```text\n{g.text}```")
        parts.append(f"\nThe parent to modify travels {fitness * 1000:.0f} mm ({_describe(descriptors)}):"
                     f"\n```text\n{parent.text}```\n")
        parts.append(_ASK)
        return "\n".join(parts)

    def _chat(self, prompt: str, seed: int) -> str:
        body = {"model": self.model, "stream": False, "think": self.think, "keep_alive": "60m",  # loaded between batches
                "messages": [{"role": "system", "content": self._system}, {"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "seed": seed, "num_predict": 1500}}
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

    def propose(self, parent: StickGenome, fitness: float, descriptors: dict[str, float],
                others: list[tuple[StickGenome, float, dict[str, float]]], seed: int
                ) -> tuple[StickGenome | None, dict[str, Any]]:
        prompt = self.prompt(parent, fitness, descriptors, others)
        rid = hashlib.sha256(json.dumps([self.model, self.temperature, self.think, self._system, prompt, seed])
                             .encode()).hexdigest()[:24]
        row = self._answers.get(rid)
        if row is None:
            t0 = time.perf_counter()
            try:
                answer = self._chat(prompt, seed)
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                self.stats["error"] += 1
                return None, {"status": "error", "error": str(exc)}
            row = {"id": rid, "model": self.model, "parent": parent.key, "seed": seed, "answer": answer,
                   "seconds": round(time.perf_counter() - t0, 2)}
            self._answers[rid] = row
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            with self.cache.open("a") as f:
                f.write(json.dumps(row) + "\n")
        else:
            self.stats["cached"] += 1
        child, info = self.parse(row["answer"], parent)
        self.stats[info["status"]] += 1
        info["seconds"] = row.get("seconds")
        return child, info

    def parse(self, answer: str, parent: StickGenome) -> tuple[StickGenome | None, dict[str, Any]]:
        idea = _IDEA.search(answer)
        info: dict[str, Any] = {"idea": idea.group(1).strip() if idea else ""}
        blocks = _BLOCK.findall(answer)
        text = blocks[-1] if blocks else answer
        try:
            child = from_text(text, self.space)
        except (ParseError, IndexError) as exc:
            return None, {**info, "status": "unparsed", "error": str(exc)}
        if child.key == parent.key:
            return None, {**info, "status": "unchanged"}
        return replace(child, parents=(parent.key,)), {**info, "status": "ok", "key": child.key}
