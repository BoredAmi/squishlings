# Squishlings

Evolve soft creatures that crawl. Squishlings grows bodies out of one-centimetre cubes of real silicones
(Ecoflex 00-30, Dragon Skin 10), gives them muscles that pulse on oscillators, and lets evolution find the ones that
travel furthest. The idea is the one [Framsticks](https://www.framsticks.com/) made famous with stick creatures;
here every creature is a soft solid, simulated with [softsim](https://github.com/BoredAmi/softsim)'s material cards on
MuJoCo flex.

![The three champions of the first run, crawling](docs/media/crawlers.gif)

*The three best creatures of the first run (2³ mesh cubes per voxel). Grey: Ecoflex 00-30, blue: Dragon Skin 10,
reds and oranges: the four muscle groups. Each travels 0.5-0.6 m in 2.7 s.*

## Quick start

```bash
pip install -e .                                   # needs softsim >= 0.2.0 (voxel bodies and muscles)
squishlings creatures                              # the packaged seeds and champions
softsim run squishlings/creatures/champion_1.yaml  # run one; any softsim command works on a creature
squishlings evolve jobs/crawler_fine.yaml --jobs 6                    # evolve (~4 min a generation, resumable)
squishlings evolve jobs/crawler_fine.yaml --check 3                   # exploit check of the 3 best
squishlings replay runs/crawler_fine --best 3 --gif crawlers.gif --html crawlers.html
```

The HTML replay is one self-contained page: open it in a browser, orbit the camera, pick a creature, scrub through
time. Muscles brighten as they contract.

## How a creature is made

- **Body:** a grid of up to 6 × 6 × 4 cells. Each cell is empty, a tissue (`a`, `b`: material cards) or a muscle
  group (`1`-`4`). It becomes a softsim `voxels` body; every label is a material zone. Cubes are bonded through their
  faces, and only the largest connected part survives, standing on the ground.
- **Muscles:** softsim's `type: muscle` actuator, an active stress along a fibre (x, y or z) or isotropic, driven by
  `amplitude · sin(2π f t + phase)`. Each group has its own phase and direction. Frequency (1-12 Hz) and amplitude
  (5-40 kPa) are shared by the creature and evolve too. The muscle is an effective model: no real device produces a
  volumetric stress. It is the standard abstraction of evolved soft robots. To build a creature, replace each group
  with a pneumatic chamber or a cable that gives the same strain; softsim validates both.
- **Evolution:** a (μ+λ) strategy. Tournament selection, one-cut crossover, mutation of cells, phases, directions and
  scalars; the best of parents and children survive, so the champion is never lost. Hand-made creatures can seed the
  first population. Every simulation is cached in `evaluations.jsonl`: a run is deterministic for a seed, and a
  stopped run resumes without re-simulating (a laptop shutdown cost one generation).
- **Fitness:** the planar distance the centre of mass travels after 0.3 s of settling, in 3 s on MuJoCo. Any metric of
  the `locomotion` scenario can serve: `distance_x`, `mean_speed`, `path_length`, `max_tilt`, `turn`, `flipped`.

## How far to trust a distance

Evolution finds whatever moves the creature, including numerical artefacts. `--check` re-runs the champions with
each cube split into 2³, half the time step, and friction ×0.8 and ×1.2, and flags any gait whose distance moves by
more than 25%. What it found on the first runs (2026-11-17, laptop, 6 workers):

| Run | Creatures | Distance as evolved | One mesh level finer | Time step / 2 | Friction ±20% |
|---|---|---|---|---|---|
| `crawler.yaml`, 1 cube per voxel, 30 generations | 744 | 882 / 822 / 817 mm | −45% / −45% / −80% | +2-4%, −6% | under 9% |
| `crawler_fine.yaml`, 2³ per voxel, 12 generations | 208 | 683 / 672 / 667 mm | −17% / −13% / −26% (3³) | | |

The gaits hop more than they slide, so friction barely matters. The mesh does: MuJoCo's linear tetrahedra are too
stiff in bending, and a one-cube limb is six of them, so the coarse run evolved creatures that rely on stiff legs.
Evolving at 2³ halves the error, and its winners beat the coarse winners at 3³ too (569-585 against 333-360 mm).
**Distances rank creatures at the run's resolution; they are not converged speeds.** Evolve with `subdivide: 2` and
check the winners one level finer.

Two more effects of the numerics:

- **Friction:** MuJoCo uses the larger of the two friction coefficients in a contact, and the silicone cards say 1.0, so
  a ground friction of 0.8 alone does nothing. Squishlings writes one friction (`genome.friction`, default the
  ground's) into the ground, the body and every zone.
- **Symmetry:** the six tetrahedra of each cube are not mirror symmetric, so a symmetric body drifts sideways (the
  inchworm about 1 mm per 5 mm forward).

Also: on MuJoCo every run prints "Flex stiffness is too ill-conditioned" and "Inertia matrix is too close to
singular" warnings, nearly all on the first time step, where the solver starts from rest with nothing to warm-start
from. They do not affect the results (no run failed, replays reproduce the evolved distances exactly).

## Files

| | |
|---|---|
| `squishlings/genome.py` | the encoding and its operators (mutation, crossover, repair) |
| `squishlings/build.py` | genome ↔ softsim scenario |
| `squishlings/evolve.py` | the evolution loop and the exploit check |
| `squishlings/locomotion.py` | the `locomotion` scenario (registered with softsim as `scenario: locomotion`) |
| `squishlings/replay.py` | GIF and HTML replays |
| `squishlings/creatures/` | `base.yaml` (shared settings), seeds (`inchworm`, `quad`), the first run's champions |
| `jobs/` | `crawler.yaml` (fast, exploratory), `crawler_fine.yaml` (recommended) |

## Ideas

Neural controllers reading touch and strain sensors instead of open-loop oscillators (Framsticks' brains); swimming
and climbing tasks; turning muscle groups into buildable chambers and cables; a GPU backend for larger populations.

Licence: Apache-2.0.
