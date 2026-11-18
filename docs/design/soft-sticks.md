# Soft sticks: a Framsticks-style body encoding for squishlings

Status: sketch, nothing implemented. Written 2026-11-18.

## Why

Voxel bodies work, but they have three problems (README, "How far to trust a distance"):

- Thin limbs are badly resolved: six linear tets per cube are too stiff in bending.
- The tet split is asymmetric, so symmetric bodies drift sideways.
- Staircase shapes are hard to cast.

Framsticks builds bodies from sticks: a tree of segments joined end to end. A soft version makes each stick a
silicone capsule and fuses the tree into one solid. Two things follow:

- Limbs are long and thin by construction, so we can mesh them finely enough to bend properly.
- The genome is a short text tree. That suits the LLM mutation idea far better than 3D ASCII slices.

The plan is to keep voxels as the baseline and add sticks as a second `encoding`.

## Genome: an indented text tree

One segment per line, and indentation gives the tree. The first line is the root. Each child grows from its
parent's far end.

```text
# quadruped: torso with two mirrored leg pairs
seg len=24 r=6 mat=a
  seg len=14 r=3 mat=a yaw=60  pitch=-35 muscle=1 bend roll=0 sym
    seg len=9 r=2.5 mat=b pitch=-30
  seg len=14 r=3 mat=a yaw=120 pitch=-35 muscle=2 bend roll=0 sym
    seg len=9 r=2.5 mat=b pitch=-30
---
freq=6 amp=20k phase1=0 phase2=3.14
```

| Field | Meaning | Range |
| --- | --- | --- |
| `len` | segment length, mm | 5-40 |
| `r` | radius, mm (optional `r2` gives a taper to the far end) | 2-8 |
| `mat` | tissue label, as for voxels (`a` Ecoflex 00-30, `b` Dragon Skin 10) | job's tissues |
| `yaw`, `pitch` | direction relative to the parent's axis, degrees | ±180, ±90 |
| `muscle=g` | the segment is muscle group `g` (shares that group's phase) | 1-4 or absent |
| `axial` / `bend` | muscle kind: contracts along the axis, or bends (see Muscles) | |
| `roll` | bend plane, degrees about the segment's own axis | ±180 |
| `sym` | also add the mirror image of this subtree across the parent's vertical plane | flag |

The footer after `---` holds the oscillator: shared frequency and amplitude, and one phase per group, the same
as for voxels. A group no longer needs a fibre direction, because the segment's axis provides it.

The text is the canonical form. The genome key is the hash of the normalised text, which has rounded numbers and
a fixed field order.

Limits keep simulations affordable and comparable with voxel runs: at most 12 segments after `sym` expansion,
depth at most 4, and a bounding box of about 6 × 6 × 4 cm.

## Phenotype: the tree to a mesh

1. **Forward kinematics.** The root lies along +x at the origin. A child starts at its parent's end, with
   direction = parent direction rotated by `yaw` (about the parent's local up) and then `pitch`. Mirrored
   subtrees negate the lateral component.
2. **Solids, in gmsh's OpenCASCADE kernel:**
   - one cylinder (or cone if tapered) per segment;
   - one sphere at every joint and at every free end, so joints are rounded and castable;
   - a `bend` segment is cut lengthwise by a plane through its axis (rotated by `roll`) into two halves.
3. **`BooleanFragments` on everything.** Every sub-volume survives with conformal shared faces, so the result
   meshes as one bonded solid that still knows which piece is which. Each piece gets a 3D physical group: `s3`,
   or `s3_flex` and `s3_ext` for halves. A joint sphere takes its parent's tissue.
4. **Mesh.** Element size is about `min(r)/2`, so a limb has a few elements across and can bend. Write a `.msh`.
5. **softsim body.** `geometry: {type: mesh_file, path: <key>.msh, center: base}` puts the lowest point on the
   ground. Zones and muscle regions refer to the physical groups by name.
6. **Cache.** Meshes go to `<output>/meshes/<key>.msh`, the same way `evaluations.jsonl` caches results.

If meshing fails (for example near-tangent overlaps between non-adjacent segments), the creature is logged as
invalid with fitness 0 and is not retried. Overlaps themselves are allowed: they fuse, as in Framsticks.

## Muscles

- **`axial`:** one `type: muscle` actuator on the segment's region, with `direction` = the segment's axis
  (reference configuration). The segment shortens and lengthens.
- **`bend`:** two actuators on the halves, both along the axis. The flexor uses the group's phase, the extensor
  phase + π. The segment curls in its `roll` plane, which is how real two-chamber soft actuators bend. This is
  the soft equivalent of a Framsticks muscle bending a joint.

Amplitude and frequency are shared by the creature, as now. The README's caveat still applies: a volumetric
stress is an effective model, and each muscle still needs to become a buildable chamber or cable.

## Operators

All operators return a **repaired** genome. Repair clamps every number to its range, drops fields that make no
sense (`roll` without `bend`), prunes the deepest leaves until the segment limit holds, and renormalises the text.

- **Mutate.** Pick 1-2 of the following:
  - a log-normal step on one number;
  - change `mat`;
  - add, remove or regroup a muscle;
  - switch `axial` and `bend`;
  - toggle `sym`;
  - add a leaf;
  - delete a non-root subtree;
  - duplicate a subtree onto another node.

  Phase, frequency and amplitude steps are the same as for voxels.
- **Crossover.** Swap a random subtree of A for a random subtree of B (genetic-programming style). Each phase
  comes from either parent, and the scalars are averaged.
- **Random genome.** A root plus 1-6 random children, depth at most 3.
- **LLM mutation, later.** The prompt holds the parent's text and distance, plus the best and worst few of the
  population with their distances. The model returns a new text. If parsing fails, ask once more, then fall back
  to `mutate`. Ollama runs locally, and a 7-14B model fits the RTX 3060.

## Code changes

- `squishlings/sticks.py`: the genome, parser and printer, repair, and operators (the counterpart of `genome.py`).
- `squishlings/sticks_mesh.py`: forward kinematics and gmsh, tree → `.msh` plus region names.
- `build.py`: a `SticksBuilder` that emits the `mesh_file` body, zones and actuators.
- `evolve.py`: `genome.encoding: voxels | sticks`. Hide both encodings behind one small interface (`random`,
  `mutate`, `crossover`, `repair`, `key`, `scenario`, `to_dict` / `from_dict`), and include the encoding in the
  cache hash.
- Seeds: `inchworm_sticks.txt` (three segments in a row, bending) and `quad_sticks.txt` (above).
- `--check`: the mesh refinement level becomes "element size / 2" instead of "subdivide + 1".

## Check in softsim first

1. Does the MuJoCo backend accept a `mesh_file` body, and do a `.msh` file's 3D physical groups work as `region`
   for both zones and `type: muscle`? Spike: two fused capsules, one of them a muscle.
2. Is a muscle's `direction` taken in the reference configuration, element by element? A bent limb otherwise
   pulls along the wrong axis.
3. Cost: tet count and time per creature for a 12-segment body at `r/2` elements, against about 15 s for a 2³
   voxel creature. If it is much slower, coarsen the torso and keep the limbs fine.
4. Bending convergence: a single cantilevered capsule with a `bend` muscle at two mesh sizes. Its tip deflection
   should change by much less than the 13-26% the 2³ voxels lose.

### Spike results (2026-11-18, `spikes/sticks_spike.py`)

Test body: a passive torso (30 mm long, r = 5 mm) fused to a 25 mm `bend` segment (r = 4 mm). Each run is 2 s on
MuJoCo, oscillating at 4 Hz.

1. **Works.** gmsh fragments into one conformal solid in 0.2-0.4 s. The `.msh` physical groups work as both zone
   and muscle regions through `mesh_file`. With the muscles off, nothing moves.
2. **Answered: muscle direction is in the body frame.** softsim uses `m = F m0 / |F m0|`, so the fibre follows
   the material. Bending works: tip excursion is linear in amplitude (3.5 / 7.8 / 13.6 mm at 4 / 8 / 15 kPa).
   30 kPa is far too strong for a 4 mm Ecoflex limb (the tip whips through about 25 mm), so stick amplitudes need
   a lower range than voxels, roughly 2-15 kPa.
3. **Cost is about 4× a 2³ voxel creature.** At element size r/2 this body has 3057 tets and takes about 40 s
   per 2 s simulated (about 60 s for 3 s). At r/4 it has 15034 tets and takes about 245 s.
4. **Bending is not converged at r/2.** Tip excursion grows 21-25% at r/4. That is the same band as the 2³ voxels
   (13-26%), so sticks do *not* fix the linear-tet bending stiffness by themselves.
5. **New problem: round bodies roll.** The finer (rounder) mesh rolls onto its side even at 4 kPa (tilt 87-94°).
   The coarse mesh stays upright, and its faceting is what holds it. Torsos need a flat base, for example a
   flattened or elliptical cross-section or a flat-bottom option, or else enough legs to stand on.

## Experiment

Run a 2 × 2 grid: {voxels, sticks} × {random mutation, random + 25% LLM mutation}. Give every cell the same
number of simulations, use at least 3 seeds per cell, and use the same locomotion scenario. Compare:

- best distance against simulations spent;
- how much of each champion's distance survives `--check`;
- whether the champions can be built.

## First evolution (2026-11-18, prototype)

`python -m squishlings.sticks_evolve --generations 4 --population 8 --offspring 8 --jobs 6`, 2 s per creature
(1.7 s of gait), seeded with an inchworm and a walker. Changes from the sketch, found while building it:

- **Fuse and label instead of fragmenting.** Fragmenting overlapping capsules meshed only 4 of 20 random
  creatures. Fusing to one solid and labelling each tet by its nearest capsule meshes reliably.
- **Joint spheres of 1.1 r.** At 1.0 r, sphere and cylinder are tangent and gmsh fails on the seams (39 of 42
  test creatures meshed). At 1.1 r all 42 mesh, and the mesher retries at 1.05 and 1.2 r anyway.
- **Rigid placement.** gmsh's `affineTransform` turns cylinders into B-splines that then fail to mesh, so pieces
  are placed with rotate and translate instead.
- **Flat feet.** The body is cut 1.5 mm above its lowest point. No creature flipped.

Results: 40 creatures, 0 mesh or simulation failures, median 4151 tets, about 150 s per creature with 6 in
parallel. Best distance by generation: 5.5 → 6.4 → 11.2 → 20.7 → 20.7 mm; the median rose from 0.3 to 10.3 mm.
The champion is the walker with its leg pair copied onto the tail by the `copy` mutation, so it now has four
legs. It moves straight along +x with a maximum tilt of 17°. This is far slower than the voxel crawlers
(600-700 mm in 2.7 s), because the amplitudes are much lower, the gait is shorter, and there were only
4 generations.

## Long run (2026-11-18)

`python -m squishlings.sticks_evolve --generations 20 --population 16 --offspring 16 --jobs 8 --duration 3 --seed 1
--out runs/sticks_long --seeds runs/sticks_test/best_{1,2,3}.txt`: 336 creatures in 5.7 h, no mesh or simulation
failures. Best distance by generation: 33 mm (0) → 329 (10) → 393 (15-20); the median reached 340 mm.

The champion keeps the walker's plan: a torso with two mirrored leg pairs and a tail. Evolution turned the torso
into a bending muscle and pushed the oscillator to 15 kPa (the upper bound) at 7 Hz. It crawls towards its tail
(−x) with at most 25° of tilt.

Exploit check (`spikes/sticks_check.py`, top 3):

| Creature | Evolved (3 mm mesh, ~5k tets) | Mesh 2 mm (~9k tets) | Time step / 2 | Friction ×0.8 | Friction ×1.2 |
| --- | --- | --- | --- | --- | --- |
| c5b1dbf9 | 393 mm | +2% | +4% | −16% | +6% |
| ea806619 | 391 mm | +10% | −2% | −16% | +12% |
| 21dcc0ed | 362 mm | +11% | −1% | −16% | +14% |

- **The mesh matters far less than for voxels.** Distances grow 2-11% on the finer mesh, where 2³ voxels lost
  13-26% one level finer. The finer mesh is only 1.7× the tets, though, against 3.4× for the voxel check, and the
  sign suggests the coarse mesh is still slightly too stiff. The order of the top 3 changes under refinement, so
  close distances do not rank reliably.
- **These gaits crawl.** Friction moves the distance by 6-16%, while the voxel gaits hop and are under 9%.
- **Compared at the finer mesh,** sticks reach about 400-430 mm and the 2³ voxel champions 569-585 mm at 3³.
  The voxels had 2 runs (42 generations in total) and an amplitude limit of 40 kPa, against 15 kPa here.
