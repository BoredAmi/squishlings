"""``squishlings``: evolve soft voxel creatures and watch them (``squishlings --help``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _evolve(args: argparse.Namespace) -> int:
    from squishlings.evolve import Evolution

    evo = Evolution(args.job, jobs=args.jobs)
    if args.generations is not None:
        evo.generations = args.generations
    report = evo.check(args.check) if args.check else evo.run()
    print(json.dumps(report, indent=2, default=float))
    return 0


def _replay(args: argparse.Namespace) -> int:
    from squishlings import replay

    if not (args.gif or args.html):
        print("error: give --gif FILE and/or --html FILE", file=sys.stderr)
        return 2
    run = Path(args.run)
    if run.is_dir():
        records = replay.record_best(run, args.best, args.set, args.terrain, args.niche)
    elif args.terrain:
        from softsim.config.loader import load_yaml

        raw = replay.on_terrain(load_yaml(run), run.parent, args.terrain)
        records = [replay.record(raw, f"{run.stem} on {args.terrain}", run.parent, args.set)]
    else:
        records = [replay.record(run, run.stem, run.parent, args.set)]
    for r in records:
        print(f"{r['name']}: distance {1e3 * r['summary'].get('distance', float('nan')):.1f} mm")
    if args.gif:
        print(replay.gif(records, args.gif))
    if args.html:
        print(replay.html(records, args.html, args.title))
    return 0


def _terrain(args: argparse.Namespace) -> int:
    from softsim.config.loader import load_yaml

    from squishlings.terrain import parse, transfer

    terrains = parse(load_yaml(args.terrains)) if args.terrains else None
    out = transfer(args.runs, args.top, terrains, args.jobs)
    names = list(terrains or parse({"slope": None, "ridges": None, "sand": None}))
    print(f"{'run':28s} {'#':>2s} {'flat':>7s} " + " ".join(f"{n:>13s}" for n in names) + f" {'mean':>7s}")
    for run, rows in out.items():
        for r in rows:
            if "error" in r:
                print(f"{Path(run).name:28s} {r['rank']:2d} error: {r['error'][:60]}")
                continue
            cells = " ".join(f"{1e3 * r[n]:6.0f} ({100 * r[n] / r['flat']:4.0f}%)" if r["flat"] else f"{1e3 * r[n]:13.0f}"
                             for n in names)
            print(f"{Path(run).name:28s} {r['rank']:2d} {1e3 * r['flat']:7.0f} {cells} {1e3 * r['mean']:7.0f}")
    print("distances in mm; slope: uphill along the creature's own flat heading; % of its flat distance")
    return 0


def _creatures(args: argparse.Namespace) -> int:
    import shutil

    from squishlings import CREATURES

    names = sorted(p.name for p in CREATURES.iterdir() if p.name.endswith(".yaml"))
    if args.copy:
        dest = Path(args.copy)
        dest.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copyfile(str(CREATURES / n), dest / n)
        print(f"copied {len(names)} creatures to {dest}")
    else:
        print("\n".join(names))
    return 0


def main(argv: list[str] | None = None) -> int:
    from squishlings import __version__

    ap = argparse.ArgumentParser(prog="squishlings", description=__doc__.split("\n")[0].strip("`: "))
    ap.add_argument("--version", action="version", version=f"squishlings {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("evolve", help="evolve creatures from a job file (resumable)")
    p.add_argument("job", help="job YAML, e.g. jobs/crawler.yaml")
    p.add_argument("--jobs", type=int, default=1, help="creatures simulated in parallel")
    p.add_argument("--generations", type=int, default=None, help="override the job's number of generations")
    p.add_argument("--check", type=int, default=0, metavar="N",
                   help="instead of evolving: re-run the N best creatures of the finished run with a finer mesh, "
                        "half the time step and friction x0.8/x1.2, and flag gaits that depend on them (check.yaml)")
    p.set_defaults(func=_evolve)

    p = sub.add_parser("replay", help="record the best creatures of a run (or one creature YAML): GIF and/or HTML")
    p.add_argument("run", help="an evolve output directory, or a creature scenario YAML")
    p.add_argument("--best", type=int, default=3, help="how many of the best creatures (run directory)")
    p.add_argument("--gif", help="write a GIF here")
    p.add_argument("--html", help="write a self-contained HTML player here")
    p.add_argument("--title", default="Squishlings", help="title of the HTML page")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override (repeatable)")
    p.add_argument("--terrain", choices=["slope", "ridges", "sand"], help="replay on this terrain (default flat)")
    p.add_argument("--niche", choices=["flat", "slope", "ridges", "sand"],
                   help="the best of this ground's specialist archive (runs with archive.per_terrain)")
    p.set_defaults(func=_replay)

    p = sub.add_parser("terrain", help="transfer test: re-run the best creatures of finished runs on every terrain")
    p.add_argument("runs", nargs="+", help="evolve output directories")
    p.add_argument("--top", type=int, default=3, help="how many of each run's best creatures")
    p.add_argument("--jobs", type=int, default=1, help="simulations in parallel")
    p.add_argument("--terrains", help="YAML {terrain: params} (default: slope, ridges and sand with defaults)")
    p.set_defaults(func=_terrain)

    p = sub.add_parser("creatures", help="list the packaged creatures (seeds, champions) or copy them")
    p.add_argument("--copy", metavar="DIR", help="copy them to DIR")
    p.set_defaults(func=_creatures)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
