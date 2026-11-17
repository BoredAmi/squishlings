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
    records = (replay.record_best(run, args.best, args.set) if run.is_dir()
               else [replay.record(run, run.stem, run.parent, args.set)])
    for r in records:
        print(f"{r['name']}: distance {1e3 * r['summary'].get('distance', float('nan')):.1f} mm")
    if args.gif:
        print(replay.gif(records, args.gif))
    if args.html:
        print(replay.html(records, args.html, args.title))
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
    p.set_defaults(func=_replay)

    p = sub.add_parser("creatures", help="list the packaged creatures (seeds, champions) or copy them")
    p.add_argument("--copy", metavar="DIR", help="copy them to DIR")
    p.set_defaults(func=_creatures)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
