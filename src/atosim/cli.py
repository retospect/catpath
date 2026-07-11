"""Command-line entry point.

Subcommands (mirrored by the Snakemake rules):
  atosim run <cfg>                     run all seeds in-process + write outputs
  atosim seed <cfg> --seed N --out F   run ONE seed -> partial JSON (fan-out unit)
  atosim aggregate <cfg> --partials .. combine partials -> outputs

The bare form ``atosim <cfg>`` is shorthand for ``atosim run <cfg>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .multi import run_multi, write_multi
from .pipeline import aggregate_partials, run, run_one_seed, write_outputs
from .sweep import run_sweep, write_sweep

_COMMANDS = {"run", "seed", "aggregate", "sweep", "multi"}


def _load(args) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if getattr(args, "backend", None):
        cfg.mlip.backend = args.backend
    if getattr(args, "models", None):
        cfg.mlip.models = [m.strip() for m in args.models.split(",")]
    if getattr(args, "seeds", None):
        cfg.search.seeds = [int(s) for s in args.seeds.split(",")]
    if getattr(args, "reagents", None) is not None:
        r = args.reagents.strip()
        cfg.reagents = [x.strip() for x in r.split(",") if x.strip()] if r else []
    if getattr(args, "name", None):
        cfg.name = args.name
    if getattr(args, "outdir", None):
        cfg.outdir = args.outdir
    return cfg


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # bare `atosim <cfg>` -> `atosim run <cfg>`
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]

    p = argparse.ArgumentParser(prog="atosim", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("config", nargs="?", help="run config YAML")
    common.add_argument("--backend",
                        help="override MLIP backend (emt|mace|chgnet|fairchem|grace|auto)")
    common.add_argument("--models", help="multi-model: comma-separated, e.g. small,medium "
                                         "or mace:small,mace:medium")
    common.add_argument("--seeds", help="override seeds, comma-separated")
    common.add_argument("--reagents", help="available reagent adatoms, comma-separated "
                                           "(e.g. H or H,O); empty string = reagent-free only")
    common.add_argument("--name", help="override run name")
    common.add_argument("--outdir", help="override output directory")

    sub.add_parser("run", parents=[common], help="run all seeds + write outputs")

    ps = sub.add_parser("seed", parents=[common], help="run one seed -> partial")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--out", required=True, help="partial JSON output path")

    pa = sub.add_parser("aggregate", parents=[common], help="combine partials")
    pa.add_argument("--partials", nargs="+", required=True)

    pw = sub.add_parser("sweep", parents=[common],
                        help="run the network across surfaces -> multi-row energy map")
    pw.add_argument("--elements", required=True,
                    help="comma-separated surface elements, e.g. Pd,Pt,Cu,Ni")

    sub.add_parser("multi", parents=[common],
                   help="run several substrates (config `substrates:`) -> combined energy map")

    args = p.parse_args(argv)
    cfg = _load(args)

    if args.cmd == "multi":
        specs = cfg.substrate_runs()
        print(f"atosim multi: {len(specs)} substrate(s) on {cfg.slab.element}")
        multi = run_multi(cfg)
        outdir = write_multi(cfg, multi)
        print(f"\nDone -> {outdir}")
        return 0

    if args.cmd == "sweep":
        elements = [e.strip() for e in args.elements.split(",")]
        print(f"atosim sweep: {cfg.substrate} -> {cfg.target} on {elements}")
        sweep = run_sweep(cfg, elements)
        outdir = write_sweep(cfg, sweep)
        print(f"\nDone -> {outdir}")
        return 0

    if args.cmd == "seed":
        partial = run_one_seed(cfg, args.seed)
        Path(args.out).write_text(json.dumps(partial, indent=2))
        print(f"wrote partial seed={args.seed} -> {args.out}")
        return 0

    if args.cmd == "aggregate":
        partials = [json.loads(Path(f).read_text()) for f in args.partials]
        results = aggregate_partials(cfg, partials)
        outdir = write_outputs(cfg, results)
        print(f"aggregated {len(partials)} partial(s) -> {outdir}")
        return 0

    # run
    print(f"atosim: {cfg.substrate} -> {cfg.target} on {cfg.slab.element} "
          f"[{cfg.mlip.backend}] seeds={cfg.search.seeds}")
    results = run(cfg)
    outdir = write_outputs(cfg, results)
    if results.warnings:
        print(f"\n{len(results.warnings)} warning(s):")
        for w in results.warnings[:20]:
            print("  -", w)
    print(f"\nDone -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
