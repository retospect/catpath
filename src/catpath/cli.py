"""catpath — reaction-pathway explorer for catalyst surfaces with ML potentials.

A run is one YAML config, OR just command-line flags (no file needed):

  catpath run --substrate NO --target NH3 --element Pd --network auto
  catpath run my.yaml --backend mace          # config file + overrides

Commands:
  run <cfg>          run all seeds in-process, write graph/energy-map/results
  states <cfg>       relax states only (no NEB) -> per-model JSON  (for compare)
  barriers <cfg>     NEB for every step        -> per-model JSON  (for compare)
  compare --states.. box-plot several models   (states or barriers JSONs)
  multi <cfg>        several substrates -> union energy map
  sweep <cfg> --elements Pd,Pt,Cu   same network across surfaces
  seed / aggregate   one-seed partial / combine partials (Snakemake fan-out)

The bare form ``catpath <cfg>`` is shorthand for ``catpath run <cfg>``.
Full field reference: docs/CONFIG.md. Every command takes the chemistry flags
(--substrate/--target/--element/--network) and --backend/--device/--seeds/etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .multi import run_multi, write_multi
from .pipeline import (
    aggregate_partials, run, run_barriers, run_one_seed, run_states, write_outputs,
)
from .sweep import run_sweep, write_sweep

_COMMANDS = {"run", "seed", "aggregate", "sweep", "multi",
             "states", "barriers", "compare"}


_NETWORKS = ("ammonia", "branching", "oxidation", "auto")


def _load(args) -> Config:
    path = getattr(args, "config", None)
    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(
                f"catpath: config file not found: {path}\n"
                "  hint: pass a YAML file (see examples/), or run with NO file and set the\n"
                "  chemistry on the CLI:\n"
                "    catpath run --substrate NO --target NH3 --element Pd --network auto")
        try:
            cfg = Config.from_yaml(p)
        except Exception as e:  # noqa: BLE001 - surface any loader error with a hint
            raise SystemExit(
                f"catpath: could not read config {path}: {e}\n"
                "  hint: the file must be YAML. Quote chemical labels — `substrate: \"NO\"`,\n"
                "  because a bare NO is YAML's boolean false. See docs/CONFIG.md.") from e
    else:
        cfg = Config()

    # chemistry overrides — enough to run with no config file at all
    if getattr(args, "substrate", None):
        cfg.substrate = args.substrate
        cfg.substrates = [args.substrate]
    if getattr(args, "target", None):
        cfg.target = args.target
    if getattr(args, "element", None):
        cfg.slab.element = args.element
    if getattr(args, "network", None):
        if args.network not in _NETWORKS:
            raise SystemExit(
                f"catpath: unknown network '{args.network}'.\n"
                f"  hint: choose one of {', '.join(_NETWORKS)} "
                "(auto = autodetect intermediates).")
        cfg.network = args.network

    if getattr(args, "backend", None):
        cfg.mlip.backend = args.backend
    if getattr(args, "device", None):
        cfg.mlip.device = args.device
    if getattr(args, "models", None):
        cfg.mlip.models = [m.strip() for m in args.models.split(",")]
    if getattr(args, "seeds", None):
        try:
            cfg.search.seeds = [int(s) for s in args.seeds.split(",")]
        except ValueError:
            raise SystemExit(f"catpath: --seeds must be integers, e.g. --seeds 0,1,2 "
                             f"(got {args.seeds!r})") from None
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
    # bare `catpath <cfg>` -> `catpath run <cfg>`
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]

    p = argparse.ArgumentParser(prog="catpath", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("config", nargs="?",
                        help="run config YAML (optional — omit and use the flags below)")
    common.add_argument("--substrate", help='starting species, e.g. "NO" (no config needed)')
    common.add_argument("--target", help='ending species, e.g. "NH3"')
    common.add_argument("--element", help="catalyst surface element, e.g. Pd")
    common.add_argument("--network",
                        help="reaction network: ammonia|branching|oxidation|auto")
    common.add_argument("--backend",
                        help="override MLIP backend (emt|mace|chgnet|fairchem|grace|auto)")
    common.add_argument("--device", help="override compute device (cpu|cuda)")
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

    pst = sub.add_parser("states", parents=[common],
                         help="relax states only (no NEB) -> per-model JSON for `compare`")
    pst.add_argument("--out", required=True, help="states JSON output path")
    pst.add_argument("--reference", choices=["formation", "substrate"],
                     default="formation",
                     help="formation (gas-ref, cross-model comparable) | substrate (root-relative)")

    pb = sub.add_parser("barriers", parents=[common],
                        help="run NEB for every step -> per-model barrier JSON for `compare`")
    pb.add_argument("--out", required=True, help="barriers JSON output path")

    pc = sub.add_parser("compare",
                        help="combine `states`/`barriers` JSONs -> box plot (auto-detected)")
    pc.add_argument("--states", nargs="+", required=True,
                    help="states OR barriers JSON files (one per model)")
    pc.add_argument("--out", required=True, help="box-plot PNG output path")
    pc.add_argument("--title", default="State energies by model")
    pc.add_argument("--anchor", default="",
                    help="pin this state to 0 for every model (default: the substrate "
                         "root; 'none' = absolute formation energies)")
    pc.add_argument("--layout", choices=["ordered", "packed"], default="ordered",
                    help="ordered = reaction order (substrate left); packed = energy columns")
    pc.add_argument("--heights", nargs="+", default=None,
                    help="states JSONs to pair with barrier JSONs -> transition-state "
                         "height plot (E_form(reactant)+Ea), highest point starred")

    args = p.parse_args(argv)

    if args.cmd == "compare":  # no run config; just merge JSONs -> plot
        runs = [json.loads(Path(f).read_text()) for f in args.states]
        if runs and "steps" in runs[0]:            # barrier JSONs
            if args.heights:                       # + states -> TS-height plot
                from .viz import compare_ts_heights
                state_runs = [json.loads(Path(f).read_text()) for f in args.heights]
                compare_ts_heights(runs, state_runs, args.out, title=args.title)
                print(f"wrote TS-height plot ({len(runs)} model(s)) -> {args.out}")
                return 0
            from .viz import compare_barriers
            compare_barriers(runs, args.out, title=args.title)
            print(f"wrote barrier box plot ({len(runs)} model(s)) -> {args.out}")
            return 0
        from .viz import compare_boxplot
        if args.anchor.lower() == "none":
            anchor = None
        elif args.anchor:
            anchor = args.anchor
        else:  # default: pin the substrate root (first state in reaction order)
            order = next((r["order"] for r in runs if r.get("order")), [])
            anchor = order[0] if order else None
        compare_boxplot(runs, args.out, title=args.title, anchor=anchor,
                        layout=args.layout)
        print(f"wrote box plot ({len(runs)} model(s), anchor={anchor}, "
              f"{args.layout}) -> {args.out}")
        return 0

    cfg = _load(args)

    if args.cmd == "states":
        data = run_states(cfg, reference=args.reference)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"wrote states for {data['model']} "
              f"({len(data['states'])} states, seeds={data['seeds']}) -> {args.out}")
        return 0

    if args.cmd == "barriers":
        data = run_barriers(cfg)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"wrote barriers for {data['model']} "
              f"({len(data['steps'])} steps, seeds={data['seeds']}) -> {args.out}")
        return 0

    if args.cmd == "multi":
        specs = cfg.substrate_runs()
        print(f"catpath multi: {len(specs)} substrate(s) on {cfg.slab.element}")
        multi = run_multi(cfg)
        outdir = write_multi(cfg, multi)
        print(f"\nDone -> {outdir}")
        return 0

    if args.cmd == "sweep":
        elements = [e.strip() for e in args.elements.split(",")]
        print(f"catpath sweep: {cfg.substrate} -> {cfg.target} on {elements}")
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
    print(f"catpath: {cfg.substrate} -> {cfg.target} on {cfg.slab.element} "
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
