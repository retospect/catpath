# catpath

[![CI](https://github.com/retospect/catpath/actions/workflows/ci.yml/badge.svg)](https://github.com/retospect/catpath/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/catpath.svg)](https://pypi.org/project/catpath/)
[![Python](https://img.shields.io/pypi/pyversions/catpath.svg)](https://pypi.org/project/catpath/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**Reaction-pathway explorer for catalyst surfaces, driven by ML interatomic
potentials.** Give it an environment (a metal surface), a substrate, and a
target; it builds the reaction network, relaxes every intermediate, finds the
barriers with climbing-image NEB, and reports energies with **honest
uncertainty** — pooled across random seeds *and* across ML potentials.

- **Reaction networks** — curated templates *or* rule-based **autodetection** of
  intermediates (`network: auto`).
- **Pluggable ML potentials** — `mace`, `chgnet`, `fairchem` (UMA), `grace`, or
  `auto` (best installed). `emt` is a dependency-free dev backend.
- **Barriers** — climbing-image NEB with automatic retry on non-convergence.
- **Cross-model comparison** — run the same network under several potentials and
  box-plot where they agree and disagree (intermediates, barriers, and which
  transition state is the true rate-limiting "highest point").
- **Reproducible** — every run writes a provenance snapshot; unstable results are
  flagged low-confidence rather than reported as precise numbers.

![NO→NH3 reaction energy profile on Pd](https://raw.githubusercontent.com/retospect/catpath/main/docs/img/energy_profile.png)

*NO→NH₃ on Pd (MACE): every intermediate as a level line, transition states as
barrier bumps with Ea, competing pathways in colour, ± uncertainty bands — one
`catpath run`.* **→ more outputs and their commands in the
[gallery](docs/GALLERY.md).**

## Install

```bash
pip install catpath
```

The default **`emt`** backend is pure numpy/ASE (no torch, no GPU) and runs the
whole pipeline anywhere — great for trying it out and for CI. For real numbers,
add exactly one ML backend (their dependencies conflict, so **one per
environment**):

```bash
pip install "catpath[mace]"      # MACE-MP-0 universal potential (GPU)
pip install "catpath[chgnet]"    # CHGNet (CPU-friendly)
pip install "catpath[fairchem]"  # Meta FAIRChem / UMA (adsorbates on metals)
pip install "catpath[grace]"     # GRACE foundation models
```

## Quickstart

```bash
# no config file needed — set the chemistry on the command line:
catpath run --substrate NO --target NH3 --element Pd --network auto

# or point at a YAML config (see examples/):
catpath run examples/no_to_no3_pd.yaml

# discover the intermediates automatically, on a real ML potential:
catpath run examples/auto_ammonia.yaml --backend auto
```

Run `catpath --help` (or `catpath run --help`) for every flag. Config files and
flags mix freely — flags override the file.

Outputs land in `runs/<name>/`:

| File | Contents |
|---|---|
| `graph_thumbs.png` | reaction energy-profile with active-site structure thumbnails |
| `graph_network.png` | node/DAG view of the network (red = low-confidence) |
| `energy_map.png` | substrate × intermediate heatmap; ★ = rate-limiting state |
| `results.json` | nodes, edges, barriers, mean ± spread, warnings |
| `methods.md` | a deterministic methods paragraph for your write-up |
| `config.snapshot.yaml` | provenance snapshot for exact reproduction |

## Compare several ML potentials

Because the backends can't share an environment, run `states` / `barriers` in
each one's env, then `compare` the JSONs:

```bash
catpath states   my.yaml --backend chgnet   --out s_chgnet.json
catpath states   my.yaml --backend fairchem --out s_uma.json
catpath compare  --states s_*.json --out intermediates.png     # box plot per state

catpath barriers my.yaml --backend chgnet   --out b_chgnet.json
catpath compare  --states b_*.json --out barriers.png          # Ea, rate-limiting ringed
catpath compare  --states b_*.json --heights s_*.json --out ts_heights.png
```

![Cross-model comparison of intermediate formation energies](https://raw.githubusercontent.com/retospect/catpath/main/docs/img/models_intermediates.png)

State energies are referenced to per-element gas-phase chemical potentials
computed *in each potential*, so composition-changing states are comparable
across models. See the [gallery](docs/GALLERY.md) and
[`examples/README.md`](examples/README.md) for the full set of commands.

## CLI

```
catpath run <cfg>            # all seeds in-process + outputs
catpath states <cfg>         # relax states only (no NEB) -> per-model JSON
catpath barriers <cfg>       # NEB for every step -> per-model JSON
catpath compare --states ... # box plots (states or barriers, auto-detected)
catpath multi <cfg>          # several substrates -> union energy map
catpath sweep <cfg> --elements Pd,Pt,Cu   # same network across surfaces
```

Everything is one YAML file — see [`docs/CONFIG.md`](docs/CONFIG.md) for every
field, and [`docs/USAGE.md`](docs/USAGE.md) for extension points.

## Development

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest
```

## Contributing

Issues and PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). Maintained by
Reto Stamm.

## Acknowledgements

catpath was requested by **Muhammad Umer**, whose help shaping what it should do
got the project off the ground.

Built with **Claude** (Anthropic) via Claude Code, with research assistance from
**Perplexity**.

## License

GPL-3.0-or-later. Built on [ASE](https://wiki.fysik.dtu.dk/ase/) (LGPL) and
RDKit (BSD).
