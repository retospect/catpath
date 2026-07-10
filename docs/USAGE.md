# Usage & extension

## Configuration

A run is one YAML file (see [`examples/no_to_no3_pd.yaml`](../examples/no_to_no3_pd.yaml)).

```yaml
name: no_to_no3_pd
substrate: "NO"      # NOTE: quote chemical labels — bare NO is YAML boolean false
target: "NO3"
slab:
  element: Pd
  size: [3, 3, 4]    # nx, ny, layers
  vacuum: 10.0
  fix_layers: 2      # freeze bottom N layers
mlip:
  backend: emt       # emt | mace | fairchem
  model: null        # checkpoint for mace/fairchem
  device: cpu        # cpu | cuda
search:
  neb_images: 5
  fmax: 0.05         # relaxation force convergence (eV/Å)
  max_steps: 200
  neb_fmax: 0.1      # NEB band convergence
  neb_max_steps: 80
  seeds: [0, 1, 2]   # ≥3 for a mean ± spread
  rmsd_thresh: 0.7   # Å, "same structure"
  energy_thresh: 0.05 # eV, spread tolerance → low_confidence flag
```

> The loader strips YAML's `NO/YES/ON/OFF → bool` coercion, but quoting labels is
> still the safe habit.

## CLI

```bash
atosim run <cfg>                        # all seeds in-process + outputs
atosim seed <cfg> --seed 0 --out p0.json # one seed → partial JSON (fan-out unit)
atosim aggregate <cfg> --partials p*.json # combine partials → outputs
atosim <cfg>                            # shorthand for `run`
# overrides: --backend --seeds 0,1,2 --name --outdir
```

## Snakemake

`workflow/Snakefile` fans out `seed` jobs (one per seed) then runs a single
`aggregate` job. `workflow/config.yaml` selects the run config and seed list.

```bash
snakemake -s workflow/Snakefile -n        # dry run (show DAG)
snakemake -s workflow/Snakefile -c8       # 8 parallel jobs
snakemake -s workflow/Snakefile --dag | dot -Tpng > dag.png
```

Because each seed is an independent job with a declared output, a failed/killed
seed re-runs on its own; unchanged seeds are cached.

## Extension points

| Want to… | Edit |
|---|---|
| Add an MLIP backend | `calculators.make_calculator` (add an `elif`) |
| Add a reaction / intermediate | `network.build_network` (add a `StepSpec`) |
| Change adsorption sites / poses | `structures.SITE_NAMES`, `poses`, `place_fragments` |
| Tune convergence criteria | `relax.relax`, `config.SearchConfig` |
| Change validation thresholds | `validate.geometry_ok`, `config.SearchConfig` |
| Multi-substrate energy map | pass more rows to `viz.energy_map` |

### Production accuracy

The `emt` backend is a placeholder. On the GB10 GPU box:

```bash
uv sync --extra mace
# set mlip.backend: mace, mlip.device: cuda in the config
```

`fairchem` (Open Catalyst) models are purpose-built for adsorbates on metal
surfaces and are the recommended production choice for the NO/Pd chemistry.

### GPU on the DGX Spark / GB10 (Blackwell, sm_121)

torch's `cu128` aarch64 wheel works for plain tensor ops, but MACE/e3nn trigger a
runtime **nvrtc JIT** compile that fails on the GB10 with:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

`sm_121` support first shipped in CUDA **12.9**, while the wheel bundles nvrtc
12.8. Fix (already pinned in the `mace` extra) — upgrade just the nvrtc package;
its soname `libnvrtc.so.12` is stable across 12.x so torch loads the newer one:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install mace-torch "nvidia-cuda-nvrtc-cu12>=12.9"
```

Then set `mlip.backend: mace`, `mlip.device: cuda`. Note: for small slabs (~40
atoms) CPU MACE can be *faster* than GPU due to launch overhead; GPU wins as the
cell grows. Use `default_dtype=float64` (the code does) for geometry opt/NEB.

## Reproducibility

Every run writes `config.snapshot.yaml`. Same config + same seeds → identical
outputs (adsorbate poses are seeded via `numpy.random.default_rng`; the slab and
layout are deterministic).
