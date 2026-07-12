# Usage & extension

## Configuration

A run is one YAML file (see [`examples/no_to_no3_pd.yaml`](../examples/no_to_no3_pd.yaml)).

```yaml
name: no_to_no3_pd
substrate: "NO"      # NOTE: quote chemical labels — bare NO is YAML boolean false
target: "NO3"
network: ammonia     # curated: ammonia|branching|oxidation — or `auto` to autodetect
slab:
  element: Pd
  size: [3, 3, 4]    # nx, ny, layers
  vacuum: 10.0
  fix_layers: 2      # freeze bottom N layers
mlip:
  backend: auto      # emt | mace | chgnet | fairchem | grace | auto (best installed ML)
  model: null        # model/checkpoint name (backend-specific)
  device: cpu        # cpu | cuda
search:
  neb_images: 5
  fmax: 0.05         # relaxation force convergence (eV/Å)
  max_steps: 200
  neb_fmax: 0.1      # NEB band convergence
  neb_max_steps: 80
  neb_retries: 1     # retry a non-converged NEB with a denser band + more steps
  seeds: [0, 1, 2]   # ≥3 for a mean ± spread
  rmsd_thresh: 0.7   # Å, "same structure"
  energy_thresh: 0.05 # eV, spread tolerance → low_confidence flag
```

> The loader strips YAML's `NO/YES/ON/OFF → bool` coercion, but quoting labels is
> still the safe habit.

## CLI

```bash
catpath run <cfg>                        # all seeds in-process + outputs
catpath seed <cfg> --seed 0 --out p0.json # one seed → partial JSON (fan-out unit)
catpath aggregate <cfg> --partials p*.json # combine partials → outputs
catpath sweep <cfg> --elements Pd,Pt,Cu   # same network across surfaces → multi-row map
catpath multi <cfg>                       # several substrates (config `substrates:`) → union map
catpath states <cfg> --out s_mace.json    # relax states only (no NEB) → per-model JSON
catpath barriers <cfg> --out b_mace.json  # NEB for every step → per-model barrier JSON
catpath compare --states s_*.json --out cmp.png   # merge → box plot per state, dots per model
catpath compare --states b_*.json --out bars.png  # barrier JSONs → Ea box plot, rate-limiting ringed
catpath compare --states b_*.json --heights s_*.json --out ts.png  # TS heights, highest point starred
catpath <cfg>                            # shorthand for `run`
catpath run --substrate NO --target NH3 --element Pd --network auto   # no config file
# chemistry flags: --substrate --target --element --network
# overrides:       --backend --models --seeds 0,1,2 --reagents H,O --device cuda --name --outdir
# the config arg is optional; flags override whatever the file sets.
```

### Cross-model comparison (backends that can't share an env)

The ML backends have conflicting deps, so run `states` in each backend's venv,
then `compare` the JSONs. Each state becomes a box (its pooled distribution)
with one dot per sample coloured by model; states are packed into the fewest
non-overlapping columns (distinct energies share a column, clusters shift right):

```bash
.venv-chgnet/bin/catpath   states cmp.yaml --backend chgnet   --device cuda --out s_chgnet.json
.venv-fairchem/bin/catpath states cmp.yaml --backend fairchem --device cpu  --out s_fairchem.json
uv run catpath           states cmp.yaml --backend mace     --device cuda --out s_mace.json
uv run catpath compare --states s_*.json --out compare.png
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
| Add an MLIP backend | `calculators._load` (+ `_MODULE`/`_EXTRA`/`AUTO_ORDER`) |
| Add a reaction / intermediate | `network.build_network` (add a `StepSpec`), or `network: auto` to autodetect them (`explore.py`) |
| Change adsorption sites / poses | `structures.SITE_NAMES`, `poses`, `place_fragments` |
| Tune convergence criteria | `relax.relax`, `config.SearchConfig` |
| Change validation thresholds | `validate.geometry_ok`, `config.SearchConfig` |
| Multi-substrate energy map | pass more rows to `viz.energy_map` |

### Production accuracy

The `emt` backend is **not ML** — a placeholder to exercise the pipeline. Real
runs use a machine-learned potential; install exactly **one** ML backend per env
(their torch/e3nn pins conflict — uv treats the extras as mutually exclusive):

```bash
uv sync --extra mace        # or: --extra chgnet | --extra fairchem | --extra grace
# then set mlip.backend: mace (or auto), mlip.device: cuda in the config
```

`backend: auto` picks the best **installed** ML backend (mace → fairchem → grace
→ chgnet) and **errors if none is installed** rather than silently using EMT — so
you never mistake a semi-empirical smoke for a real result. `fairchem` (Meta
FAIRChem / UMA, OC20 task) is purpose-built for adsorbates on metal surfaces and
is the recommended production choice for the NO/Pd chemistry; its weights are
Hugging-Face-license-gated (needs a login).

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
