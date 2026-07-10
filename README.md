# atosim

Reproducible, ML-potential reaction-pathway explorer. Given an **environment**
(a metal surface), a **substrate**, and a **target**, it explores candidate
reaction pathways, estimates energies and barriers (NEB), and emits a **reaction
graph** plus a **substrate × intermediate energy map** — with honest uncertainty
from **multi-seed** runs.

**First target:** `NO → NO₃` oxidation on **Pd(111)**.

> **Design principle:** stability is an acceptance criterion. If a result depends
> on the random seed, it's flagged low-confidence rather than reported as a
> precise number. See [`PLAN.md`](PLAN.md) for the full spec.

## Install

Uses [`uv`](https://docs.astral.sh/uv/). The default **EMT** backend is pure
numpy/ASE (supports Pd, N, O) — no GPU or torch needed, so it runs anywhere.

```bash
uv sync --extra dev --extra orchestration
uv pip install -e .
```

Production ML potentials are optional and pluggable (installed on the GPU box):

```bash
uv sync --extra mace   # MACE-MP-0 universal potential (GPU)
```

## Run

```bash
# whole pipeline (all seeds), in-process:
uv run atosim examples/no_to_no3_pd.yaml

# or orchestrated by Snakemake (fans seeds out across jobs, restartable):
uv run snakemake -s workflow/Snakefile -c4
```

Outputs land in `runs/<name>/`:

| File | Contents |
|---|---|
| `graph.png` | reaction graph, energy-ordered; red = low-confidence node |
| `energy_map.png` | substrate × intermediate heatmap; ★ = rate-limiting state |
| `results.json` | nodes, edges, barriers, mean ± spread, warnings |
| `nodes.csv` / `edges.csv` | machine-readable graph |
| `energy_map.csv` | the energy matrix |
| `config.snapshot.yaml` | provenance snapshot for exact reproduction |

## How it works

```
inputs → build network (NO+O→NO2→NO2+O→NO3)
       → per seed: rattle poses → cheap pre-relax → MLIP relax → CI-NEB barrier
       → aggregate across seeds (mean ± spread; flag unstable states)
       → reaction graph + energy map → validate → outputs
```

- **Pluggable MLIP** (`calculators.py`): `emt` (dev) · `mace` · `fairchem`.
- **Validation layers** (`validate.py`): RDKit sanitization (molecules only),
  geometry sanity (clashes / detachment), convergence, cross-seed stability.
- **Uncertainty** (`uncertainty.py`): mean ± std over seeds; `low_confidence`
  when the spread exceeds tolerance.

> ⚠️ The bundled **EMT** backend is qualitative only — it exercises the full
> pipeline without heavy dependencies. Switch `mlip.backend` to `mace` or
> `fairchem` for quantitative barriers.

## Development

```bash
uv run pytest            # 27 tests, ~1.3s (EMT, tiny slabs)
```

Module map: `config` · `calculators` · `structures` · `relax` · `neb` ·
`network` · `validate` · `uncertainty` · `graph` · `viz` · `pipeline` · `cli`.
See [`docs/USAGE.md`](docs/USAGE.md) for configuration and extension points.
