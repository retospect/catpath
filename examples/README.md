# Examples

Every run is one YAML file. The configs here use the dependency-free **`emt`**
backend so they run anywhere; add `--backend auto` (or set `mlip.backend`) once
you have an ML potential installed (`pip install "autocatpath[mace]"` etc.). Full
field reference: [`../docs/CONFIG.md`](../docs/CONFIG.md).

## The obvious calls

```bash
# 1. A basic run: NO -> NO3 oxidation on Pd, all seeds, in-process.
autocatpath run examples/no_to_no3_pd.yaml

# 2. NO -> NH3 reduction on the curated ammonia network.
autocatpath run examples/no_to_nh3_pd.yaml

# 3. Let the intermediates be DISCOVERED from rules instead of curated.
autocatpath run examples/auto_ammonia.yaml

# 4. Restrict the network to the reagents you actually have (H only here).
autocatpath run examples/no_reduction_only.yaml

# 5. Pool several models for combined model+seed error bars.
autocatpath run examples/no_to_nh3_pd_multimodel.yaml     # needs autocatpath[mace]

# 6. Several substrates -> one union-column energy map.
autocatpath multi examples/multi_substrate.yaml

# 7. The same network across surfaces (catalyst screen).
autocatpath sweep examples/no_to_no3_pd.yaml --elements Pd,Pt,Cu,Ni

# 8. Ray-traced active-site thumbnails (falls back to matplotlib without povray).
autocatpath run examples/render_povray.yaml
```

## Comparing ML potentials

The ML backends have conflicting dependencies, so each lives in its own
environment. Run `states` / `barriers` in each, then `compare` the JSONs
(no config needed for `compare`):

```bash
# in each backend's env (here shown with explicit --backend overrides):
autocatpath states   examples/no_to_nh3_pd.yaml --backend mace     --out s_mace.json
autocatpath states   examples/no_to_nh3_pd.yaml --backend chgnet   --out s_chgnet.json
autocatpath states   examples/no_to_nh3_pd.yaml --backend fairchem --out s_uma.json

# intermediates: one box per state, dots per model, anchored at the substrate
autocatpath compare --states s_*.json --out intermediates.png

# barriers: Ea per elementary step, each model's rate-limiting step ringed
autocatpath barriers examples/no_to_nh3_pd.yaml --backend mace   --out b_mace.json
autocatpath barriers examples/no_to_nh3_pd.yaml --backend chgnet --out b_chgnet.json
autocatpath compare --states b_*.json --out barriers.png

# transition-state heights on a common scale, each model's highest point starred
autocatpath compare --states b_*.json --heights s_*.json --out ts_heights.png
```

Useful overrides on any run: `--backend`, `--device cuda`, `--seeds 0,1,2`,
`--reagents H,O`, `--name`, `--outdir`.

## The configs

| File | What it shows |
|---|---|
| `no_to_no3_pd.yaml` | basic oxidation run (`network: oxidation`) |
| `no_to_nh3_pd.yaml` | reduction on the curated `ammonia` network |
| `auto_ammonia.yaml` | `network: auto` — rule-based intermediate autodetection |
| `no_reduction_only.yaml` | `reagents:` filter (H only) |
| `no_to_nh3_pd_multimodel.yaml` | `mlip.models:` — pooled model+seed uncertainty |
| `multi_substrate.yaml` | `substrates:` — several rows for `autocatpath multi` |
| `render_povray.yaml` | `render.backend: povray` ray-traced thumbnails |
