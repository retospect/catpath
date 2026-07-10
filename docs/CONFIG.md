# Configuration reference — how you tell atosim what to do

A run is **one YAML file**: `atosim run my.yaml`. Every field below, with the
honest status of each "how many / which" axis you might want to control.

```yaml
name: nh3_run                # output folder name (runs/<name>/)
substrate: "NO"              # starting species label (QUOTE it — bare NO = YAML false)
target: "NH3"                # ending species label
network: ammonia             # which reaction network template (see §Intermediates)

slab:                        # the ENVIRONMENT (catalyst surface)
  element: Pd                # any ASE fcc metal (EMT: Pd Pt Cu Ni Ag Au Al)
  size: [3, 3, 4]            # nx, ny, layers
  vacuum: 10.0               # Å of vacuum above the slab
  fix_layers: 2              # freeze this many bottom layers
  relax_lattice: true        # fit the lattice constant to the potential (removes strain)
  a: null                    # or pin the lattice constant (Å) explicitly

mlip:                        # the POTENTIAL(s)
  backend: mace              # emt | mace | fairchem
  models: ["small","medium"] # HOW MANY / WHICH models  -> see §Models
  model: null                # single-model shorthand (used if `models` empty)
  device: cuda               # cpu | cuda

search:
  seeds: [0, 1, 2]           # HOW MANY / WHICH seeds   -> see §Seeds
  neb_images: 7              # NEB band images (TS resolution)
  fmax: 0.05                 # relaxation force convergence (eV/Å)
  max_steps: 200
  neb_fmax: 0.1              # NEB band convergence
  neb_max_steps: 150
  rmsd_thresh: 0.7           # Å, "same structure" threshold
  energy_thresh: 0.05        # eV, spread over this -> low_confidence flag

substrates: []               # multi-substrate rows -> see §Substrates (LIMITED)
outdir: runs
```

## §Seeds — fully controllable ✅
`search.seeds: [0, 1, 2]` — an explicit list. **How many** = length; **which** =
the values (they seed the pose perturbations deterministically). Use ≥3 for a
mean ± spread; bump to 5–10 if a state is flagged low-confidence.

## §Models — fully controllable ✅
`mlip.models: ["small", "medium"]` — an explicit list; **how many** = length,
**which** = the names. Each entry is either a model name (uses `mlip.backend`) or
`backend:model` (e.g. `mace:small`, `fairchem:<ckpt>`) to mix backends. Every
`(model × seed)` combination is run and pooled, so error bars capture **both**
model and seed variance. Leave `models: []` and set `model:` for a single model.

## §Reagents — template-driven, NOT yet a config list ⚠️
The reagents in the NO→NH₃ chemistry are the adatoms **O\*** and **H\*** that get
added along the path. Right now they are **baked into the chosen `network`
template** (the `+O*` / `+H*` "supply" steps and the composite states like
`NO+H`), *not* specified as a `reagents:` list. So today you pick reagents
implicitly by picking the network. There is no `reagents: [H, O]` knob yet —
adding one (that auto-inserts hydrogenation/oxidation steps) is a clean
extension.

## §Intermediates — curated templates, NOT autodetected ⚠️
**We do not autodetect intermediates.** They are **curated** in
`src/atosim/network.py` as three hand-built templates:

| `network:` | states | routes |
|---|---|---|
| `ammonia` (default) | 16 | dissociation, N-hydrogenation → NH₃, water branch, HNO/NOH fork, a site isomer |
| `branching` | 9 | dissociation + oxidation + reduction fork |
| `oxidation` | 4 | linear NO→NO₂→NO₃ |

To add/adjust intermediates you edit that file (add a `StateSpec` and a
`StepSpec`). The original vision of **rule-guided automatic** intermediate
generation (apply reaction templates / graph-rewrite rules, prune, expand) is
**not implemented** — it's the biggest open design item. What exists instead is
reliable, inspectable, hand-curated networks.

## §Substrates — single substrate per run (multi is limited) ⚠️
One run explores **one** substrate→target network. The `substrates:` list is
currently only a **label** for the energy-map row and does **not** launch
separate networks. Two real ways to get multiple rows today:
- **`atosim sweep --elements Pd,Pt,Cu,Ni`** — same network across surfaces
  (catalyst screen). Fully works.
- Run the tool once per substrate and combine the energy maps.

True multi-substrate (different starting molecules, each its own network, one
matrix) needs per-substrate network templates — another clean extension.

---

### Summary
| Axis | How you set it | Status |
|---|---|---|
| Seeds (how many / which) | `search.seeds: [...]` | ✅ first-class |
| Models (how many / which) | `mlip.models: [...]` | ✅ first-class |
| Reagents (which) | implied by `network:` | ⚠️ template-driven, no list yet |
| Intermediates | `network:` template + editing `network.py` | ⚠️ curated, **not** autodetected |
| Substrates (how many) | one per run; `sweep` for surfaces | ⚠️ single-substrate networks |
| Surface / lattice | `slab:` block (auto-relaxed) | ✅ first-class |
