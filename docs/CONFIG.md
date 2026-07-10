# Configuration reference — how you tell atosim what to do

A run is **one YAML file**: `atosim run my.yaml`. Every field below, with the
honest status of each "how many / which" axis you might want to control.

```yaml
name: nh3_run                # output folder name (runs/<name>/)
substrate: "NO"              # starting species label (QUOTE it — bare NO = YAML false)
target: "NH3"                # ending species label
network: ammonia             # which reaction network template (see §Intermediates)
reagents: ["H"]              # WHICH adatoms are available (filters branches) -> see §Reagents
                             #   omit for the full template; [] = reagent-free (dissociation) only

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

substrates:                  # multi-substrate rows -> see §Substrates
  - {substrate: "NO", target: "NH3", network: ammonia}
  - {substrate: "NO", target: "NO3", network: oxidation}

render:                      # how active-site thumbnails/gallery are drawn
  backend: matplotlib        # matplotlib (flat, no deps) | povray (ray-traced) -> see §Rendering
  width: 320                 # povray canvas width per view (px)
  bonds: true                # povray ball-and-stick bonds

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

## §Reagents — first-class filter ✅
`reagents:` lists **which adatoms are available** (`["H"]`, `["H", "O"]`, `[]`).
Every supply link's required reagent is **derived from its stoichiometry** (the
element it adds), not hardcoded, so a link is kept only when its adatom is
allowed; any state left unreachable — and the steps touching it — is pruned.

| `reagents:` | effect on the `ammonia` template |
|---|---|
| *(omitted / `null`)* | full curated network (back-compat default) |
| `["H"]` | full reduction chain (ammonia uses only H\*) |
| `[]` | reagent-free only: dissociation + site isomer (`NO`, `N+O`, `NO@top`) |

On the `branching` template, `["O"]` keeps dissociation + oxidation (→NO₃) and
drops the reduction fork; `["H"]` does the reverse. Override at the CLI with
`--reagents H,O` (empty string `--reagents ""` = `[]`). This does **not** invent
new intermediates — it gates the curated ones by available reagent.

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

## §Substrates — multi-substrate via `atosim multi` ✅
Give `substrates:` a list of **spec dicts** (`{substrate, target, network,
reagents}`; omitted fields inherit the top-level values) and run
`atosim multi my.yaml`. Each entry gets its own full run + per-run artifacts,
and all rows are stacked into one **union-column** energy map
(`runs/<name>_multi/energy_map.png`): columns are the union of every network's
states, cells are blank/`NaN` where a substrate never visits that state, and the
★ rate-limiting marker tolerates the gaps. A bare-string entry (`- "NO"`) still
works and inherits the top-level target/network/reagents.

Two other multi-row paths:
- **`atosim sweep --elements Pd,Pt,Cu,Ni`** — same network across surfaces
  (catalyst screen); columns align exactly (intersection).
- Run once per substrate and combine the maps by hand.

## §Rendering — matplotlib default, optional POV-Ray ✅
`render.backend` picks how the active-site thumbnails and gallery are drawn:

| backend | look | dependency |
|---|---|---|
| `matplotlib` (default) | flat CPK circles, top+side, shared zoom | none |
| `povray` | ray-traced ball-and-stick, shadows/soft light, same fixed camera | `povray` binary |

Both use the **same fixed top+side cameras and zoom window** (computed from the
adsorbate's projected centroid), so states stay directly comparable across
backends. POV-Ray needs the binary (`sudo apt-get install -y povray`); if it is
absent, atosim **prints a warning, records it in `results.json`, and falls back
to matplotlib** — the run never fails. Override at the CLI is via the config
`render:` block (no dedicated flag).

---

### Summary
| Axis | How you set it | Status |
|---|---|---|
| Seeds (how many / which) | `search.seeds: [...]` | ✅ first-class |
| Models (how many / which) | `mlip.models: [...]` | ✅ first-class |
| Reagents (which) | `reagents: [...]` (filters branches) | ✅ first-class |
| Intermediates | `network:` template + editing `network.py` | ⚠️ curated, **not** autodetected |
| Substrates (how many) | `substrates: [...]` + `atosim multi` | ✅ multi-substrate |
| Surface / lattice | `slab:` block (auto-relaxed) | ✅ first-class |
| Rendering | `render.backend: matplotlib\|povray` | ✅ first-class (povray optional) |
