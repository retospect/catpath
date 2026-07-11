# Configuration reference — how you tell atosim what to do

A run is **one YAML file**: `atosim run my.yaml`. Every field below, with the
honest status of each "how many / which" axis you might want to control.

```yaml
name: nh3_run                # output folder name (runs/<name>/)
substrate: "NO"              # starting species label (QUOTE it — bare NO = YAML false)
target: "NH3"                # ending species label
network: ammonia             # ammonia|branching|oxidation (curated) OR auto (see §Intermediates)
reagents: ["H"]              # WHICH adatoms are available (filters branches) -> see §Reagents
                             #   omit for the full template; [] = reagent-free (dissociation) only

slab:                        # the ENVIRONMENT (catalyst surface)
  element: Pd                # any ASE fcc metal (EMT: Pd Pt Cu Ni Ag Au Al)
  size: [3, 3, 4]            # nx, ny, layers
  vacuum: 10.0               # Å of vacuum above the slab
  fix_layers: 2              # freeze this many bottom layers
  relax_lattice: true        # fit the lattice constant to the potential (removes strain)
  a: null                    # or pin the lattice constant (Å) explicitly

mlip:                        # the POTENTIAL(s)   -> see §Potentials
  backend: auto              # emt | mace | chgnet | fairchem | grace | auto
  models: ["small","medium"] # HOW MANY / WHICH models  -> see §Models
  model: null                # single-model shorthand (used if `models` empty)
  device: cuda               # cpu | cuda
  task: null                 # FAIRChem/UMA task head (default "oc20": adsorbates on metals)

search:
  seeds: [0, 1, 2]           # HOW MANY / WHICH seeds   -> see §Seeds
  neb_images: 7              # NEB band images (TS resolution)
  fmax: 0.05                 # relaxation force convergence (eV/Å)
  max_steps: 200
  neb_fmax: 0.1              # NEB band convergence
  neb_max_steps: 150
  neb_retries: 1             # retry a non-converged NEB with a denser band + more steps
  rmsd_thresh: 0.7           # Å, "same structure" threshold
  energy_thresh: 0.05        # eV, spread over this -> low_confidence flag

auto:                        # only used when network: auto  -> see §Intermediates
  max_extra: 4               # reagent-atom budget = len(substrate atoms) + this
  max_states: 600            # safety cap on how many states the explorer generates
  prune_energy: null         # eV above root; drop rougher branches (null = keep all)

substrates:                  # multi-substrate rows -> see §Substrates
  - {substrate: "NO", target: "NH3", network: ammonia}
  - {substrate: "NO", target: "NO3", network: oxidation}

render:                      # how active-site thumbnails/gallery are drawn
  backend: matplotlib        # matplotlib (flat, no deps) | povray (ray-traced) -> see §Rendering
  width: 320                 # povray canvas width per view (px)
  bonds: true                # povray ball-and-stick bonds

outdir: runs
```

## §Potentials — pluggable ML backends ✅
`mlip.backend` picks who computes energies/forces. Every backend is an ASE
calculator behind a **lazy import**, so an uninstalled one costs nothing until
selected.

| backend | what | install | notes |
|---|---|---|---|
| `emt` | ASE Effective Medium Theory | *(none)* | **Not ML, not accurate** — dev/CI only. Pd Pt Cu Ni Ag Au Al C N O H. |
| `mace` | MACE-MP-0 universal | `pip install atosim[mace]` | GPU; solid general default. |
| `chgnet` | CHGNet universal | `pip install atosim[chgnet]` | CPU-friendly. |
| `fairchem` | Meta FAIRChem / UMA | `pip install atosim[fairchem]` | Purpose-built for adsorbates-on-metals (OC20 task); UMA weights **license-gated** (HF login). |
| `grace` | GRACE foundation models | `pip install atosim[grace]` | TensorFlow-based. |
| `auto` | best **installed** ML backend | — | resolves in order mace → fairchem → grace → chgnet. |

The ML backends are *universal* (whole periodic table), so only EMT restricts
elements. **`backend: auto` raises if no ML potential is installed** — it never
silently drops to EMT, so a run that asked for a real potential either uses one
or tells you to install one. The resolved backend is logged and recorded in
`results.json` / `methods.md`.

⚠️ **You cannot install all backends in one environment** — their transitive
pins conflict (e.g. `mace-torch` needs `e3nn==0.4.4`, `fairchem-core` needs
`e3nn>=0.5`). uv treats the extras as mutually exclusive; install **one per env**
(or use `auto` with whatever is present). `fairchem`/`chgnet`/`grace` are wired to
each library's current ASE-calculator API but were authored without the package
installed here — expect a possible version tweak on first real use; `mace` and
`emt` are exercised in CI.

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

## §Intermediates — curated templates OR autodetected ✅
Two ways to get a network. **Curated** templates in `src/atosim/network.py` are
hand-built and tuned; **`network: auto`** derives the intermediates from rules.

### Curated templates
| `network:` | states | routes |
|---|---|---|
| `ammonia` (default) | 16 | dissociation, N-hydrogenation → NH₃, water branch, HNO/NOH fork, a site isomer |
| `branching` | 9 | dissociation + oxidation + reduction fork |
| `oxidation` | 4 | linear NO→NO₂→NO₃ |

To add/adjust intermediates you edit that file (add a `StateSpec` and a
`StepSpec`). Geometries are hand-placed and tuned.

### `network: auto` — rule-guided autodetection
Set `network: auto` and atosim **generates** the intermediates from
`substrate` → `target` (see `src/atosim/explore.py`). It applies three
elementary graph-rewrite rules to a molecular graph of the adsorbate:

- **dissociate** (barriered step) — break one heavy–heavy bond; the byproduct
  rides along as a co-adsorbed spectator (surface mass is conserved).
- **supply** (barrierless link) — stage one reagent adatom (`+H*`/`+O*`).
- **react** (barriered step) — bond the staged reagent to an atom with spare
  valence.

These compose into dissociation, hydrogenation chains, the associative HNO/NOH
fork and the water branch — the same routes as the curated `ammonia` template,
but *derived*. It then prunes by valence, an atom budget, and keeps only states
on a path from substrate to target. The result is provably a DAG, and every
reaction step's endpoints share an element ordering so NEB can interpolate.

```yaml
substrate: "NO"
target: "NH3"
network: auto      # reagents optional: derived as target-minus-substrate ({H} here)
```

`reagents:` is optional under `auto` — if omitted it defaults to the elements
the target needs more of than the substrate (NO→NH₃ ⇒ `{H}`; NO→NO₃ ⇒ `{O}`).
Pass it explicitly to restrict branches (e.g. `reagents: ["H"]`).

**Scale controls** (the `auto:` block) tame the exhaustive default network:

| knob | effect |
|---|---|
| `max_extra` | reagent-atom budget = `len(substrate atoms) + max_extra`. Lower ⇒ fewer intermediates. Must be large enough to reach the target *including co-adsorbed byproducts* — NO→NH₃ carries a spectator O, so NH₃+O needs 5 atoms ⇒ `max_extra ≥ 3` (default 4 has slack). |
| `max_states` | hard cap on generated states (safety limit for large substrates). |
| `prune_energy` | if set (eV), a fast deterministic pre-relax scores every state and branches more than this above the substrate are dropped, keeping only what still connects root→target. `null` keeps every path. |

Geometry: auto endpoints use covalent-radius bond lengths and an upward
substituent cone (so nothing floats off the slab), then relax — good enough that
representative steps converge cleanly under a real NEB budget, though a curated
template is still the tuned choice for a production run. `auto` is for
**discovery/coverage** of the pathway space. Example: `examples/auto_ammonia.yaml`.

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
| Potential (which backend) | `mlip.backend: emt\|mace\|chgnet\|fairchem\|grace\|auto` | ✅ pluggable (+ safe `auto`) |
| Seeds (how many / which) | `search.seeds: [...]` | ✅ first-class |
| Models (how many / which) | `mlip.models: [...]` | ✅ first-class |
| Reagents (which) | `reagents: [...]` (filters branches) | ✅ first-class |
| Intermediates | `network:` template, or `network: auto` (rule-guided) | ✅ curated **and** autodetected |
| Substrates (how many) | `substrates: [...]` + `atosim multi` | ✅ multi-substrate |
| Surface / lattice | `slab:` block (auto-relaxed) | ✅ first-class |
| Rendering | `render.backend: matplotlib\|povray` | ✅ first-class (povray optional) |
