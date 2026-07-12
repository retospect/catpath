# catpath — backlog / TODO

Running list of deferred work so we don't forget. Newest asks captured; roughly
priority-ordered within sections.

## In progress
- [ ] **(a) Structure thumbnails** — gallery ✅ done; still to wire into the
  pipeline and add on the energy profile and node/DAG graph, each in
  **with- and without-thumbnail** variants. Dual top+side, fixed camera/zoom.

## Next up (agreed order: a → c → b)
- [x] **(c) Reagent list config knob** — `reagents: [H, O]` filters the network to
  branches reachable with the allowed adatoms (reagent per supply link derived
  from stoichiometry). `[]` = dissociation only; omit = full template. CLI:
  `--reagents H,O`. Tests in `test_network.py`.
- [x] **(c) Multi-substrate networks** — `catpath multi` runs several
  `(substrate, target, network, reagents)` specs from `substrates:` and stacks
  them into one union-column energy map (NaN where a state is absent). See
  `multi.py`, `examples/multi_substrate.yaml`, `test_multi.py`.
- [x] **(b) Intermediate autodetection** — `network: auto` (`src/catpath/explore.py`).
  Rule-guided explorer over a molecular graph of the adsorbate: dissociate a
  heavy–heavy bond, supply a reagent adatom, bond it. Prunes by valence + atom
  budget + reachability to the target. Provably acyclic; step endpoints share an
  element ordering so NEB interpolates. Reagents default to target-minus-substrate.
  Verified end-to-end on EMT (NO→NH3, 26 states/23 steps). See
  `examples/auto_ammonia.yaml`, `tests/test_explore.py`. STILL OPEN below.

## Packaging & release ("ready for GitHub")
- [x] **Name chosen: `catpath`** (PyPI-available; package renamed atosim→catpath).
- [x] **pip installable** — `pip install catpath` (+ `[mace]`/`[chgnet]`/
  `[fairchem]`/`[grace]` extras). `uv build` → wheel+sdist verified; clean-venv
  install runs the full pipeline end-to-end. GPL-3.0-or-later license + bundled
  LICENSE. README has install/quickstart/CLI + `examples/README.md` how-to.
- [x] **CI + trusted publishing** — `.github/workflows/ci.yml` (ruff + pytest on
  EMT, Python 3.11/3.12) and `.github/workflows/workflow.yml` (OIDC trusted
  publish to PyPI on release; env `pipy`, repo `retospect/catpath`).
- [x] **README with graphs** — badges + cross-model hero image (`docs/img/`).
- [ ] **Still open**: CONTRIBUTING.md, issue/PR templates; a minimal Python-API
  usage example in the README; publish the first release (tag → the workflow
  auto-publishes). Optionally embed a reaction-profile + gallery image too.
- [ ] **First PyPI release** — create the `pipy` GitHub Environment (Settings →
  Environments), tag `v0.1.0`, cut a GitHub Release; `workflow.yml` publishes.

## HPC / scale
- [ ] **SLURM integration** — submit runs to a cluster. Snakemake has a native
  SLURM executor (`--executor slurm`); wire the seed/model fan-out to it and
  document resource requests (GPU per job). Also a plain `sbatch` wrapper.

## Visualization / design (dedicated pass)
- [x] **POV-Ray render backend** — `render.backend: povray` ray-traces the
  active-site thumbnails/gallery (same fixed top+side camera/zoom as matplotlib);
  falls back to matplotlib with a warning if the `povray` binary is absent.
  See `render.py`, `examples/render_povray.yaml`, `test_render.py`. Binary
  installed + verified end-to-end (gallery + profile/DAG thumbnails ray-trace,
  shared camera/zoom holds). Still open: adsorbate atoms read small vs the slab
  — see "Tighten thumbnail zoom / emphasize reagent atoms" below.
- [ ] **Discuss + choose color schemes** — consistent palette across profile,
  energy map, and gallery; emphasize adsorbate atoms vs slab; light/dark
  variants; colorblind-safe; per-pathway colors; CPK vs custom element colors.
- [ ] Better layouts, interactive HTML export, per-model overlay on the profile.
- [ ] Tighten thumbnail zoom / emphasize reagent atoms (currently small vs slab).

## Physics / method
- [x] **`network: auto` geometry tuning** — materialised endpoints now use
  covalent-radius bond lengths, an upward substituent cone (66° lean so chain
  tips like N-O-H stay under the 3.5 Å detachment limit), covalent-radius anchor
  heights, footprint-based fragment spacing, and distinct sites (primary at fcc,
  co-adsorbed byproducts/staged reagents at hcp). Eliminated the spurious
  "detached from slab" warnings; representative steps (NO+H→HNO, NH2+H+O→NH3+O)
  converge cleanly under a real NEB budget (Ea 0.08 / 0.15 eV). NOTE: remaining
  NEB non-convergence in quick smokes is *band-budget* limited (small
  neb_images/neb_max_steps), not geometry — see "NEB auto-retry" below.
- [x] **`network: auto` scale controls** — `auto:` config block exposes
  `max_extra` (reagent-atom budget), `max_states` (hard breadth cap), and
  `prune_energy` (deterministic rough-energy pruning that drops branches >N eV
  above the root, keeping only what still connects root→target; skips itself if
  it would sever the target). See `AutoConfig`, `explore.prune_by_rough_energy`,
  `tests/test_explore.py`.
- [x] **NEB auto-retry** — `search.neb_retries` (default 1). On non-convergence,
  retry with a ~1.5× denser band and 2× the step budget; return the first
  converged attempt else the most-refined one. Cut NO→NH3 EMT smoke
  non-convergence from 14/23 → 4/23. See `neb.neb_barrier`, `tests/test_neb.py`.
- [ ] **Strain sensitivity study** — option to vary lattice constant and measure
  how much adsorption energies shift (complements lattice relaxation).
- [x] **ML potential registry + `auto`** — `make_calculator` now dispatches
  emt | mace | chgnet | fairchem | grace via lazy imports (optional pip extras;
  uv `conflicts` keeps their incompatible torch pins in separate envs).
  `backend: auto` resolves to the best *installed* ML backend and **errors
  rather than silently using EMT**. Verified: `auto` -> MACE computes a real
  single-point (NO/Pd -70.4 eV). See `calculators.py`, `tests/test_calculators.py`.
- [ ] **fairchem / chgnet / grace first real run** — the three new backends are
  wired to each library's current ASE-calculator API but were authored without
  the package installed (only mace+emt are here). Install one and do a real
  single-point/relax to confirm the API call + `task`/model defaults; adjust if
  the upstream signature drifted. Then compare adsorption energies vs MACE.
- [ ] **Ensemble / committee uncertainty** — beyond seeds×models; per-model
  breakdown in `results.json`.
- [ ] **GPU**: drop the nvrtc pin once an aarch64 `cu130` torch wheel exists.

## Testing
- [ ] Integration test for thumbnail/gallery rendering (EMT).
- [ ] Golden-file checks for provenance caption / methods text.
