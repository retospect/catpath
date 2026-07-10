# atosim — backlog / TODO

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
- [x] **(c) Multi-substrate networks** — `atosim multi` runs several
  `(substrate, target, network, reagents)` specs from `substrates:` and stacks
  them into one union-column energy map (NaN where a state is absent). See
  `multi.py`, `examples/multi_substrate.yaml`, `test_multi.py`.
- [ ] **(b) Intermediate autodetection** — the big one. Rule-guided explorer:
  apply reaction templates / graph-rewrite rules to generate intermediates,
  prune (valence/charge/reactive-distance/rough-energy), expand around best
  paths. Replaces hand-curated `network.py` templates. This is the original spec
  vision, currently NOT implemented.

## Packaging & release ("ready for GitHub")
- [ ] **Pick a better project name** before shipping to PyPI (atosim is a
  placeholder; check name availability on PyPI/GitHub).
- [ ] **pip installable** — it's already a hatchling project; add build/publish
  (PyPI) and document `pip install atosim`. Write the **pip interface how-to**:
  install, `atosim run ...`, minimal Python API example.
- [ ] **README with finished graphs** — embed example profile / gallery / energy
  map images; add badges, quickstart, screenshots.
- [ ] **GitHub-ready** — LICENSE file, CI (GitHub Actions: pytest on EMT),
  CONTRIBUTING, issue templates, tidy `.gitignore`.

## HPC / scale
- [ ] **SLURM integration** — submit runs to a cluster. Snakemake has a native
  SLURM executor (`--executor slurm`); wire the seed/model fan-out to it and
  document resource requests (GPU per job). Also a plain `sbatch` wrapper.

## Visualization / design (dedicated pass)
- [x] **POV-Ray render backend** — `render.backend: povray` ray-traces the
  active-site thumbnails/gallery (same fixed top+side camera/zoom as matplotlib);
  falls back to matplotlib with a warning if the `povray` binary is absent.
  See `render.py`, `examples/render_povray.yaml`, `test_render.py`. NOTE: the
  binary still needs a manual `sudo apt-get install -y povray` to actually
  ray-trace; until then it transparently falls back. Once installed, do a visual
  pass to tune bbox/zoom parity, lighting, and adsorbate emphasis.
- [ ] **Discuss + choose color schemes** — consistent palette across profile,
  energy map, and gallery; emphasize adsorbate atoms vs slab; light/dark
  variants; colorblind-safe; per-pathway colors; CPK vs custom element colors.
- [ ] Better layouts, interactive HTML export, per-model overlay on the profile.
- [ ] Tighten thumbnail zoom / emphasize reagent atoms (currently small vs slab).

## Physics / method
- [ ] **NEB auto-retry** — on non-convergence, retry with more images/steps or
  a different interpolation before reporting a (spurious) barrier.
- [ ] **Strain sensitivity study** — option to vary lattice constant and measure
  how much adsorption energies shift (complements lattice relaxation).
- [ ] **fairchem backend** — wire Open Catalyst models; compare vs MACE.
- [ ] **Ensemble / committee uncertainty** — beyond seeds×models; per-model
  breakdown in `results.json`.
- [ ] **GPU**: drop the nvrtc pin once an aarch64 `cu130` torch wheel exists.

## Testing
- [ ] Integration test for thumbnail/gallery rendering (EMT).
- [ ] Golden-file checks for provenance caption / methods text.
