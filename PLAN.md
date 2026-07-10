# atosim — Reaction Pathway Explorer

A reproducible, Snakemake-orchestrated pipeline that, given a **substrate** (or a
set of substrates), an **environment** (e.g. a metal surface), optional
**reagents**, and optional **start/end states**, explores candidate reaction
pathways with an ML potential, estimates energies and barriers (including
transition regions via NEB), and emits a **reaction graph** plus a
**substrate × intermediate energy map** — with honest uncertainty from
**multi-seed / multi-model** runs.

**First target:** `NO → NO3` on a **Pd(111)** surface (see §15).

**Design principle:** stability is an acceptance criterion. If the answer depends
on the random seed, it is not an answer — it is a low-confidence flag. We run
multiple seeds and multiple models and report mean ± spread, not a single fake
precise number.

---

## 1. Goal

Turn `(environment, substrate(s), start, end)` into a validated, reproducible
reaction graph and energy map:

- Propose plausible intermediates via **rule-guided** exploration (not brute force).
- Place reagents/adsorbates at chemistry-informed sites with **multiple poses**.
- Cheap pre-relaxation → accurate ML relaxation → NEB transition/barrier search.
- Report **mean ± spread** of energies and barriers across seeds and models.
- Validate in layers; flag low-confidence results instead of overclaiming.

---

## 2. Inputs

| Input | Format | Notes |
|---|---|---|
| Substrate(s) | SMILES / SDF / xyz / ASE atoms | one or many (rows of the energy map) |
| Environment | `{surface: Pd(111), size, vacuum, solvent, T}` | surface built by ASE or from a file |
| Reagents / adatoms | list `{id, species, constraints}` | e.g. extra `O*` for oxidation |
| Start / End state | structure or label | defines the search target (NO → NO3) |
| MLIP config | `{backend, model(s), checkpoints, device}` | pluggable; EMT for dev, MACE/fairchem for prod |
| Search params | `{pose_count, neb_images, fmax, seeds, energy_thresh, rmsd_thresh}` | |

Everything collapses into **one YAML config** that is snapshotted per run.

---

## 3. Architecture

**Orchestration: Snakemake.** The pipeline is a Snakemake DAG — each stage
(pose-gen → pre-relax → filter → MLIP-relax → NEB → graph/energy-map) is a rule
with explicit inputs/outputs. This gives us for free: dependency tracking,
restart-from-failure, parallel job scheduling across seeds/models, and a
provenance record. Modes are just Snakemake targets (`compute`, `graph`,
`energymap`, `all`).

**One container, modular modes.** Target hardware: this **NVIDIA GB10 (DGX
Spark, ARM64 + Blackwell)** box. Avoid Docker-on-Mac (ARM/x86 mismatch, no GPU).

**Image layering (minimize rebuild churn):**
1. **Base** — OS + CUDA + heavy deps (ASE, RDKit, torch, MLIP). Rarely changes.
2. **App** — pipeline code, templates, viz. Rebuilds fast.
3. **Dev** — mount working dir; cache model weights in a mounted volume.

**MLIP is pluggable** behind one interface so backends swap without touching the
pipeline:
```python
class Calculator(Protocol):
    def energy_forces(self, atoms) -> tuple[float, np.ndarray]: ...
```
- **`emt`** — ASE built-in Effective Medium Theory. Supports Pd, N, O, H. Pure
  numpy, installs anywhere (incl. aarch64), no GPU. **Default for dev/CI/demo.**
  *Not quantitatively accurate — a placeholder to exercise the full pipeline.*
- **`mace`** — MACE-MP-0 universal potential. Production accuracy on GB10 GPU.
- **`fairchem`** — Open Catalyst (OC20/OC22) models, purpose-built for adsorbates
  on metal surfaces. Best fit for the NO/Pd chemistry.

### Stack
ASE (structures/dynamics/NEB) · RDKit (cheminformatics validation, thumbnails) ·
NetworkX (graph) · Matplotlib (PNG/SVG + energy-map heatmap) · PyYAML (config) ·
Snakemake (orchestration) · pytest (tests). Torch/MACE optional (prod backend).

---

## 4. Core workflow

```
inputs → propose intermediates → place adsorbates (N poses)
      → cheap pre-relax (EMT/FF, capped) → filter
      → MLIP relax (seeds × models) → NEB barrier search
      → graph + energy map → validate → outputs
```

### 4.1 Intermediate generation (guided, pruned)
Templates / functional-group reactivity / graph-rewrite rules propose
transformations; apply to the substrate graph; **cheap filter** (valence, charge,
reactive-distance, rough energy) before any expensive MLIP call; expand another
rule round around top paths. Allowed templates & scoring conditioned on
`environment`.

### 4.2 Adsorbate placement / active site
Never trust one placement. Generate a **pose ensemble** over adsorption sites
(top / bridge / fcc / hcp on Pd(111)), orientations, and heights near plausible
reactive atoms → cheap pre-relax → keep low-energy, chemically reasonable poses.
**Placed correctly ⇔** multiple poses relax to similar geometry & barrier.

### 4.3 Cheap pre-relaxation
EMT (or UFF/MMFF for molecules), short & constrained: limited steps, capped
per-step displacement, loose tol. Goal = clear clashes, not final geometry. Then
hand off to the accurate MLIP.

### 4.4 MLIP relaxation & transition search
Relax survivors. For each reacting state pair (atom-conserving), run **NEB**
(with climbing image) along the band to locate the transition state; barrier =
E(TS) − E(reactant). Record energy, barrier, convergence quality, metadata. (A
simple bracketed/binary search along a linear interpolation is available as a
cheap fallback when NEB is overkill.)

### 4.5 Uncertainty — two axes
- **Seeds (stochastic variance):** 3–5 per model to start; → 8–10 if spread large
  or path ranking flips.
- **Models (model uncertainty):** several checkpoints/architectures.
- Report **mean ± spread** per model + cross-model agreement on top pathways.
  Disagreement ⇒ report the spread as uncertainty.

---

## 5. Validation (layers)

1. **RDKit sanitization** — molecule-like inputs & intermediates only.
2. **Geometry sanity (post-relax)** — valences, bond lengths/angles, no
   too-close/too-far atoms, reacting atoms connect along a plausible path.
3. **Convergence** — stop when |ΔE| < thresh **and** fmax < force thresh for
   **several consecutive steps**.
4. **Stability** — similar results across seeds and ideally models.

> ⚠️ RDKit does **not** understand periodic surfaces / adsorbate `*` notation.
> Validate surface-adsorbed states via the MLIP / geometry checks; use RDKit only
> for molecule-like parts (this is exactly the NO/Pd situation).

Fail any layer ⇒ not done: retry with different init / tighter settings /
different model, or abandon the edge (with a logged reason).

### Similarity metrics
Structure: **RMSD** after alignment (molecules) or key distances/angles around
the reactive region (surfaces). Energy/barrier: compare means & spreads. Pathway:
same sequence of states & transition regions (different mechanism ≠ similar).

### Starter thresholds (guardrails; tighten with observed spread)
| Quantity | Start |
|---|---|
| RMSD (reactive region) | 0.5–1.0 Å |
| Barrier/energy tolerance (screening) | 1–2 kcal/mol (≈4–8 kJ/mol) |
| Force at convergence (fmax) | 0.01–0.05 eV/Å |
| Seeds per model | 3–5 (→ 8–10 if unstable) |

---

## 6. Graph construction
Directed graph; **nodes = states**, **edges = reactions**. Node props: energy
(mean ± spread), validation status, confidence. Edge props: barrier (mean ±
spread), optional rate-like score from barrier, convergence quality.
Low-confidence barriers explicitly marked.

## 7. Visualization
- **Reaction graph** — layered/force-directed (fixed layout for deterministic
  diffs), RDKit molecule thumbnails as nodes, energy/barrier labels. Export
  PNG/SVG + optional interactive HTML.
- **Substrate × intermediate energy map** — a matrix/heatmap: **one row per
  substrate**, **one column per intermediate state** along the path; cell color =
  state energy (relative to substrate reference). A **★ marks the highest-energy
  state in each row** (the rate-limiting peak). Optional "substrate vs target"
  reduced view (two columns: start vs end). Export PNG/SVG + CSV of the matrix.

## 8. Outputs
- **Results** — JSON/CSV: nodes, edges, energies, barriers, uncertainty.
- **Energy map** — CSV matrix + PNG/SVG heatmap with ★ peaks.
- **Graph visuals** — PNG/SVG (+ optional HTML).
- **Log** — seeds, model versions, params, warnings.
- **Provenance/config snapshot** — YAML to reproduce the run exactly (plus the
  Snakemake DAG).

---

## 9. Acceptance criteria
Bounded intermediates/path · energy tolerance met · readable graph & legible
thumbnails · runtime limits respected · **seed stability** (top pathways &
barrier ranking reproduce within tolerance, else flagged low-confidence) · all
validation layers pass or the edge is flagged/abandoned (never silently kept).

## 10. Failure modes & fallbacks
NEB non-convergent → retry different init / tighter settings; after N retries
abandon edge with logged reason. Placement unstable across poses → more poses /
constraints. High seed spread → more seeds / tighter convergence / downgrade
trust. Chemical nonsense post-relax → discard / re-init.

## 11. Reproducibility
Fixed seed policy, versioned models/checkpoints, full provenance logs, per-run
config snapshot, Snakemake DAG. Same inputs + same seed policy + same layout ⇒
reproducible outputs and meaningful diffs.

## 12. Evaluation
Benchmark reactions with known pathways (NO oxidation on Pd is well studied);
compare barriers qualitatively & quantitatively; track edge precision, barrier
accuracy, seed/model stability, graph usability.

---

## 13. Build milestones
- **M0 — Scaffold & env.** Package layout, pyproject, pytest, config loader +
  snapshot, EMT backend, `.venv`. *Exit:* toy Pd relax runs end-to-end.
- **M1 — Structure builder.** Pd(111) slab, adsorbate placement (sites/poses),
  constraints (fix bottom layers). *Exit:* NO on Pd(111) built & pre-relaxed.
- **M2 — MLIP interface + relaxation.** Pluggable calc, ASE relax, convergence
  (consecutive-step criterion). *Exit:* relax NO*/O* on Pd, metrics logged.
- **M3 — NEB barriers.** Interpolate reactant→product, climbing-image NEB, barrier
  extraction. *Exit:* barrier for `NO* + O* → NO2*`.
- **M4 — Intermediate generation.** Template proposer + pruning for the NO→NO3
  network. *Exit:* auto-enumerate NO→NO2→NO3 states.
- **M5 — Uncertainty.** Seed sweep + multi-model runner, mean/spread, stability &
  similarity (RMSD/energy/pathway). *Exit:* error bars + flags on M3.
- **M6 — Graph + energy map + validation wiring.** NetworkX graph, RDKit
  thumbnails, energy-map heatmap with ★. *Exit:* full artifacts for NO→NO3/Pd.
- **M7 — Snakemake orchestration.** Wrap stages as rules; targets = modes.
- **M8 — Evaluation harness.** Benchmarks, metrics, regression tracking.

## 14. Open questions
- Production MLIP: MACE-MP-0 vs fairchem/OC20 for NO/Pd? (drives base image)
- Reaction coordinate: IDPP interpolation + CI-NEB (default) vs learned?
- Template source for M4: curated vs reaction DB vs learned edit model?
- "snakemake as a substrate" — confirmed as **Snakemake orchestration** (not a
  molecule); revisit if a chemical substrate was meant.

## 15. First target — NO → NO3 on Pd(111)
Oxidation network, each elementary step atom-conserving (NEB-ready):
1. `NO* + O* → NO2*`
2. `NO2* + O* → NO3*`

Slab: `fcc111('Pd')`, e.g. 3×3×4 layers, ~10 Å vacuum, bottom 2 layers fixed.
Adsorption sites probed: top / bridge / fcc / hcp. Dev accuracy via EMT
(qualitative only); production via MACE/fairchem on the GB10 GPU. Deliverable: a
reaction graph NO→NO2→NO3 with barriers + a 1-row (or multi-substrate) energy map
starring the rate-limiting state.
