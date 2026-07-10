"""Deterministic caption / methods generation from run metadata.

This is a *template*, not an LLM: the same run always yields the same text, which
is what provenance requires.  It produces (1) a one-line caption for the figure
footer, (2) a full methods paragraph for a sidecar / report, and (3) a dict of
PNG tEXt metadata embedded directly in the image.
"""

from __future__ import annotations

from . import __version__
from .config import Config


def _models(cfg: Config, results) -> list[str]:
    if getattr(results, "models", None):
        return results.models
    m = cfg.mlip.model
    return [f"{cfg.mlip.backend}:{m}" if m else cfg.mlip.backend]


def caption(cfg: Config, results) -> str:
    """One-line footer summarising how the figure was produced."""
    models = _models(cfg, results)
    n = len(models) * len(cfg.search.seeds)
    return (
        f"potential: {', '.join(models)}  |  device: {cfg.mlip.device}  |  "
        f"relax: BFGS (fmax {cfg.search.fmax} eV/A)  |  "
        f"TS: CI-NEB ({cfg.search.neb_images} images)  |  "
        f"{len(models)} model(s) x {len(cfg.search.seeds)} seed(s) = {n} samples  |  "
        f"band / Ea+- = 1 s.d.  |  atosim v{__version__}"
    )


def methods_text(cfg: Config, results) -> str:
    """Full methods paragraph (Markdown) for a sidecar file or a report."""
    models = _models(cfg, results)
    n = len(models) * len(cfg.search.seeds)
    nx, ny, nz = cfg.slab.size
    n_states = len(results.node_energies)
    n_rxn = len([e for e in results.edges])
    return f"""\
# Methods — {cfg.substrate} -> {cfg.target} on {cfg.slab.element}

Reaction pathways for **{cfg.substrate} -> {cfg.target}** on a
**{cfg.slab.element}({nx}x{ny}) {nz}-layer (111) slab** (bottom {cfg.slab.fix_layers}
layers fixed, {cfg.slab.vacuum} A vacuum) were explored with the *atosim* pipeline
(v{__version__}). The `{cfg.network}` reaction network comprised {n_states} adsorbate
states and {n_rxn} elementary steps.

Energies and forces were evaluated with the machine-learned interatomic
potential(s): **{', '.join(models)}** (device: {cfg.mlip.device}). For each of
{len(cfg.search.seeds)} random seed(s) {list(cfg.search.seeds)} and each model,
adsorbate poses were perturbed, pre-relaxed, then relaxed with BFGS to a maximum
force of {cfg.search.fmax} eV/A. Transition states were located with
climbing-image NEB ({cfg.search.neb_images} images, converged to
{cfg.search.neb_fmax} eV/A). Adsorbate states were referenced to the substrate
per run before pooling, so reported values combine **{len(models)} model(s) x
{len(cfg.search.seeds)} seed(s) = {n} samples**; all energies and barriers are
means +- 1 standard deviation across those samples. States/barriers whose spread
exceeds {cfg.search.energy_thresh} eV are flagged low-confidence.

Validation: RDKit sanitisation (molecule-like species), post-relaxation geometry
checks (clashes / desorption), NEB convergence, and cross-seed/model stability.
"""


def png_metadata(cfg: Config, results) -> dict[str, str]:
    """PNG tEXt chunks embedded in the image (machine-readable provenance)."""
    return {
        "Title": f"{cfg.substrate}->{cfg.target} on {cfg.slab.element} reaction profile",
        "Software": f"atosim v{__version__}",
        "Description": methods_text(cfg, results),
        "Comment": caption(cfg, results),
    }
