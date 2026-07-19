"""Pluggable ML-potential interface.

The whole pipeline talks to a calculator only through :func:`make_calculator`,
which returns a fresh ASE calculator instance.  Backends are swappable without
touching the rest of the code; each machine-learned family is just an ASE
calculator behind a **lazy import** (so the light EMT path never needs torch, and
an uninstalled backend costs nothing until you select it):

* ``emt``      - ASE Effective Medium Theory.  Pure-numpy, installs anywhere, no
                 GPU.  **Not ML and not quantitatively accurate** - a dev/CI
                 placeholder to exercise the full pipeline.  Pd Pt Cu Ni Ag Au Al
                 C N O H only.
* ``mace``     - MACE-MP-0 universal potential (``pip install catpath[mace]``).
* ``chgnet``   - CHGNet universal potential (``pip install catpath[chgnet]``).
* ``fairchem`` - Meta FAIRChem / UMA, purpose-built for adsorbates on metals via
                 the OC20 task (``pip install catpath[fairchem]``; UMA weights are
                 license-gated -- needs a Hugging Face login).
* ``grace``    - GRACE foundation models (``pip install catpath[grace]``).
* ``auto``     - resolve to the best *installed* ML backend (see AUTO_ORDER);
                 raises if none is present rather than silently using EMT.

The ML backends are *universal* (trained across most of the periodic table), so
only EMT restricts the element set.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

from .config import MLIPConfig

# Elements EMT has parameters for (guard so we fail loudly, not silently wrong).
EMT_ELEMENTS = {"Ag", "Al", "Au", "C", "Cu", "H", "N", "Ni", "O", "Pd", "Pt"}

# backend -> the top-level module that must be importable for it to work.
_MODULE = {"mace": "mace", "chgnet": "chgnet",
           "fairchem": "fairchem", "grace": "tensorpotential"}
# backend -> the pip extra that installs it (for actionable error messages).
_EXTRA = {"mace": "mace", "chgnet": "chgnet",
          "fairchem": "fairchem", "grace": "grace"}
# `auto` preference order: reliably-usable first.  MACE downloads its weights
# with no login; FAIRChem/UMA is best for catalysis but its weights are gated.
AUTO_ORDER = ["mace", "fairchem", "grace", "chgnet"]

# Real ML potentials (everything the registry knows except EMT).
ML_BACKENDS = set(_MODULE)


def _installed(backend: str) -> bool:
    """True if the backend's package is importable (no heavy import performed)."""
    mod = _MODULE.get(backend)
    if mod is None:
        return backend == "emt"
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> dict[str, bool]:
    """Map every known backend to whether it can be loaded here (emt always)."""
    return {"emt": True, **{b: _installed(b) for b in _MODULE}}


def resolve_backend(backend: str) -> str:
    """Resolve ``auto`` to the best installed ML backend; pass others through.

    Raises if ``auto`` finds no ML potential installed -- deliberately, so a run
    that asked for a real potential never silently falls back to EMT.
    """
    if backend.lower() != "auto":
        return backend
    for cand in AUTO_ORDER:
        if _installed(cand):
            return cand
    hint = " ".join(f"catpath[{_EXTRA[b]}]" for b in AUTO_ORDER)
    raise RuntimeError(
        "backend: auto found no ML potential installed. Install one of: "
        f"{hint} (or set backend: emt explicitly for a non-ML smoke test)."
    )


def _load(backend: str, cfg: MLIPConfig):
    """Import and construct the ASE calculator for a concrete backend."""
    if backend == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    try:
        if backend == "mace":
            from mace.calculators import mace_mp

            # float64 is recommended for geometry optimization / NEB (vs float32).
            return mace_mp(model=cfg.model or "medium", device=cfg.device,
                           default_dtype="float64")
        if backend == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator

            return CHGNetCalculator(use_device=cfg.device)
        if backend == "fairchem":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            predictor = pretrained_mlip.get_predict_unit(
                cfg.model or "uma-s-1p1", device=cfg.device)
            return FAIRChemCalculator(predictor, task_name=cfg.task or "oc20")
        if backend == "grace":
            from tensorpotential.calculator import grace_fm

            return grace_fm(cfg.model or "GRACE-2L-OMAT")
    except ImportError as e:
        extra = _EXTRA.get(backend, backend)
        raise RuntimeError(
            f"backend '{backend}' is not installed: pip install catpath[{extra}] "
            f"(import failed: {e})"
        ) from e
    raise ValueError(f"unknown MLIP backend: {backend!r}")


# Process-lifetime cache of built ML calculators, keyed on the resolved backend
# plus the cfg fields that pick a model. An ML potential costs a full torch.load
# to construct (the MACE model is hundreds of MB, moved onto the GPU), and the
# pipeline builds a calculator per relaxation and per NEB image — so one run
# would otherwise do hundreds of identical loads, which dominates wall time and
# starves the GPU (idle between reloads). An ASE calculator recomputes on every
# Atoms it scores, so a single cached instance safely serves the whole run. EMT
# is pure-numpy and cheap, so it's never cached (keeps the light/CI path
# side-effect-free and hands back a genuinely fresh instance).
_CALC_CACHE: dict[tuple[str, str | None, str, str | None], "Calculator"] = {}


def make_calculator(cfg: MLIPConfig, *, cache: bool = True) -> "Calculator":
    """Return an ASE calculator for ``cfg`` (``auto`` resolved to a backend).

    ML backends are memoized for the process lifetime and reused for an identical
    ``(backend, model, device, task)`` — the model loads **once**, not once per
    relaxation / NEB image (the load, not the force eval, dominates). Pass
    ``cache=False`` for a guaranteed-fresh instance; EMT is always fresh.
    """
    backend = resolve_backend(cfg.backend)
    if not cache or backend == "emt":
        return _load(backend, cfg)
    key = (backend, cfg.model, cfg.device, cfg.task)
    calc = _CALC_CACHE.get(key)
    if calc is None:
        calc = _load(backend, cfg)
        _CALC_CACHE[key] = calc
    return calc


def reset_calculator_cache() -> None:
    """Drop cached ML calculators (frees GPU memory; mainly for tests)."""
    _CALC_CACHE.clear()


def check_supported(symbols: set[str], cfg: MLIPConfig) -> None:
    """Raise early if the chosen backend cannot handle these elements."""
    backend = resolve_backend(cfg.backend).lower()
    if backend == "emt":
        missing = symbols - EMT_ELEMENTS
        if missing:
            raise ValueError(
                f"EMT backend does not support elements {sorted(missing)}; "
                "use an ML backend (mace|chgnet|fairchem|grace) or backend: auto."
            )
    # ML backends are universal (whole periodic table) -> no element restriction.
