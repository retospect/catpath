"""Pluggable ML-potential interface.

The whole pipeline talks to a calculator only through :func:`make_calculator`,
which returns a fresh ASE calculator instance.  Backends are swappable without
touching the rest of the code:

* ``emt``      - ASE built-in Effective Medium Theory.  Supports Pd, N, O, H;
                 pure-numpy, installs anywhere, no GPU.  **Default for dev/CI.**
                 *Not quantitatively accurate* - a placeholder to exercise the
                 full pipeline.
* ``mace``     - MACE-MP-0 universal potential (production accuracy, GPU).
* ``fairchem`` - Open Catalyst models, purpose-built for adsorbates on metals.

``mace`` / ``fairchem`` are imported lazily so the light EMT path never needs
torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

from .config import MLIPConfig

# Elements EMT has parameters for (guard so we fail loudly, not silently wrong).
EMT_ELEMENTS = {"Ag", "Al", "Au", "C", "Cu", "H", "N", "Ni", "O", "Pd", "Pt"}


def make_calculator(cfg: MLIPConfig) -> "Calculator":
    backend = cfg.backend.lower()
    if backend == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    if backend == "mace":
        from mace.calculators import mace_mp  # type: ignore

        return mace_mp(model=cfg.model or "medium", device=cfg.device)
    if backend == "fairchem":
        # fairchem >=1.0 exposes an ASE calculator factory; kept lazy on purpose.
        from fairchem.core import OCPCalculator  # type: ignore

        return OCPCalculator(checkpoint_path=cfg.model, cpu=(cfg.device == "cpu"))
    raise ValueError(f"unknown MLIP backend: {cfg.backend!r}")


def check_supported(symbols: set[str], cfg: MLIPConfig) -> None:
    """Raise early if the chosen backend cannot handle these elements."""
    if cfg.backend.lower() == "emt":
        missing = symbols - EMT_ELEMENTS
        if missing:
            raise ValueError(
                f"EMT backend does not support elements {sorted(missing)}; "
                "use backend='mace' or 'fairchem'."
            )
