"""Reaction network: rule-guided intermediate generation.

For the first target we hard-code the well-studied NO oxidation network on a
metal(111) surface as a set of atom-conserving elementary steps::

    NO*  + O*  ->  NO2*
    NO2* + O*  ->  NO3*

Each step's reactant and product are built with the **same adsorbate atom order**
(so NEB can interpolate).  The extra oxygen is modelled as a co-adsorbed adatom
already present in the reactant (keeping atom counts equal across the step).

``build_network`` returns a generic structure so future substrates / templates
can plug in without changing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ase import Atoms

from .config import SlabConfig
from .structures import build_slab, place_fragments


@dataclass
class StateSpec:
    """A named adsorbate configuration, defined by fragment placements."""

    name: str
    label: str  # molecule-like formula for display / RDKit (e.g. "NO2")
    specs: list[dict]

    def build(self, slab: Atoms) -> Atoms:
        return place_fragments(slab, self.specs)


@dataclass
class StepSpec:
    name: str
    reactant: StateSpec
    product: StateSpec


@dataclass
class Network:
    slab_cfg: SlabConfig
    steps: list[StepSpec] = field(default_factory=list)

    def slab(self) -> Atoms:
        return build_slab(self.slab_cfg)

    def states(self) -> dict[str, StateSpec]:
        out: dict[str, StateSpec] = {}
        for st in self.steps:
            out[st.reactant.name] = st.reactant
            out[st.product.name] = st.product
        return out


# --- NO -> NO3 template ------------------------------------------------------

def _no_plus_o() -> StateSpec:
    return StateSpec(
        "NO+O", "NO",
        [
            {"symbol": "N", "site": "fcc", "height": 1.8},
            {"symbol": "O", "site": "fcc", "height": 3.0},          # the N-O oxygen
            {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.2},  # adatom O*
        ],
    )


def _no2() -> StateSpec:
    return StateSpec(
        "NO2", "NO2",
        [
            {"symbol": "N", "site": "fcc", "height": 2.0},
            {"symbol": "O", "site": "fcc", "height": 2.4, "dx": 0.9, "dy": 0.6},
            {"symbol": "O", "site": "fcc", "height": 2.4, "dx": -0.9, "dy": 0.6},
        ],
    )


def _no2_plus_o() -> StateSpec:
    return StateSpec(
        "NO2+O", "NO2",
        [
            {"symbol": "N", "site": "fcc", "height": 2.0},
            {"symbol": "O", "site": "fcc", "height": 2.4, "dx": 0.9, "dy": 0.6},
            {"symbol": "O", "site": "fcc", "height": 2.4, "dx": -0.9, "dy": 0.6},
            {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.4},  # adatom O*
        ],
    )


def _no3() -> StateSpec:
    return StateSpec(
        "NO3", "NO3",
        [
            {"symbol": "N", "site": "fcc", "height": 2.2},
            {"symbol": "O", "site": "fcc", "height": 2.5, "dx": 1.1, "dy": 0.0},
            {"symbol": "O", "site": "fcc", "height": 2.5, "dx": -0.55, "dy": 0.95},
            {"symbol": "O", "site": "fcc", "height": 2.5, "dx": -0.55, "dy": -0.95},
        ],
    )


def build_network(slab_cfg: SlabConfig) -> Network:
    """The NO -> NO2 -> NO3 oxidation network."""
    return Network(
        slab_cfg=slab_cfg,
        steps=[
            StepSpec("NO+O->NO2", _no_plus_o(), _no2()),
            StepSpec("NO2+O->NO3", _no2_plus_o(), _no3()),
        ],
    )
