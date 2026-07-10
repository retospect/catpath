"""Reaction network: rule-guided intermediate generation (branching DAG).

Two networks are available:

* ``oxidation`` - the minimal linear chain NO+O -> NO2 -> NO2+O -> NO3.
* ``branching`` (default) - a richer DAG rooted at adsorbed NO with THREE
  competing pathways, so the reaction graph shows multiple routes:

      dissociation:  NO -> N + O
      oxidation:     NO -(+O*)-> NO+O -> NO2 -(+O*)-> NO2+O -> NO3
      reduction:     NO -(+H*)-> NO+H -> HNO      (H binds N)
                                 NO+H -> NOH      (H binds O)   <- the fork

Every **reaction** step is atom-conserving so NEB can interpolate; the extra
adatom (O* or H*) is carried in the reactant.  **Supply** links (``+O*`` / ``+H*``)
bridge states of different stoichiometry and carry no barrier - they only wire
the graph together.  Adding more branches (NH -> NH2 -> NH3, OH -> H2O, ...) is
just more ``StepSpec``/link entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
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
    links: list[tuple[str, str]] = field(default_factory=list)  # supply edges

    def slab(self) -> Atoms:
        return build_slab(self.slab_cfg)

    def states(self) -> dict[str, StateSpec]:
        out: dict[str, StateSpec] = {}
        for st in self.steps:
            out.setdefault(st.reactant.name, st.reactant)
            out.setdefault(st.product.name, st.product)
        return out

    def order(self) -> list[str]:
        """Topological ordering of all states (columns of the energy map)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.states())
        for s in self.steps:
            g.add_edge(s.reactant.name, s.product.name)
        for a, b in self.links:
            g.add_edge(a, b)
        try:
            return list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            return list(self.states())


# --- state library (placements chosen so fragments do not overlap) ----------

def _NO() -> StateSpec:
    return StateSpec("NO", "NO", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "O", "site": "fcc", "height": 3.0},
    ])


def _N_O() -> StateSpec:  # dissociated N* + O*
    return StateSpec("N+O", "N.O", [
        {"symbol": "N", "site": "fcc", "height": 1.6},
        {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.4},
    ])


def _NO_O() -> StateSpec:
    return StateSpec("NO+O", "NO", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "O", "site": "fcc", "height": 3.0},
        {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.2},
    ])


def _NO2() -> StateSpec:
    return StateSpec("NO2", "NO2", [
        {"symbol": "N", "site": "fcc", "height": 2.0},
        {"symbol": "O", "site": "fcc", "height": 2.4, "dx": 0.9, "dy": 0.6},
        {"symbol": "O", "site": "fcc", "height": 2.4, "dx": -0.9, "dy": 0.6},
    ])


def _NO2_O() -> StateSpec:
    return StateSpec("NO2+O", "NO2", [
        {"symbol": "N", "site": "fcc", "height": 2.0},
        {"symbol": "O", "site": "fcc", "height": 2.4, "dx": 0.9, "dy": 0.6},
        {"symbol": "O", "site": "fcc", "height": 2.4, "dx": -0.9, "dy": 0.6},
        {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.4},
    ])


def _NO3() -> StateSpec:
    return StateSpec("NO3", "NO3", [
        {"symbol": "N", "site": "fcc", "height": 2.2},
        {"symbol": "O", "site": "fcc", "height": 2.5, "dx": 1.1, "dy": 0.0},
        {"symbol": "O", "site": "fcc", "height": 2.5, "dx": -0.55, "dy": 0.95},
        {"symbol": "O", "site": "fcc", "height": 2.5, "dx": -0.55, "dy": -0.95},
    ])


def _NO_H() -> StateSpec:  # NO* + H* (H as a separate adatom)
    return StateSpec("NO+H", "NO", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "O", "site": "fcc", "height": 3.0},
        {"symbol": "H", "site": "hcp", "height": 1.2, "dx": 2.2},
    ])


def _HNO() -> StateSpec:  # H bound to N
    return StateSpec("HNO", "HNO", [
        {"symbol": "N", "site": "fcc", "height": 1.9},
        {"symbol": "O", "site": "fcc", "height": 3.1},
        {"symbol": "H", "site": "fcc", "height": 1.9, "dx": 1.0, "dy": 0.3},
    ])


def _NOH() -> StateSpec:  # H bound to O
    return StateSpec("NOH", "NOH", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "O", "site": "fcc", "height": 3.0},
        {"symbol": "H", "site": "fcc", "height": 3.6},  # atop the O
    ])


# --- ammonia (NO reduction) states: N* hydrogenation chain -------------------

def _N_H() -> StateSpec:  # N* + H*
    return StateSpec("N+H", "N", [
        {"symbol": "N", "site": "fcc", "height": 1.6},
        {"symbol": "H", "site": "hcp", "height": 1.1, "dx": 2.2},
    ])


def _NH() -> StateSpec:
    return StateSpec("NH", "NH", [
        {"symbol": "N", "site": "fcc", "height": 1.7},
        {"symbol": "H", "site": "fcc", "height": 2.8},
    ])


def _NH_H() -> StateSpec:  # NH* + H*
    return StateSpec("NH+H", "NH", [
        {"symbol": "N", "site": "fcc", "height": 1.7},
        {"symbol": "H", "site": "fcc", "height": 2.8},
        {"symbol": "H", "site": "hcp", "height": 1.1, "dx": 2.2},
    ])


def _NH2() -> StateSpec:
    return StateSpec("NH2", "NH2", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "H", "site": "fcc", "height": 2.8, "dx": 0.9, "dy": 0.5},
        {"symbol": "H", "site": "fcc", "height": 2.8, "dx": -0.9, "dy": 0.5},
    ])


def _NH2_H() -> StateSpec:  # NH2* + H*
    return StateSpec("NH2+H", "NH2", [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "H", "site": "fcc", "height": 2.8, "dx": 0.9, "dy": 0.5},
        {"symbol": "H", "site": "fcc", "height": 2.8, "dx": -0.9, "dy": 0.5},
        {"symbol": "H", "site": "hcp", "height": 1.1, "dx": 2.4},
    ])


def _NH3() -> StateSpec:
    return StateSpec("NH3", "NH3", [
        {"symbol": "N", "site": "fcc", "height": 2.0},
        {"symbol": "H", "site": "fcc", "height": 2.9, "dx": 1.0, "dy": 0.0},
        {"symbol": "H", "site": "fcc", "height": 2.9, "dx": -0.5, "dy": 0.87},
        {"symbol": "H", "site": "fcc", "height": 2.9, "dx": -0.5, "dy": -0.87},
    ])


def build_oxidation_network(slab_cfg: SlabConfig) -> Network:
    """Minimal linear chain (used by the fast tests)."""
    return Network(slab_cfg, steps=[
        StepSpec("NO+O->NO2", _NO_O(), _NO2()),
        StepSpec("NO2+O->NO3", _NO2_O(), _NO3()),
    ])


def build_branching_network(slab_cfg: SlabConfig) -> Network:
    """Dissociation + oxidation + reduction, rooted at adsorbed NO."""
    return Network(
        slab_cfg,
        steps=[
            StepSpec("NO->N+O", _NO(), _N_O()),          # dissociation
            StepSpec("NO+O->NO2", _NO_O(), _NO2()),      # oxidation
            StepSpec("NO2+O->NO3", _NO2_O(), _NO3()),
            StepSpec("NO+H->HNO", _NO_H(), _HNO()),      # reduction, N-H
            StepSpec("NO+H->NOH", _NO_H(), _NOH()),      # reduction, O-H (fork)
        ],
        links=[
            ("NO", "NO+O"),      # +O*
            ("NO", "NO+H"),      # +H*
            ("NO2", "NO2+O"),    # +O*
        ],
    )


# --- site isomers ("adsorbed this way and that") and the water byproduct ------

def _NO_top() -> StateSpec:  # NO adsorbed at an ontop site (vs fcc hollow)
    return StateSpec("NO@top", "NO", [
        {"symbol": "N", "site": "ontop", "height": 1.9},
        {"symbol": "O", "site": "ontop", "height": 3.1},
    ])


def _O_H() -> StateSpec:  # O* + H*
    return StateSpec("O+H", "O", [
        {"symbol": "O", "site": "fcc", "height": 1.5},
        {"symbol": "H", "site": "hcp", "height": 1.1, "dx": 2.2},
    ])


def _OH() -> StateSpec:
    return StateSpec("OH", "OH", [
        {"symbol": "O", "site": "fcc", "height": 1.9},
        {"symbol": "H", "site": "fcc", "height": 2.9},
    ])


def _OH_H() -> StateSpec:  # OH* + H*
    return StateSpec("OH+H", "OH", [
        {"symbol": "O", "site": "fcc", "height": 1.9},
        {"symbol": "H", "site": "fcc", "height": 2.9},
        {"symbol": "H", "site": "hcp", "height": 1.1, "dx": 2.2},
    ])


def _H2O() -> StateSpec:
    return StateSpec("H2O", "O", [  # label "O" for RDKit-free display
        {"symbol": "O", "site": "fcc", "height": 2.2},
        {"symbol": "H", "site": "fcc", "height": 2.9, "dx": 0.9, "dy": 0.3},
        {"symbol": "H", "site": "fcc", "height": 2.9, "dx": -0.9, "dy": 0.3},
    ])


def build_ammonia_network(slab_cfg: SlabConfig) -> Network:
    """NO reduction to ammonia, rooted at adsorbed NO.

        dissociation:   NO -> N + O
        hydrogenation:  N -(+H*)-> N+H -> NH -(+H*)-> NH+H -> NH2 -(+H*)-> NH2+H -> NH3
        associative:    NO -(+H*)-> NO+H -> HNO   (fork)
                                    NO+H -> NOH
    """
    return Network(
        slab_cfg,
        steps=[
            StepSpec("NO->N+O", _NO(), _N_O()),        # dissociation
            StepSpec("NO->NO@top", _NO(), _NO_top()),  # site isomer (diffusion)
            StepSpec("N+H->NH", _N_H(), _NH()),        # N hydrogenation chain
            StepSpec("NH+H->NH2", _NH_H(), _NH2()),
            StepSpec("NH2+H->NH3", _NH2_H(), _NH3()),
            StepSpec("NO+H->HNO", _NO_H(), _HNO()),    # associative fork
            StepSpec("NO+H->NOH", _NO_H(), _NOH()),
            StepSpec("O+H->OH", _O_H(), _OH()),        # O -> water byproduct
            StepSpec("OH+H->H2O", _OH_H(), _H2O()),
        ],
        links=[
            ("NO", "NO+H"),      # +H*
            ("N+O", "N+H"),      # N branch: O* to reservoir, +H*
            ("N+O", "O+H"),      # O branch: N* to reservoir, +H*
            ("NH", "NH+H"),      # +H*
            ("NH2", "NH2+H"),    # +H*
            ("OH", "OH+H"),      # +H*
        ],
    )


def build_network(slab_cfg: SlabConfig, kind: str = "ammonia") -> Network:
    builders = {
        "oxidation": build_oxidation_network,
        "branching": build_branching_network,
        "ammonia": build_ammonia_network,
    }
    if kind not in builders:
        raise ValueError(f"unknown network kind: {kind!r}")
    return builders[kind](slab_cfg)
