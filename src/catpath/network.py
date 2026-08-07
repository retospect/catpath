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

The ``ammonia`` network additionally comes in two **templates** (``kind="ammonia"``,
``template=`` on :func:`build_network`):

* ``"parked"`` (default) - after dissociation (``NO -> N+O``), whichever
  fragment is NOT being hydrogenated is approximated as parking off to a
  reservoir instantly (``N+O -> N+H`` drops the O, ``N+O -> O+H`` drops the
  N) -- a bookkeeping shortcut, not a modeled intermediate.
* ``"coadsorbed"`` - the verify tier that removes that approximation: both
  fragments stay in-cell (``N+O+H``, ``NH+O+H``, ...) until a product
  (H2O or NH3) actually desorbs.  See :func:`build_coadsorbed_ammonia_network`.
"""

from __future__ import annotations

from collections import Counter
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

    def adsorbate_counts(self) -> Counter:
        """Element -> count of adsorbate atoms (cheap; no slab build)."""
        c: Counter = Counter()
        for s in self.specs:
            c[s["symbol"]] += 1
        return c


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
    #: States that appear only as a link endpoint -- never a StepSpec
    #: reactant/product (e.g. the ``coadsorbed`` template's bare "N"/"O"
    #: landing spots after a desorption link). ``states()`` merges these in;
    #: ``pipeline.run_one_seed`` gives each a standalone relax (no partner,
    #: no NEB -- a link never carries a barrier) so it still gets a real
    #: energy and becomes a proper graph node.
    extra_states: list[StateSpec] = field(default_factory=list)
    #: An externally-prepared slab to score *instead of* building one from
    #: ``slab_cfg`` (the precis ``structure`` seam — the caller owns the
    #: geometry: alloy / adatom / facet / constraints). ``None`` = build from
    #: the label, as before. See :meth:`slab`.
    prebuilt_slab: Atoms | None = None

    def slab(self) -> Atoms:
        """The slab to run the reaction on.

        When a ``prebuilt_slab`` was injected we score *that* (a copy, so the
        caller's Atoms is never mutated) rather than building an fcc(111) slab
        from ``slab_cfg``. A prepared slab may lack ASE's ``adsorbate_info``
        (an extxyz round-trip drops that nested dict), which the named-site
        placement (``fcc``/``hcp``/``top``) needs; for the clean-fcc(111) first
        cut we transplant it from a reference slab built at the same ``slab_cfg``
        so placement still resolves. Edited/arbitrary slabs place adsorbates at
        an explicit anchor instead (the precis ``eye`` active-site, §7.5).
        """
        if self.prebuilt_slab is None:
            return build_slab(self.slab_cfg)
        s = self.prebuilt_slab.copy()
        if "n_slab" not in s.info:
            s.info["n_slab"] = len(s)
        if "adsorbate_info" not in s.info:
            ref = build_slab(self.slab_cfg)
            info = ref.info.get("adsorbate_info")
            if info is not None:
                s.info["adsorbate_info"] = info
        return s

    def states(self) -> dict[str, StateSpec]:
        out: dict[str, StateSpec] = {}
        for st in self.steps:
            out.setdefault(st.reactant.name, st.reactant)
            out.setdefault(st.product.name, st.product)
        for st in self.extra_states:
            out.setdefault(st.name, st)
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


def _ammonia_shared() -> tuple[list[StepSpec], list[tuple[str, str]]]:
    """The part of the ammonia network identical between the ``"parked"`` and
    ``"coadsorbed"`` templates: dissociation, the site isomer, the associative
    fork (``NO+H -> HNO | NOH``), the single-species N-hydrogenation chain
    (``N+H -> NH -> NH+H -> NH2 -> NH2+H -> NH3``), and the single-species
    O -> water chain (``O+H -> OH -> OH+H -> H2O``). Shared by reference (one
    definition) so the two templates never duplicate-drift.

    NOT included: the two ``N+O -> {N+H, O+H}`` **parking** links -- those are
    exactly the fragment-parking approximation the templates disagree on; each
    builder supplies its own continuation from ``N+O``.
    """
    steps = [
        StepSpec("NO->N+O", _NO(), _N_O()),        # dissociation
        StepSpec("NO->NO@top", _NO(), _NO_top()),  # site isomer (diffusion)
        StepSpec("N+H->NH", _N_H(), _NH()),        # N hydrogenation chain
        StepSpec("NH+H->NH2", _NH_H(), _NH2()),
        StepSpec("NH2+H->NH3", _NH2_H(), _NH3()),
        StepSpec("NO+H->HNO", _NO_H(), _HNO()),    # associative fork
        StepSpec("NO+H->NOH", _NO_H(), _NOH()),
        StepSpec("O+H->OH", _O_H(), _OH()),        # O -> water byproduct
        StepSpec("OH+H->H2O", _OH_H(), _H2O()),
    ]
    links = [
        ("NO", "NO+H"),      # +H*
        ("NH", "NH+H"),      # +H*
        ("NH2", "NH2+H"),    # +H*
        ("OH", "OH+H"),      # +H*
    ]
    return steps, links


def build_ammonia_network(slab_cfg: SlabConfig) -> Network:
    """NO reduction to ammonia, rooted at adsorbed NO (the ``"parked"`` template).

        dissociation:   NO -> N + O
        hydrogenation:  N -(+H*)-> N+H -> NH -(+H*)-> NH+H -> NH2 -(+H*)-> NH2+H -> NH3
        associative:    NO -(+H*)-> NO+H -> HNO   (fork)
                                    NO+H -> NOH

    After dissociation, whichever fragment is not being hydrogenated is
    approximated as parking off to a reservoir instantly (``N+O -> N+H``
    drops the O, ``N+O -> O+H`` drops the N) -- see
    :func:`build_coadsorbed_ammonia_network` for the verify tier that removes
    this approximation.
    """
    steps, links = _ammonia_shared()
    return Network(
        slab_cfg,
        steps=steps,
        links=[
            *links,
            ("N+O", "N+H"),      # N branch: O* to reservoir, +H* (PARKING)
            ("N+O", "O+H"),      # O branch: N* to reservoir, +H* (PARKING)
        ],
    )


# --- coadsorbed template: the dissociative branch without fragment parking --

def _fork(reactant_specs: list[dict], moves: dict[int, dict]) -> list[dict]:
    """Copy ``reactant_specs``, repositioning the atom(s) at ``moves`` (index
    -> partial spec overrides) -- the "chemical fork" transformation (a
    free/newly-supplied H binds to one fragment or the other). Every other
    atom's spec is copied unchanged and every index keeps its symbol, so a
    step built from ``(reactant_specs, _fork(reactant_specs, ...))`` is always
    atom-for-atom order-matched -- the invariant NEB interpolation needs (see
    the :mod:`catpath.structures` module docstring).
    """
    out = [dict(s) for s in reactant_specs]
    for idx, new_pos in moves.items():
        out[idx] = {**out[idx], **new_pos}
    return out


# free/unbonded H, awaiting a fork (bonds to whichever fragment "wins")
_H_FREE = {"symbol": "H", "site": "bridge", "height": 1.1, "dx": -2.2, "dy": 1.3}


def _N_O_H() -> StateSpec:  # N* + O* + H* -- all three still separate
    specs = [
        {"symbol": "N", "site": "fcc", "height": 1.6, "dx": 0.0, "dy": 0.0},
        {"symbol": "O", "site": "hcp", "height": 1.6, "dx": 2.6, "dy": 0.0},
        dict(_H_FREE),
    ]
    return StateSpec("N+O+H", "N.O.H", specs)


def _NH_O() -> StateSpec:  # H bonds N: NH* + O* (fork of N+O+H)
    specs = _fork(_N_O_H().specs, {2: {"site": "fcc", "height": 2.8, "dx": 0.0, "dy": 0.0}})
    return StateSpec("NH+O", "NH.O", specs)


def _N_OH() -> StateSpec:  # H bonds O: N* + OH* (fork of N+O+H)
    specs = _fork(_N_O_H().specs, {
        1: {"height": 1.9},
        2: {"site": "hcp", "height": 2.9, "dx": 2.6, "dy": 0.0},
    })
    return StateSpec("N+OH", "N.OH", specs)


def _NH_O_H() -> StateSpec:  # NH* + O* + H* (+H* supply onto NH+O)
    specs = [*_NH_O().specs, dict(_H_FREE)]
    return StateSpec("NH+O+H", "NH.O.H", specs)


def _NH2_O() -> StateSpec:  # new H also binds N -> NH2* + O* (fork of NH+O+H)
    specs = _fork(_NH_O_H().specs, {
        0: {"height": 1.8},
        2: {"height": 2.8, "dx": 0.9, "dy": 0.5},
        3: {"height": 2.8, "dx": -0.9, "dy": 0.5},
    })
    return StateSpec("NH2+O", "NH2.O", specs)


def _NH_OH_from_NH_O_H() -> StateSpec:  # new H binds O -> NH* + OH* (fork of NH+O+H)
    specs = _fork(_NH_O_H().specs, {
        1: {"height": 1.9},
        3: {"site": "hcp", "height": 2.9, "dx": 2.6, "dy": 0.0},
    })
    return StateSpec("NH+OH", "NH.OH", specs)


def _N_OH_H() -> StateSpec:  # N* + OH* + H* (+H* supply onto N+OH)
    specs = [*_N_OH().specs, dict(_H_FREE)]
    return StateSpec("N+OH+H", "N.OH.H", specs)


def _NH_OH_from_N_OH_H() -> StateSpec:  # new H binds N -> NH* + OH* (fork of N+OH+H)
    specs = _fork(_N_OH_H().specs, {
        0: {"height": 1.7},
        3: {"site": "fcc", "height": 2.8, "dx": 0.0, "dy": 0.0},
    })
    return StateSpec("NH+OH", "NH.OH", specs)


def _N_H2O() -> StateSpec:  # new H completes water -> N* + H2O* (fork of N+OH+H)
    specs = _fork(_N_OH_H().specs, {
        1: {"height": 2.2},
        2: {"height": 2.9, "dx": 3.5, "dy": 0.3},
        3: {"site": "hcp", "height": 2.9, "dx": 1.7, "dy": 0.3},
    })
    return StateSpec("N+H2O", "N.H2O", specs)


def _NH2_O_H() -> StateSpec:  # NH2* + O* + H* (+H* supply onto NH2+O)
    specs = [*_NH2_O().specs, dict(_H_FREE)]
    return StateSpec("NH2+O+H", "NH2.O.H", specs)


def _NH3_O() -> StateSpec:  # new H completes NH3 -> NH3* + O* (fork of NH2+O+H)
    specs = _fork(_NH2_O_H().specs, {
        0: {"height": 2.0},
        2: {"height": 2.9, "dx": 1.0, "dy": 0.0},
        3: {"height": 2.9, "dx": -0.5, "dy": 0.87},
        4: {"site": "fcc", "height": 2.9, "dx": -0.5, "dy": -0.87},
    })
    return StateSpec("NH3+O", "NH3.O", specs)


def _NH2_OH_from_NH2_O_H() -> StateSpec:  # new H binds O -> NH2* + OH* (fork of NH2+O+H)
    specs = _fork(_NH2_O_H().specs, {
        1: {"height": 1.9},
        4: {"site": "hcp", "height": 2.9, "dx": 2.6, "dy": 0.0},
    })
    return StateSpec("NH2+OH", "NH2.OH", specs)


def _NH_OH_H() -> StateSpec:  # NH* + OH* + H* (+H* supply onto NH+OH)
    specs = [*_NH_OH_from_NH_O_H().specs, dict(_H_FREE)]
    return StateSpec("NH+OH+H", "NH.OH.H", specs)


def _NH2_OH_from_NH_OH_H() -> StateSpec:  # new H binds N -> NH2* + OH* (fork of NH+OH+H)
    specs = _fork(_NH_OH_H().specs, {
        0: {"height": 1.8},
        2: {"height": 2.8, "dx": 0.9, "dy": 0.5},
        4: {"site": "fcc", "height": 2.8, "dx": -0.9, "dy": 0.5},
    })
    return StateSpec("NH2+OH", "NH2.OH", specs)


def _NH_H2O() -> StateSpec:  # new H completes water -> NH* + H2O* (fork of NH+OH+H)
    specs = _fork(_NH_OH_H().specs, {
        1: {"height": 2.2},
        3: {"height": 2.9, "dx": 3.5, "dy": 0.3},
        4: {"site": "hcp", "height": 2.9, "dx": 1.7, "dy": 0.3},
    })
    return StateSpec("NH+H2O", "NH.H2O", specs)


def _NH2_OH_H() -> StateSpec:  # NH2* + OH* + H* (+H* supply onto NH2+OH)
    specs = [*_NH2_OH_from_NH2_O_H().specs, dict(_H_FREE)]
    return StateSpec("NH2+OH+H", "NH2.OH.H", specs)


def _NH3_OH() -> StateSpec:  # new H completes NH3 -> NH3* + OH* (fork of NH2+OH+H)
    specs = _fork(_NH2_OH_H().specs, {
        0: {"height": 2.0},
        2: {"height": 2.9, "dx": 1.0, "dy": 0.0},
        3: {"height": 2.9, "dx": -0.5, "dy": 0.87},
        5: {"site": "fcc", "height": 2.9, "dx": -0.5, "dy": -0.87},
    })
    return StateSpec("NH3+OH", "NH3.OH", specs)


def _NH2_H2O() -> StateSpec:  # new H completes water -> NH2* + H2O* (fork of NH2+OH+H)
    specs = _fork(_NH2_OH_H().specs, {
        1: {"height": 2.2},
        4: {"height": 2.9, "dx": 3.5, "dy": 0.3},
        5: {"site": "hcp", "height": 2.9, "dx": 1.7, "dy": 0.3},
    })
    return StateSpec("NH2+H2O", "NH2.H2O", specs)


def _N_bare() -> StateSpec:  # N*, alone -- the landing spot once H2O has desorbed
    return StateSpec("N", "N", [{"symbol": "N", "site": "fcc", "height": 1.6}])


def _O_bare() -> StateSpec:  # O*, alone -- the landing spot once NH3 has desorbed
    return StateSpec("O", "O", [{"symbol": "O", "site": "fcc", "height": 1.5}])


def build_coadsorbed_ammonia_network(slab_cfg: SlabConfig) -> Network:
    """NO reduction to ammonia, ``"coadsorbed"`` (verify) template: the
    fragment-parking approximation is removed -- after dissociation, both N*
    and O* stay in-cell together, each further H* supply forks on WHICH
    fragment takes it, until a product (H2O or NH3) actually desorbs::

        N+O -(+H*)-> N+O+H -c-> NH+O | N+OH
        NH+O -(+H*)-> NH+O+H -c-> NH2+O | NH+OH
        N+OH -(+H*)-> N+OH+H -c-> NH+OH | N+H2O
        NH+OH -(+H*)-> NH+OH+H -c-> NH2+OH | NH+H2O
        NH2+O -(+H*)-> NH2+O+H -c-> NH3+O | NH2+OH
        NH2+OH -(+H*)-> NH2+OH+H -c-> NH3+OH | NH2+H2O

    H2O desorbs as soon as it forms (``N+H2O -> N``, ``NH+H2O -> NH``,
    ``NH2+H2O -> NH2``), landing on the shared single-species N-branch, which
    continues to NH3 exactly as in the parked template; ``NH3+O``/``NH3+OH``
    desorb NH3, landing on ``O``/``OH``, which continue hydrogenating to
    water via the shared single-species O-branch. Every desorption/re-entry
    link is barrier-less bookkeeping (no NEB), same convention as the
    existing ``+O*``/``+H*`` supply links -- see :mod:`catpath.pipeline`
    ``_gas_energy`` for why a genuine desorption ΔE is NOT wired in here (a
    known, reported gap; every such edge stays at the same "no fabricated
    number" 0.0 the rest of the supply-link convention already uses).

    The associative fork and everything before scission are shared byte-for-byte
    with :func:`build_ammonia_network` (see :func:`_ammonia_shared`) -- only the
    dissociative branch's continuation differs.
    """
    shared_steps, shared_links = _ammonia_shared()
    coadsorbed_steps = [
        StepSpec("N+O+H->NH+O", _N_O_H(), _NH_O()),
        StepSpec("N+O+H->N+OH", _N_O_H(), _N_OH()),
        StepSpec("NH+O+H->NH2+O", _NH_O_H(), _NH2_O()),
        StepSpec("NH+O+H->NH+OH", _NH_O_H(), _NH_OH_from_NH_O_H()),
        StepSpec("N+OH+H->NH+OH", _N_OH_H(), _NH_OH_from_N_OH_H()),
        StepSpec("N+OH+H->N+H2O", _N_OH_H(), _N_H2O()),
        StepSpec("NH2+O+H->NH3+O", _NH2_O_H(), _NH3_O()),
        StepSpec("NH2+O+H->NH2+OH", _NH2_O_H(), _NH2_OH_from_NH2_O_H()),
        StepSpec("NH+OH+H->NH2+OH", _NH_OH_H(), _NH2_OH_from_NH_OH_H()),
        StepSpec("NH+OH+H->NH+H2O", _NH_OH_H(), _NH_H2O()),
        StepSpec("NH2+OH+H->NH3+OH", _NH2_OH_H(), _NH3_OH()),
        StepSpec("NH2+OH+H->NH2+H2O", _NH2_OH_H(), _NH2_H2O()),
    ]
    coadsorbed_links = [
        ("N+O", "N+O+H"),        # +H* supply, fork point 1
        ("NH+O", "NH+O+H"),      # +H* supply, fork point 2
        ("N+OH", "N+OH+H"),      # +H* supply, fork point 3
        ("NH2+O", "NH2+O+H"),    # +H* supply, fork point 4
        ("NH+OH", "NH+OH+H"),    # +H* supply, fork point 5
        ("NH2+OH", "NH2+OH+H"),  # +H* supply, fork point 6
        ("N+H2O", "N"),          # H2O desorbs -> lands on the shared N-branch
        ("NH+H2O", "NH"),
        ("NH2+H2O", "NH2"),
        ("NH3+O", "O"),          # NH3 desorbs -> lands on the shared O-branch
        ("NH3+OH", "OH"),
        ("N", "N+H"),            # bare N* rejoins the shared N-branch, +H*
        ("O", "O+H"),            # bare O* rejoins the shared O-branch, +H*
    ]
    return Network(
        slab_cfg,
        steps=[*shared_steps, *coadsorbed_steps],
        links=[*shared_links, *coadsorbed_links],
        extra_states=[_N_bare(), _O_bare()],
    )


def _added_elements(reactant: StateSpec, product: StateSpec) -> set[str]:
    """Elements that INCREASE from reactant to product -> the reagent(s) supplied.

    Uses Counter subtraction, which keeps only positive differences, so an atom
    that leaves to a reservoir (e.g. the O in ``N+O -> N+H``) is not mistaken for
    a reagent -- only the *added* species (here H) count.
    """
    diff = product.adsorbate_counts() - reactant.adsorbate_counts()
    return set(diff)


def filter_by_reagents(net: Network, reagents: list[str]) -> Network:
    """Keep only the part of ``net`` reachable using the allowed reagent adatoms.

    A **supply link** ``a -> b`` requires whatever element it adds (derived from
    the stoichiometry, not hardcoded); it is dropped if that element is not in
    ``reagents``.  **Reaction** steps are atom-conserving (no reagent).  After
    pruning links, any state no longer reachable from the substrate (root) -- and
    every step/link touching it -- is removed.  So ``reagents=[]`` collapses the
    network to the reagent-free steps (e.g. dissociation / site isomers) only.
    """
    allowed = set(reagents)
    states = net.states()

    kept_links = [(a, b) for a, b in net.links
                  if _added_elements(states[a], states[b]) <= allowed]

    adj: dict[str, set[str]] = {}
    for s in net.steps:
        adj.setdefault(s.reactant.name, set()).add(s.product.name)
    for a, b in kept_links:
        adj.setdefault(a, set()).add(b)

    root = net.order()[0]
    reachable = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in reachable:
                reachable.add(v)
                stack.append(v)

    steps = [s for s in net.steps
             if s.reactant.name in reachable and s.product.name in reachable]
    links = [(a, b) for a, b in kept_links if a in reachable and b in reachable]
    extra_states = [st for st in net.extra_states if st.name in reachable]
    return Network(net.slab_cfg, steps=steps, links=links, extra_states=extra_states)


def build_network(slab_cfg: SlabConfig, kind: str = "ammonia",
                  reagents: list[str] | None = None,
                  substrate: str = "NO", target: str | None = None,
                  max_extra: int = 4, max_states: int = 600,
                  template: str = "parked") -> Network:
    """Build a reaction network.

    ``kind="auto"`` autodetects the intermediates from ``substrate`` -> ``target``
    (rule-guided; see :mod:`catpath.explore`), bounded by ``max_extra`` (reagent
    atom budget) and ``max_states``; the curated template kinds ignore
    ``substrate``/``target``/``max_*`` and are filtered by ``reagents`` as before.

    ``template`` selects between the two ``kind="ammonia"`` variants:
    ``"parked"`` (default -- the fragment-parking approximation, unchanged
    behavior) or ``"coadsorbed"`` (the verify tier that removes it -- see
    :func:`build_coadsorbed_ammonia_network`). Meaningless for any other
    ``kind`` -- ``"coadsorbed"`` there is a ``ValueError``.
    """
    if template not in ("parked", "coadsorbed"):
        raise ValueError(f"unknown template {template!r}; choose 'parked' or 'coadsorbed'")
    if kind == "auto":
        if template != "parked":
            raise ValueError("template='coadsorbed' only applies to kind='ammonia'")
        from .explore import build_auto_network
        return build_auto_network(slab_cfg, substrate=substrate,
                                  target=target or substrate, reagents=reagents,
                                  max_extra=max_extra, max_states=max_states)
    if template == "coadsorbed" and kind != "ammonia":
        raise ValueError("template='coadsorbed' only applies to kind='ammonia'")
    builders = {
        "oxidation": build_oxidation_network,
        "branching": build_branching_network,
        "ammonia": build_coadsorbed_ammonia_network if template == "coadsorbed"
                   else build_ammonia_network,
    }
    if kind not in builders:
        raise ValueError(
            f"unknown network {kind!r}; choose one of "
            f"{', '.join([*builders, 'auto'])} (auto = autodetect intermediates)")
    net = builders[kind](slab_cfg)
    if reagents is not None:  # None = full template; a list (even []) filters
        net = filter_by_reagents(net, reagents)
    return net
