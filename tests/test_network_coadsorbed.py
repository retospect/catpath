"""The ``coadsorbed`` ammonia template: the verify tier that removes the
fragment-parking approximation -- after dissociation (``NO -> N+O``), both N*
and O* stay in-cell together until a product (H2O or NH3) actually desorbs.
See :func:`catpath.network.build_coadsorbed_ammonia_network`.
"""

from __future__ import annotations

import math

import pytest

from catpath import electrochem as ec
from catpath.config import Config, SlabConfig
from catpath.network import (
    build_ammonia_network,
    build_coadsorbed_ammonia_network,
    build_network,
)

_SLAB = SlabConfig(size=(2, 2, 3))

# The 6 "chemical fork" reactants -- one per hydrogenation step of the
# dissociative branch -- and the 2 products each forks to.
_FORK_REACTANTS = {
    "N+O+H": {"NH+O", "N+OH"},
    "NH+O+H": {"NH2+O", "NH+OH"},
    "N+OH+H": {"NH+OH", "N+H2O"},
    "NH2+O+H": {"NH3+O", "NH2+OH"},
    "NH+OH+H": {"NH2+OH", "NH+H2O"},
    "NH2+OH+H": {"NH3+OH", "NH2+H2O"},
}

_DESORPTION_LINKS = {
    ("N+H2O", "N"), ("NH+H2O", "NH"), ("NH2+H2O", "NH2"),
    ("NH3+O", "O"), ("NH3+OH", "OH"),
}

_SHARED_STATES = {
    "NO", "N+O", "NO@top", "N+H", "NH", "NH+H", "NH2", "NH2+H", "NH3",
    "NO+H", "HNO", "NOH", "O+H", "OH", "OH+H", "H2O",
}


# --- topology ----------------------------------------------------------------

def test_coadsorbed_template_state_edge_counts():
    net = build_coadsorbed_ammonia_network(_SLAB)
    # 16 shared + 16 new coadsorbed + 2 extra (bare N/O landing spots)
    assert len(net.states()) == 34
    assert len(net.steps) == 21   # 9 shared + 12 chemical-fork steps
    assert len(net.links) == 17   # 4 shared + 13 new (6 supply + 5 desorb + 2 re-entry)
    assert {s.name for s in net.extra_states} == {"N", "O"}


def test_coadsorbed_shares_prefix_with_parked():
    """The associative fork and everything before scission are literally the
    same declarations (see network._ammonia_shared), not a re-typed copy."""
    parked = build_ammonia_network(_SLAB)
    coad = build_coadsorbed_ammonia_network(_SLAB)

    parked_step_names = {s.name for s in parked.steps}
    coad_step_names = {s.name for s in coad.steps}
    shared_step_names = {
        "NO->N+O", "NO->NO@top", "N+H->NH", "NH+H->NH2", "NH2+H->NH3",
        "NO+H->HNO", "NO+H->NOH", "O+H->OH", "OH+H->H2O",
    }
    assert shared_step_names <= parked_step_names
    assert shared_step_names <= coad_step_names

    shared_links = {("NO", "NO+H"), ("NH", "NH+H"), ("NH2", "NH2+H"), ("OH", "OH+H")}
    assert shared_links <= set(parked.links)
    assert shared_links <= set(coad.links)

    # the parked-only parking links must NOT be present in coadsorbed
    parking = {("N+O", "N+H"), ("N+O", "O+H")}
    assert parking <= set(parked.links)
    assert parking.isdisjoint(set(coad.links))

    # every shared single-species state's own StateSpec is the identical
    # geometry in both templates (same function, not re-typed coordinates)
    p_states, c_states = parked.states(), coad.states()
    for name in _SHARED_STATES:
        assert p_states[name].specs == c_states[name].specs, name


def test_coadsorbed_fork_points_present():
    net = build_coadsorbed_ammonia_network(_SLAB)
    products_by_reactant: dict[str, set[str]] = {}
    for s in net.steps:
        products_by_reactant.setdefault(s.reactant.name, set()).add(s.product.name)
    for reactant, products in _FORK_REACTANTS.items():
        assert products_by_reactant.get(reactant) == products, reactant
        # each fork reactant is genuinely the reactant of TWO steps
        assert [s.reactant.name for s in net.steps].count(reactant) == 2


def test_desorption_edges_are_links_not_steps():
    net = build_coadsorbed_ammonia_network(_SLAB)
    step_pairs = {(s.reactant.name, s.product.name) for s in net.steps}
    for a, b in _DESORPTION_LINKS:
        assert (a, b) in net.links
        assert (a, b) not in step_pairs  # no NEB for a desorption edge


def test_bare_landing_states_rejoin_shared_chain():
    """The bare N*/O* extra states reconnect into the shared single-species
    chain via a plain +H* supply link, same convention as everywhere else."""
    net = build_coadsorbed_ammonia_network(_SLAB)
    assert ("N", "N+H") in net.links
    assert ("O", "O+H") in net.links


def test_parked_template_unchanged():
    """Regression: the parked builder/behavior is untouched by the refactor."""
    net = build_ammonia_network(_SLAB)
    assert {"NO", "N+O", "N+H", "NH", "NH+H", "NH2", "NH2+H", "NH3"} <= set(net.states())
    assert ("N+O", "N+H") in net.links
    assert ("N+O", "O+H") in net.links
    assert net.extra_states == []
    assert len(net.steps) == 9
    assert len(net.links) == 6


# --- build_network dispatch ---------------------------------------------------

def test_build_network_template_dispatch():
    parked_default = build_network(_SLAB, "ammonia")
    parked_explicit = build_network(_SLAB, "ammonia", template="parked")
    coad = build_network(_SLAB, "ammonia", template="coadsorbed")
    assert set(parked_default.states()) == set(parked_explicit.states())
    assert set(coad.states()) != set(parked_default.states())
    assert len(coad.states()) == 34


def test_build_network_coadsorbed_rejects_non_ammonia_kind():
    with pytest.raises(ValueError):
        build_network(_SLAB, "branching", template="coadsorbed")
    with pytest.raises(ValueError):
        build_network(_SLAB, "oxidation", template="coadsorbed")


def test_build_network_rejects_unknown_template():
    with pytest.raises(ValueError):
        build_network(_SLAB, "ammonia", template="floating")


# --- structure builder: >=3 fragments (NEB order + gross sanity) ------------

def test_all_coadsorbed_steps_atom_conserving_and_order_matched():
    """NEB requires reactant/product to carry identical atoms IN ORDER (not
    just as a multiset) -- the invariant every 'chemical fork' relies on."""
    net = build_coadsorbed_ammonia_network(_SLAB)
    slab = net.slab()
    for step in net.steps:
        r, p = step.reactant.build(slab), step.product.build(slab)
        assert len(r) == len(p), step.name
        assert r.get_chemical_symbols() == p.get_chemical_symbols(), step.name


def test_three_and_more_fragment_states_build():
    """Sanity: the '+'-split fragment placement (place_fragments) handles the
    coadsorbed template's >=3-fragment states, not just the 2-fragment case
    every other template used."""
    net = build_coadsorbed_ammonia_network(_SLAB)
    slab = net.slab()
    n_slab = slab.info["n_slab"]
    multi_fragment = {
        "N+O+H": 3, "NH+O+H": 4, "N+OH+H": 4, "NH2+O+H": 5,
        "NH+OH+H": 5, "NH2+OH+H": 6, "NH3+OH": 6, "NH2+H2O": 6,
    }
    states = net.states()
    for name, n_atoms in multi_fragment.items():
        st = states[name]
        atoms = st.build(slab)
        assert len(atoms) - n_slab == n_atoms == len(st.specs), name


def test_every_state_builds_without_exception():
    net = build_coadsorbed_ammonia_network(_SLAB)
    slab = net.slab()
    n_slab = slab.info["n_slab"]
    for name, st in net.states().items():
        atoms = st.build(slab)
        assert len(atoms) - n_slab == len(st.specs), name


# --- electrochemistry: required_leaves comes out {target} automatically -----

def test_required_leaves_no_parking_links_is_target_only():
    """With the parking approximation removed, every root fragment's journey
    to a sink is modeled explicitly (no bookkeeping-only divergence off the
    NO3->NH3 path) -- required_leaves should need NO special-casing to land
    on just {target}."""
    net = build_network(_SLAB, kind="ammonia", template="coadsorbed")
    root = net.order()[0]
    assert root == "NO"
    all_edges = [(s.reactant.name, s.product.name) for s in net.steps] + list(net.links)

    path = ec.reaction_path(all_edges, root, "NH3")
    assert path and path[-1] == "NH3"

    leaves = ec.required_leaves(all_edges, net.links, root, "NH3")
    assert leaves == {"NH3"}


# --- full pipeline: real EMT on a tiny slab (mirrors test_screening.py) -----

def _tiny_coadsorbed_cfg(tmp_path):
    cfg = Config(name="coadtest", outdir=str(tmp_path))
    cfg.network = "ammonia"
    cfg.template = "coadsorbed"
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1, relax_lattice=False)
    cfg.search.seeds = [0]
    cfg.search.max_steps = 20
    cfg.search.screening = True   # relax-only -- fast, no NEB
    cfg.search.bind_preflight = False
    return cfg


def test_pipeline_coadsorbed_end_to_end_and_template_stamped(tmp_path):
    import json

    from catpath import pipeline

    cfg = _tiny_coadsorbed_cfg(tmp_path)
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    outdir = pipeline.write_outputs(cfg, res, log=lambda *a, **k: None)

    summary = json.loads((outdir / "results.json").read_text())
    assert summary["template"] == "coadsorbed"
    assert len(summary["nodes"]) == 34
    # extra states (bare N/O) got a real energy, not a NaN placeholder
    assert math.isfinite(summary["nodes"]["N"]["mean"])
    assert math.isfinite(summary["nodes"]["O"]["mean"])

    graph = json.loads((outdir / "graph.json").read_text())
    assert len(graph["nodes"]) == 34


def test_pipeline_parked_default_template_stamped(tmp_path):
    """Regression: the default (parked) template is unchanged and still
    stamps itself into results.json."""
    import json

    from catpath import pipeline

    cfg = _tiny_coadsorbed_cfg(tmp_path)
    cfg.template = "parked"
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    outdir = pipeline.write_outputs(cfg, res, log=lambda *a, **k: None)

    summary = json.loads((outdir / "results.json").read_text())
    assert summary["template"] == "parked"
    assert len(summary["nodes"]) == 16


# --- precis bridge: the pure runner also stamps "template" -------------------

COADSORBED_YAML = """
name: coadsorbed_bridge_smoke
substrate: "NO"
target: "NH3"
network: ammonia
template: coadsorbed
slab: {element: Pd, size: [2, 2, 3], vacuum: 8.0, fix_layers: 1, relax_lattice: false}
mlip: {backend: emt}
search: {seeds: [0], screening: true, max_steps: 20, pose_count: 2}
"""


def test_runner_stamps_template_coadsorbed():
    from catpath.precis import runner

    art = runner.run_pathway_from_yaml(COADSORBED_YAML)
    r = art["results_json"]
    assert r["template"] == "coadsorbed"
    assert len(r["nodes"]) == 34
    assert art["config"]["template"] == "coadsorbed"

