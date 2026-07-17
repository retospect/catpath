from catpath.config import SlabConfig
from catpath.network import (
    Network,
    build_ammonia_network,
    build_branching_network,
    build_network,
    build_oxidation_network,
    filter_by_reagents,
)
from catpath.structures import build_slab, place_fragments

_SLAB = SlabConfig(size=(2, 2, 3))


def test_oxidation_network_is_linear_chain():
    net = build_oxidation_network(SlabConfig(size=(2, 2, 3)))
    assert [s.name for s in net.steps] == ["NO+O->NO2", "NO2+O->NO3"]
    assert set(net.states()) == {"NO+O", "NO2", "NO2+O", "NO3"}


def test_branching_network_has_fork_and_links():
    net = build_branching_network(SlabConfig(size=(2, 2, 3)))
    # three pathways rooted at NO -> more than the linear chain
    assert set(net.states()) == {
        "NO", "N+O", "NO+O", "NO2", "NO2+O", "NO3", "NO+H", "HNO", "NOH"
    }
    # the reduction fork: NO+H is the reactant of two different steps
    reactants = [s.reactant.name for s in net.steps]
    assert reactants.count("NO+H") == 2
    assert ("NO", "NO+H") in net.links


def test_branching_order_starts_at_root():
    net = build_branching_network(SlabConfig(size=(2, 2, 3)))
    order = net.order()
    assert order[0] == "NO"  # topological root (the substrate)
    assert set(order) == set(net.states())


def test_ammonia_network_reaches_nh3():
    net = build_ammonia_network(SlabConfig(size=(2, 2, 3)))
    states = set(net.states())
    # dissociative hydrogenation chain to ammonia
    assert {"NO", "N+O", "N+H", "NH", "NH+H", "NH2", "NH2+H", "NH3"} <= states
    # NH3 is a leaf (terminal product)
    order = net.order()
    assert order[0] == "NO"
    assert "NH3" in order
    # reduction fork present
    assert [s.reactant.name for s in net.steps].count("NO+H") == 2


def test_all_steps_atom_conserving():
    """NEB requires reactant/product to have identical atoms (all networks)."""
    for builder in (build_oxidation_network, build_branching_network,
                    build_ammonia_network):
        net = builder(SlabConfig(size=(2, 2, 3)))
        slab = net.slab()
        for step in net.steps:
            r, p = step.reactant.build(slab), step.product.build(slab)
            assert len(r) == len(p), step.name
            assert sorted(r.get_chemical_symbols()) == sorted(p.get_chemical_symbols())


def test_build_network_dispatch():
    assert len(build_network(SlabConfig(size=(2, 2, 3)), "oxidation").steps) == 2
    assert len(build_network(SlabConfig(size=(2, 2, 3)), "branching").steps) == 5


# --- reagent filtering -------------------------------------------------------

def test_reagents_none_is_full_template():
    """reagents=None (default) leaves the curated network untouched."""
    full = build_ammonia_network(_SLAB)
    same = build_network(_SLAB, "ammonia", reagents=None)
    assert set(same.states()) == set(full.states())
    assert len(same.steps) == len(full.steps)


def test_ammonia_full_needs_only_hydrogen():
    """Ammonia chemistry uses only H* -> reagents=['H'] keeps the whole network."""
    full = build_ammonia_network(_SLAB)
    h_only = filter_by_reagents(full, ["H"])
    assert set(h_only.states()) == set(full.states())
    assert len(h_only.links) == len(full.links)


def test_no_reagents_collapses_to_dissociation():
    """reagents=[] drops every +adatom supply link -> only reagent-free steps."""
    net = build_network(_SLAB, "ammonia", reagents=[])
    assert set(net.states()) == {"NO", "N+O", "NO@top"}
    assert {s.name for s in net.steps} == {"NO->N+O", "NO->NO@top"}
    assert net.links == []
    assert net.order()[0] == "NO"


def test_oxygen_only_keeps_oxidation_drops_reduction():
    net = build_network(_SLAB, "branching", reagents=["O"])
    states = set(net.states())
    assert {"NO", "N+O", "NO+O", "NO2", "NO2+O", "NO3"} == states
    assert "HNO" not in states and "NOH" not in states and "NO+H" not in states


def test_hydrogen_only_keeps_reduction_drops_oxidation():
    net = build_network(_SLAB, "branching", reagents=["H"])
    states = set(net.states())
    assert {"NO", "N+O", "NO+H", "HNO", "NOH"} == states
    assert "NO2" not in states and "NO3" not in states


def test_filtered_steps_still_atom_conserving():
    net = build_network(_SLAB, "ammonia", reagents=["H"])
    slab = net.slab()
    for step in net.steps:
        r, p = step.reactant.build(slab), step.product.build(slab)
        assert sorted(r.get_chemical_symbols()) == sorted(p.get_chemical_symbols())


# --- injected-slab seam (the precis `structure` input; Slice 2) --------------


def test_injected_slab_is_scored_not_rebuilt():
    """`net.slab()` returns the supplied slab (a copy), not an fcc111 rebuild."""
    prepared = build_slab(SlabConfig(element="Pd", size=(2, 2, 3)))
    prepared.info["marker"] = "precis-owned"
    net = Network(SlabConfig(element="Pd", size=(2, 2, 3)), prebuilt_slab=prepared)
    got = net.slab()
    assert got.info.get("marker") == "precis-owned"  # it's the injected slab
    assert len(got) == len(prepared)
    assert got.get_chemical_formula() == prepared.get_chemical_formula()
    assert got.info["n_slab"] == len(prepared)
    # a copy — mutating the result must not touch the caller's Atoms
    got.info["marker"] = "mutated"
    assert prepared.info["marker"] == "precis-owned"


def test_injected_slab_missing_adsorbate_info_still_places_fragments():
    """A prepared slab that lost `adsorbate_info` (e.g. an extxyz round-trip)
    gets it transplanted from the cfg reference, so named-site placement works."""
    cfg = SlabConfig(element="Pd", size=(2, 2, 3))
    prepared = build_slab(cfg)
    prepared.info.pop("adsorbate_info", None)  # simulate the round-trip loss
    net = Network(cfg, prebuilt_slab=prepared)
    slab = net.slab()
    assert "adsorbate_info" in slab.info  # transplanted
    # placement at a named high-symmetry site now resolves + appends the atom
    placed = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    assert len(placed) == len(prepared) + 1
    assert "O" in placed.get_chemical_symbols()


def test_no_injection_builds_from_label_as_before():
    net = Network(SlabConfig(element="Pd", size=(2, 2, 3)))
    slab = net.slab()
    assert "adsorbate_info" in slab.info
    assert set(slab.get_chemical_symbols()) == {"Pd"}
