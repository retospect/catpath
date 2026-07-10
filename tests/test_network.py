from atosim.config import SlabConfig
from atosim.network import (
    build_ammonia_network,
    build_branching_network,
    build_network,
    build_oxidation_network,
)


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
