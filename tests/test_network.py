from atosim.config import SlabConfig
from atosim.network import build_network


def test_network_steps():
    net = build_network(SlabConfig(size=(2, 2, 3)))
    assert len(net.steps) == 2
    names = [s.name for s in net.steps]
    assert names == ["NO+O->NO2", "NO2+O->NO3"]


def test_step_endpoints_atom_conserving():
    """NEB requires reactant and product to have identical atom counts."""
    net = build_network(SlabConfig(size=(2, 2, 3)))
    slab = net.slab()
    for step in net.steps:
        r = step.reactant.build(slab)
        p = step.product.build(slab)
        assert len(r) == len(p), step.name
        # same multiset of elements
        assert sorted(r.get_chemical_symbols()) == sorted(p.get_chemical_symbols())


def test_states_unique():
    net = build_network(SlabConfig(size=(2, 2, 3)))
    states = net.states()
    assert set(states) == {"NO+O", "NO2", "NO2+O", "NO3"}
