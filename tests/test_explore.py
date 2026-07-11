"""Rule-guided intermediate autodetection (``network: auto``).

We check the abstract explorer (rules, valence, acyclicity, target-derived
reagents, prune-to-target) and the materialisation (states build without atom
overlaps; every step's endpoints share an element ordering so NEB can
interpolate).  No calculator is needed -- this is all graph + geometry.
"""

import networkx as nx
import numpy as np

from atosim import explore
from atosim.config import SlabConfig
from atosim.network import build_network
from atosim.structures import build_slab

_SLAB = SlabConfig(size=(2, 2, 3), vacuum=8.0)


# --- parsing / reagent derivation -------------------------------------------

def test_parse_molecule_counts_and_connectivity():
    g = explore.parse_molecule("NH3")
    assert sorted(g.nodes[n]["el"] for n in g) == ["H", "H", "H", "N"]
    # star: the N is bonded to all three H's
    n = next(i for i in g if g.nodes[i]["el"] == "N")
    assert g.degree(n) == 3 and g.number_of_edges() == 3


def test_default_reagents_are_target_minus_substrate():
    r = explore._default_reagents(explore.parse_molecule("NO"),
                                  explore.parse_molecule("NH3"))
    assert r == {"H"}                     # reduction needs only H
    r2 = explore._default_reagents(explore.parse_molecule("NO"),
                                   explore.parse_molecule("NO3"))
    assert r2 == {"O"}                    # oxidation needs only O


# --- exploration rules ------------------------------------------------------

def test_explore_reaches_target_and_key_intermediates():
    nodes, edges, root, goals = explore.explore("NO", "NH3")
    names = {explore._state_name(nodes[k].g) for k in nodes}
    assert goals                                   # target is reachable
    for expected in ("N+O", "HNO", "NOH", "NH+O"):  # dissociation + associative
        assert expected in names


def test_supply_links_change_composition_by_one_reagent_atom():
    nodes, edges, root, goals = explore.explore("NO", "NH3")
    for a, b, kind, meta in edges:
        na = sorted(nodes[a].g.nodes[n]["el"] for n in nodes[a].g)
        nb = sorted(nodes[b].g.nodes[n]["el"] for n in nodes[b].g)
        if kind == "link":                         # supply: exactly +1 atom
            assert len(nb) == len(na) + 1
        else:                                      # step: atom-conserving
            assert nb == na


def test_valence_limits_prevent_overfilling():
    # NH3 is valence-saturated at N: no further H can bond.
    g = explore.parse_molecule("NH3")
    n = next(i for i in g if g.nodes[i]["el"] == "N")
    assert explore._free_valence(g, n) == 0


def test_network_is_acyclic():
    net = build_network(_SLAB, "auto", substrate="NO", target="NH3")
    g = nx.DiGraph()
    g.add_nodes_from(net.states())
    for s in net.steps:
        g.add_edge(s.reactant.name, s.product.name)
    for a, b in net.links:
        g.add_edge(a, b)
    assert nx.is_directed_acyclic_graph(g)


def test_prune_drops_dead_branches():
    nodes, edges, root, goals = explore.explore("NO", "NH3")
    keep, kept_edges = explore._prune_to_target(nodes, edges, root, goals)
    assert root in keep and keep <= set(nodes)
    assert len(keep) <= len(nodes)
    # every kept node can still reach a goal
    g = nx.DiGraph([(a, b) for a, b, *_ in kept_edges])
    g.add_nodes_from(keep)
    for k in keep:
        assert k in goals or any(gk in nx.descendants(g, k) for gk in goals)


# --- materialisation --------------------------------------------------------

def test_materialised_states_build_without_overlap():
    net = build_network(SlabConfig(size=(3, 3, 3), vacuum=8.0),
                        "auto", substrate="NO", target="NH3")
    slab = build_slab(net.slab_cfg)
    n_slab = len(slab)
    for st in net.states().values():
        ads = st.build(slab).get_positions()[n_slab:]
        if len(ads) > 1:
            d = np.linalg.norm(ads[:, None, :] - ads[None, :, :], axis=-1)
            assert d[~np.eye(len(ads), dtype=bool)].min() > 0.8


def test_every_step_has_neb_compatible_endpoints():
    """Reactant and product must share an identical element sequence."""
    net = build_network(SlabConfig(size=(3, 3, 3), vacuum=8.0),
                        "auto", substrate="NO", target="NH3")
    slab = build_slab(net.slab_cfg)
    n_slab = len(slab)
    for s in net.steps:
        r = s.reactant.build(slab).get_chemical_symbols()[n_slab:]
        p = s.product.build(slab).get_chemical_symbols()[n_slab:]
        assert r == p, f"{s.name}: {r} != {p}"


def test_oxidation_target_uses_oxygen_only():
    net = build_network(_SLAB, "auto", substrate="NO", target="NO3")
    # with O the only reagent, no N-H states should appear
    assert any("NO3" in n for n in net.states())
    assert not any("H" in n for n in net.states())


def test_build_network_auto_integrates():
    net = build_network(_SLAB, "auto", substrate="NO", target="NH3")
    assert net.steps and net.order()[0] == "NO"     # root sorts first
    assert any("NH3" in n for n in net.states())     # target present
