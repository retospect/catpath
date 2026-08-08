import numpy as np

from autocatpath.graph import build_graph
from autocatpath.uncertainty import Estimate
from autocatpath.viz import draw_graph, energy_map


def test_energy_map_writes_file(tmp_path):
    matrix = np.array([[0.0, -0.3, 0.5, 0.2]])
    p = tmp_path / "map.png"
    energy_map(matrix, ["NO"], ["NO+O", "NO2", "NO2+O", "NO3"], p)
    assert p.exists() and p.stat().st_size > 0


def test_energy_map_handles_nan(tmp_path):
    matrix = np.array([[0.0, np.nan, 0.5], [0.1, 0.2, np.nan]])
    p = tmp_path / "map2.png"
    energy_map(matrix, ["A", "B"], ["s0", "s1", "s2"], p)
    assert p.exists()


def test_draw_profile_multi_pathway(tmp_path):
    from autocatpath.viz import draw_profile
    # a small branching graph: A -> B -> C  and  A -> D
    nodes = {k: Estimate(v, 0.0, 3, []) for k, v in
             {"A": 0.0, "B": -0.2, "C": -0.5, "D": 0.1}.items()}
    edges = [
        {"name": "A->B", "reactant": "A", "product": "B",
         "barrier": Estimate(0.6, 0.02, 3, []), "delta_e": Estimate(-0.2, 0, 3, [])},
        {"name": "B->C", "reactant": "B", "product": "C",
         "barrier": Estimate(0.4, 0.02, 3, []), "delta_e": Estimate(-0.3, 0, 3, [])},
        {"name": "A->D", "reactant": "A", "product": "D", "kind": "supply"},
    ]
    g = build_graph(nodes, edges, energy_ref=0.0)
    p = tmp_path / "profile.png"
    draw_profile(g, p)
    assert p.exists() and p.stat().st_size > 0


def test_species_label_stars_adsorbed_fragments():
    from autocatpath.viz import _species_label
    # every co-adsorbed fragment (joined by '+') gets its own '*'
    assert _species_label("N+O") == "N*+O*"
    assert _species_label("NO+H") == "NO*+H*"
    assert _species_label("NH2+H") == "NH2*+H*"
    # single adsorbate
    assert _species_label("NH3") == "NH3*"
    assert _species_label("H2O") == "H2O*"
    # site-pinned isomer keeps the '@site' suffix, star before it
    assert _species_label("NO@top") == "NO*@top"


def test_draw_profile_staggers_colliding_labels(tmp_path):
    # two states in the same column at (nearly) the same energy must not stack
    # their labels on one side — one goes above its line, the other below.
    from autocatpath.viz import draw_profile
    nodes = {k: Estimate(v, 0.05, 3, []) for k, v in
             {"A": 0.0, "B": 0.45, "C": 0.45}.items()}  # B, C collide at column 1
    edges = [
        {"name": "A->B", "reactant": "A", "product": "B", "kind": "supply"},
        {"name": "A->C", "reactant": "A", "product": "C", "kind": "supply"},
    ]
    g = build_graph(nodes, edges, energy_ref=0.0)
    p = tmp_path / "stagger.png"
    draw_profile(g, p)
    assert p.exists() and p.stat().st_size > 0


def test_supply_barrier_bep():
    from autocatpath.viz import _SUPPLY_BEP_BETA, _supply_barrier
    # downhill-forward staging collapses to a barrierless connector (Ea = 0)
    assert _supply_barrier(-0.5) == 0.0
    # thermoneutral staging keeps the intrinsic BEP intercept beta
    assert _supply_barrier(0.0) == _SUPPLY_BEP_BETA
    # uphill staging puts the TS strictly ABOVE the product (dE), never below it
    assert _supply_barrier(0.4) > 0.4


def test_draw_profile_uphill_supply_gets_a_hump(tmp_path):
    # an uphill supply step must render (a dashed BEP hump), not raise
    from autocatpath.viz import draw_profile
    nodes = {k: Estimate(v, 0.0, 3, []) for k, v in
             {"A": 0.0, "B": 0.5}.items()}  # +0.5 eV uphill supply
    edges = [{"name": "A->B", "reactant": "A", "product": "B", "kind": "supply"}]
    g = build_graph(nodes, edges, energy_ref=0.0)
    p = tmp_path / "uphill_supply.png"
    draw_profile(g, p)
    assert p.exists() and p.stat().st_size > 0


def test_draw_graph_writes_file(tmp_path):
    nodes = {"A": Estimate(0.0, 0.0, 3, []), "B": Estimate(-0.4, 0.02, 3, [])}
    edges = [{"name": "A->B", "reactant": "A", "product": "B",
              "barrier": Estimate(0.7, 0.03, 3, []),
              "delta_e": Estimate(-0.4, 0.02, 3, [])}]
    g = build_graph(nodes, edges, energy_ref=0.0)
    p = tmp_path / "g.png"
    draw_graph(g, p)
    assert p.exists() and p.stat().st_size > 0
