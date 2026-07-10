import numpy as np

from atosim.graph import build_graph
from atosim.uncertainty import Estimate
from atosim.viz import draw_graph, energy_map


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
    from atosim.viz import draw_profile
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


def test_draw_graph_writes_file(tmp_path):
    nodes = {"A": Estimate(0.0, 0.0, 3, []), "B": Estimate(-0.4, 0.02, 3, [])}
    edges = [{"name": "A->B", "reactant": "A", "product": "B",
              "barrier": Estimate(0.7, 0.03, 3, []),
              "delta_e": Estimate(-0.4, 0.02, 3, [])}]
    g = build_graph(nodes, edges, energy_ref=0.0)
    p = tmp_path / "g.png"
    draw_graph(g, p)
    assert p.exists() and p.stat().st_size > 0
