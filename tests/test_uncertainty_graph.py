import json

from atosim.graph import build_graph, to_csv, to_json
from atosim.uncertainty import Estimate, aggregate, rankings_consistent


def test_aggregate_mean_std():
    e = aggregate([1.0, 1.0, 1.0], spread_tol=0.05)
    assert e.mean == 1.0 and e.std == 0.0
    assert e.n == 3
    assert not e.low_confidence


def test_aggregate_flags_high_spread():
    e = aggregate([0.0, 1.0, 2.0], spread_tol=0.05)
    assert e.low_confidence
    assert e.n == 3


def test_aggregate_single_value_low_confidence():
    e = aggregate([0.5], spread_tol=0.05)
    assert e.low_confidence  # n<2


def test_aggregate_empty():
    e = aggregate([], spread_tol=0.05)
    assert e.n == 0 and e.low_confidence


def test_rankings_consistent():
    assert rankings_consistent([["a", "b"], ["a", "b"]])
    assert not rankings_consistent([["a", "b"], ["b", "a"]])


def test_build_graph_and_serialize(tmp_path):
    nodes = {
        "A": Estimate(0.0, 0.0, 3, [0, 0, 0]),
        "B": Estimate(-0.5, 0.01, 3, [-0.5, -0.5, -0.49]),
    }
    edges = [{
        "name": "A->B", "reactant": "A", "product": "B",
        "barrier": Estimate(0.8, 0.02, 3, []),
        "delta_e": Estimate(-0.5, 0.01, 3, []),
    }]
    g = build_graph(nodes, edges, energy_ref=0.0)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert g["A"]["B"]["barrier"] == 0.8

    jp = tmp_path / "g.json"
    to_json(g, jp)
    data = json.loads(jp.read_text())
    assert "nodes" in data

    to_csv(g, tmp_path / "n.csv", tmp_path / "e.csv")
    assert (tmp_path / "n.csv").exists()
    assert "barrier_eV" in (tmp_path / "e.csv").read_text()
