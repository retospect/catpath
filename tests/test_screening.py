"""SCREENING mode (``SearchConfig.screening``): a relax-only, cheap
thermodynamic tier -- endpoint relaxations (bind preflight included) run
exactly as usual, but NEB never runs and no edge ever carries a ``barrier``
key. CHE post-processing (U_L/U_opt/span) still runs and is honestly
thermodynamic-only; P_side falls back to "insufficient data" (None) since no
competing barrier is ever computed.

The ``run_one_seed`` NEB-gating unit tests reuse ``test_bind_preflight``'s
monkeypatch style (fast, no real relax); the pipeline/bridge tests run real
EMT on a tiny slab (also fast).
"""

from __future__ import annotations

import math

import pytest

from autocatpath import pipeline
from autocatpath.config import Config, ElectrochemistryConfig, SlabConfig
from autocatpath.network import Network, StateSpec, StepSpec
from autocatpath.relax import RelaxResult
from autocatpath.validate import GeometryReport


def tiny_cfg(tmp_path, screening=True):
    cfg = Config(name="screentest", outdir=str(tmp_path))
    cfg.network = "oxidation"  # fast linear chain, same as test_pipeline.py
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 20
    cfg.search.screening = screening
    return cfg


def one_step_network(slab_cfg):
    reactant = StateSpec("R", "R", [{"symbol": "O", "site": "fcc", "height": 2.0}])
    product = StateSpec("P", "P", [{"symbol": "O", "site": "hcp", "height": 2.0, "dx": 1.0}])
    return Network(slab_cfg=slab_cfg, steps=[StepSpec("R->P", reactant, product)])


def _dummy_result(atoms, energy):
    return RelaxResult(atoms=atoms, energy=energy, fmax=0.01, steps=1, converged=True)


# --- run_one_seed: NEB is never invoked in screening mode -------------------

def test_run_one_seed_screening_skips_neb_no_barrier_key(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0, reasons=[])

    def fake_bound(state, slab_, n_slab_, cfg_, seed):
        atoms = state.build(slab_)
        energy = 0.0 if state.name == "R" else -0.4
        return _dummy_result(atoms, energy), ok_geo, True, None

    monkeypatch.setattr(pipeline, "_relax_state_bound", fake_bound)

    def fake_neb(*a, **k):
        raise AssertionError("NEB must never run in screening mode")

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert "barrier" not in entry  # absent, not None-valued -- see SearchConfig.screening
    assert entry["delta_e"] == pytest.approx(-0.4)  # free: product E - reactant E


def test_run_one_seed_screening_unbound_endpoint_no_barrier_key(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0, reasons=[])
    bad_geo = GeometryReport(ok=False, min_dist=1.0, adsorbate_height=9.0,
                             reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    def fake_bound(state, slab_, n_slab_, cfg_, seed):
        atoms = state.build(slab_)
        if state.name == "P":  # product never binds -- same bind-preflight gate as usual
            return _dummy_result(atoms, 0.0), bad_geo, False, None
        return _dummy_result(atoms, 0.0), ok_geo, True, None

    monkeypatch.setattr(pipeline, "_relax_state_bound", fake_bound)

    def fake_neb(*a, **k):
        raise AssertionError("NEB must never run in screening mode")

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert "barrier" not in entry
    assert entry["delta_e"] is None
    assert any("INFEASIBLE" in w and "P " in w for w in part["warnings"]), part["warnings"]


def test_screening_off_reproduces_default_run_one_seed_shape(tmp_path, monkeypatch):
    """Regression: ``screening`` defaults to False -- an unbound endpoint still
    leaves the pre-screening ``barrier: None`` shape (byte-identical)."""
    cfg = tiny_cfg(tmp_path, screening=False)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0, reasons=[])
    bad_geo = GeometryReport(ok=False, min_dist=1.0, adsorbate_height=9.0,
                             reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    def fake_bound(state, slab_, n_slab_, cfg_, seed):
        atoms = state.build(slab_)
        if state.name == "P":
            return _dummy_result(atoms, 0.0), bad_geo, False, None
        return _dummy_result(atoms, 0.0), ok_geo, True, None

    monkeypatch.setattr(pipeline, "_relax_state_bound", fake_bound)
    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert entry["barrier"] is None
    assert entry["delta_e"] is None


# --- full pipeline: real EMT on a tiny slab ---------------------------------

def test_pipeline_screening_end_to_end(tmp_path):
    cfg = tiny_cfg(tmp_path, screening=True)
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    assert len(res.edges) == 2
    for e in res.edges:
        assert "barrier" not in e
        assert "delta_e" in e

    outdir = pipeline.write_outputs(cfg, res, log=lambda *a, **k: None)
    for f in ["results.json", "graph.json", "graph.png", "energy_map.png",
              "nodes.csv", "edges.csv", "config.snapshot.yaml"]:
        assert (outdir / f).exists(), f

    import json

    summary = json.loads((outdir / "results.json").read_text())
    assert summary["screening"] is True
    assert summary["edges"], "no edges written"
    for e in summary["edges"]:
        assert "barrier" not in e  # no barrier scalar in screening mode
        assert "delta_e" in e


def test_pipeline_default_screening_off_no_screening_key(tmp_path):
    """Regression: default (screening off) results.json carries no
    "screening" key and every edge keeps its "barrier" estimate."""
    cfg = tiny_cfg(tmp_path, screening=False)
    cfg.search.neb_images = 3
    cfg.search.neb_max_steps = 15
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    assert len(res.edges) == 2
    for e in res.edges:
        assert "barrier" in e

    outdir = pipeline.write_outputs(cfg, res, log=lambda *a, **k: None)

    import json

    summary = json.loads((outdir / "results.json").read_text())
    assert "screening" not in summary
    for e in summary["edges"]:
        assert "barrier" in e


# --- CHE post-processing over screening (no-barrier) edges ------------------

def test_electrochemistry_screening_thermodynamic_span_and_p_side_none():
    """Synthetic-energy CHE integration (mirrors
    test_electrochem.py::test_electrochemistry_integration_ammonia_template)
    but with SCREENING-shaped edges: no "barrier" key anywhere. U_L/U_opt/span
    still compute (purely thermodynamic); P_side is None -- no competing
    barrier was ever computed, so the guard reports "insufficient data"."""
    from autocatpath import electrochem as ec
    from autocatpath.graph import build_graph
    from autocatpath.network import build_network
    from autocatpath.pipeline import Results, _apply_electrochemistry
    from autocatpath.uncertainty import Estimate

    net = build_network(SlabConfig(), kind="ammonia")
    root = net.order()[0]
    assert root == "NO"

    def synth_energy(name: str) -> float:
        return -0.2 * ec.n_H_rel(name, root) + 0.05 * len(name.split("+"))

    node_energies = {
        name: Estimate(mean=synth_energy(name), std=0.01, n=3, values=[])
        for name in net.states()
    }
    # SCREENING mode: no "barrier" key on any reaction-step edge dict.
    edges = [
        {"name": s.name, "reactant": s.reactant.name, "product": s.product.name,
         "delta_e": Estimate(mean=0.0, std=0.0, n=3, values=[])}
        for s in net.steps
    ]
    results = Results(node_energies=node_energies, edges=edges,
                      pathway=net.order(), links=list(net.links))

    ref = node_energies[root].mean
    all_edges = list(edges)
    for a, b in results.links:
        if not any(e["reactant"] == a and e["product"] == b for e in all_edges):
            all_edges.append({"name": f"{a}->{b}", "reactant": a, "product": b,
                              "kind": "supply"})
    g = build_graph(node_energies, all_edges, energy_ref=ref)

    for u, v, d in g.edges(data=True):
        if d.get("kind") == "supply":
            assert d["barrier"] == 0.0  # unchanged pre-existing convention
        else:
            assert "barrier" not in d  # SCREENING: no fabricated 0.0

    cfg = Config(name="che_screening", target="NH3")
    cfg.search.screening = True
    cfg.electrochemistry = ElectrochemistryConfig(U_vs_RHE="optimal")

    che = _apply_electrochemistry(g, cfg, results, log=lambda *a, **k: None)
    assert che is not None
    # U_L only ever depends on node energies -- unaffected by absent barriers.
    assert che["U_L"] is not None and math.isfinite(che["U_L"])
    assert che["U_opt"] is not None and math.isfinite(che["U_opt"])
    assert che["span_at_UL"] is not None and che["span_at_UL"] >= 0.0
    assert che["span_at_Uopt"] is not None and che["span_at_Uopt"] >= 0.0
    # no competitor barrier was ever computed -> "insufficient data", not a
    # fabricated ratio.
    assert che["P_side"] is None


# --- end-to-end: screening survives a full YAML-driven run ------------------

SCREEN_YAML = """
name: screening_smoke
substrate: "NO"
target: "NO3"
network: oxidation
slab: {element: Pd, size: [2, 2, 3], vacuum: 8.0, fix_layers: 1, relax_lattice: false}
mlip: {backend: emt}
search: {seeds: [0], screening: true, max_steps: 40, pose_count: 2}
"""


def test_full_run_screening_mode_carries_no_barrier(tmp_path):
    """The barriers-absent invariant end-to-end, from YAML through the written
    artifacts -- not just at the `run_one_seed` entry the unit tests above poke."""
    import json

    from autocatpath.config import _load_yaml

    data = _load_yaml(SCREEN_YAML)
    data["outdir"] = str(tmp_path)
    cfg = Config.from_dict(data)

    results = pipeline.run(cfg, log=lambda *a, **k: None)
    outdir = pipeline.write_outputs(cfg, results, log=lambda *a, **k: None)

    assert results.edges, "network produced no steps"
    for e in results.edges:
        assert "barrier" not in e
        assert "delta_e" in e

    r = json.loads((outdir / "results.json").read_text())
    assert r.get("screening") is True

    g = json.loads((outdir / "graph.json").read_text())
    reaction_links = [e for e in g["links"] if e.get("kind") != "supply"]
    assert reaction_links
    for e in reaction_links:
        assert "barrier" not in e
