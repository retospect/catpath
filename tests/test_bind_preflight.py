"""Endpoint binding pre-flight (see pipeline._relax_state_bound / run_one_seed).

Gates the (expensive) climbing-image NEB on whether each endpoint's adsorbate
actually binds: a detached endpoint gets a restrained-then-released reseat
attempt before the barrier search is allowed to run at all.  EMT + a tiny slab
keeps the happy paths fast and real; the "does it reseat / does it stay
unbound" branches are driven by a monkeypatched ``geometry_ok`` /
``_relax_state_bound`` so the test doesn't depend on EMT's real potential
happening to hold a released adsorbate in a particular spot.
"""

from types import SimpleNamespace

from catpath import pipeline
from catpath.config import Config, SlabConfig
from catpath.network import Network, StateSpec, StepSpec
from catpath.relax import RelaxResult
from catpath.structures import build_slab
from catpath.validate import GeometryReport, geometry_ok


def tiny_cfg(tmp_path):
    cfg = Config(name="bindtest", outdir=str(tmp_path))
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0]
    cfg.search.max_steps = 15
    cfg.search.neb_images = 2
    cfg.search.neb_max_steps = 8
    return cfg


def bound_state():
    return StateSpec("O_bound", "O", [{"symbol": "O", "site": "fcc", "height": 2.0}])


def far_state():
    return StateSpec("O_far", "O", [{"symbol": "O", "site": "fcc", "height": 9.0}])


def _dummy_result(atoms):
    return RelaxResult(atoms=atoms, energy=0.0, fmax=0.01, steps=1, converged=True)


def one_step_network(slab_cfg):
    reactant = StateSpec("R", "R", [{"symbol": "O", "site": "fcc", "height": 2.0}])
    product = StateSpec("P", "P", [{"symbol": "O", "site": "hcp", "height": 2.0, "dx": 1.0}])
    return Network(slab_cfg=slab_cfg, steps=[StepSpec("R->P", reactant, product)])


# --- _relax_state_bound: unit tests -------------------------------------------

def test_binds_on_first_try_no_reseat_needed(tmp_path):
    cfg = tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]
    res, geo, bound = pipeline._relax_state_bound(bound_state(), slab, n_slab, cfg, seed=0)
    assert bound is True
    assert geo.ok
    assert res.atoms is not None


def test_reseats_after_tether_release(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]
    real_geo = geometry_ok
    calls = {"n": 0}

    def fake_geo(atoms, n_slab_, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:  # the plain (untethered) relax: report detached
            return GeometryReport(ok=False, min_dist=0.0, adsorbate_height=9.0,
                                  reasons=["adsorbate atom 4 detached from slab (9.00 A)"])
        return real_geo(atoms, n_slab_, *a, **k)  # post-tether-release: real check

    monkeypatch.setattr(pipeline, "geometry_ok", fake_geo)
    _res, geo, bound = pipeline._relax_state_bound(far_state(), slab, n_slab, cfg, seed=0)
    assert bound is True
    assert geo.ok
    assert calls["n"] == 2  # 1 plain check + 1 post-reseat check (reseated on first attempt)


def test_never_rebinds_stays_unbound(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]

    def always_detached(atoms, n_slab_, *a, **k):
        return GeometryReport(ok=False, min_dist=0.0, adsorbate_height=9.0,
                              reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    monkeypatch.setattr(pipeline, "geometry_ok", always_detached)
    _res, geo, bound = pipeline._relax_state_bound(far_state(), slab, n_slab, cfg, seed=0)
    assert bound is False
    assert not geo.ok
    assert geo.adsorbate_height == 9.0


# --- run_one_seed: NEB gating --------------------------------------------------

def test_run_one_seed_skips_neb_when_endpoint_unbound(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0, reasons=[])
    bad_geo = GeometryReport(ok=False, min_dist=1.0, adsorbate_height=9.0,
                             reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    def fake_bound(state, slab_, n_slab_, cfg_, seed):
        atoms = state.build(slab_)
        res = _dummy_result(atoms)
        if state.name == "P":  # the product never binds
            return res, bad_geo, False
        return res, ok_geo, True

    monkeypatch.setattr(pipeline, "_relax_state_bound", fake_bound)

    def fake_neb(*a, **k):
        raise AssertionError("NEB must not run when an endpoint is unbound")

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert entry["barrier"] is None
    assert entry["delta_e"] is None
    assert any("INFEASIBLE" in w and "detached" in w and "P " in w
               for w in part["warnings"]), part["warnings"]


def test_bind_preflight_false_reproduces_old_behavior(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    cfg.search.bind_preflight = False
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    bad_geo = GeometryReport(ok=False, min_dist=1.0, adsorbate_height=9.0,
                             reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    def fake_relax_state(state, slab_, n_slab_, cfg_, seed):
        atoms = state.build(slab_)
        return _dummy_result(atoms), bad_geo  # both endpoints geometrically "detached"

    monkeypatch.setattr(pipeline, "_relax_state", fake_relax_state)

    def fail_if_called(*a, **k):
        raise AssertionError("bind_preflight=False must not call _relax_state_bound")

    monkeypatch.setattr(pipeline, "_relax_state_bound", fail_if_called)

    called = {"neb": False}

    def fake_neb(*a, **k):
        called["neb"] = True
        return SimpleNamespace(barrier=1.23, delta_e=0.1, converged=True)

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert called["neb"] is True  # NEB ran despite both endpoints "detached"
    assert entry["barrier"] == 1.23
    assert not any("INFEASIBLE" in w for w in part["warnings"])
