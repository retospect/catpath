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

from ase import Atoms

from autocatpath import pipeline
from autocatpath.config import Config, SlabConfig
from autocatpath.network import Network, StateSpec, StepSpec
from autocatpath.relax import RelaxResult
from autocatpath.structures import build_slab
from autocatpath.validate import GeometryReport, geometry_ok


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
    res, geo, bound, diag = pipeline._relax_state_bound(bound_state(), slab, n_slab, cfg, seed=0)
    assert bound is True
    assert geo.ok
    assert res.atoms is not None
    assert diag["reseated"] is False and diag["attempts"] == []  # no reseat needed


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
    _res, geo, bound, diag = pipeline._relax_state_bound(far_state(), slab, n_slab, cfg, seed=0)
    assert bound is True
    assert geo.ok
    assert calls["n"] == 2  # 1 plain check + 1 post-reseat check (reseated on first attempt)
    assert diag["reseated"] is True and len(diag["attempts"]) == 1


def test_never_rebinds_stays_unbound(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]

    def always_detached(atoms, n_slab_, *a, **k):
        return GeometryReport(ok=False, min_dist=0.0, adsorbate_height=9.0,
                              reasons=["adsorbate atom 4 detached from slab (9.00 A)"])

    monkeypatch.setattr(pipeline, "geometry_ok", always_detached)
    cfg.search.bind_reseat_attempts = 3
    _res, geo, bound, diag = pipeline._relax_state_bound(far_state(), slab, n_slab, cfg, seed=0)
    assert bound is False
    assert not geo.ok
    assert geo.adsorbate_height == 9.0
    assert diag["reseated"] is False and len(diag["attempts"]) == 3  # all tries exhausted


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
            return res, bad_geo, False, None
        return res, ok_geo, True, None

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


# --- fragment-aware geometry_ok ------------------------------------------------

def _square_slab():
    """4 metal atoms in the z=0 plane; hollow centre at (1.375, 1.375)."""
    return [(0, 0, 0), (2.75, 0, 0), (0, 2.75, 0), (2.75, 2.75, 0)]


def test_bound_heavy_atom_with_up_pointing_H_is_not_detached():
    # N bound over the hollow (nearest metal ~2.8 A); H bonded 1.0 A above N, so
    # its nearest-metal distance (~3.6 A) exceeds the flat 3.5 A cutoff. Per-atom
    # logic would (wrongly) call this desorbed; per-fragment logic keeps it bound.
    slab = _square_slab()
    atoms = Atoms("PdPdPdPdNH", positions=slab + [(1.375, 1.375, 2.0), (1.375, 1.375, 3.0)])
    geo = geometry_ok(atoms, n_slab=4)
    assert geo.ok, geo.reasons
    assert geo.detached_fragments == 0
    assert geo.n_fragments == 1  # N-H is one bonded fragment
    assert geo.adsorbate_height > 3.5 > geo.anchor_height  # worst floats, anchor binds


def test_floated_molecule_and_dissociated_adatom_still_flagged():
    slab = _square_slab()
    # whole NH lifted off (both atoms far); plus an O adatom parked in vacuum.
    atoms = Atoms("PdPdPdPdNHO",
                  positions=slab + [(1.375, 1.375, 5.0), (1.375, 1.375, 6.0),
                                    (10.0, 10.0, 5.0)])
    geo = geometry_ok(atoms, n_slab=4)
    assert not geo.ok
    assert geo.detached_fragments == 2  # NH fragment + lone O fragment
    assert any("detached" in r for r in geo.reasons)  # trust-gate substring preserved


def test_reseated_diagnostic_has_no_trustgate_substrings(tmp_path, monkeypatch):
    # A rescued (reseated -> now bound) endpoint must emit an INFO line that does
    # NOT contain "detached" or "NEB not converged", or the precis trust-gate
    # would wrongly count the barrier as untrusted.
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)
    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=3.6, reasons=[],
                            anchor_height=2.1)
    diag = {"state": "x", "rt": 2.0, "k": 7.5, "plain_anchor": 3.9, "plain_worst": 3.9,
            "attempts": [{"anchor": 2.1, "worst": 3.6, "ok": True}], "reseated": True}

    def fake_bound(state, slab_, n_slab_, cfg_, seed):
        return _dummy_result(state.build(slab_)), ok_geo, True, diag

    monkeypatch.setattr(pipeline, "_relax_state_bound", fake_bound)
    monkeypatch.setattr(pipeline, "neb_barrier",
                        lambda *a, **k: SimpleNamespace(barrier=1.0, delta_e=0.1, converged=True))
    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    assert any("RESEATED" in w for w in part["warnings"])  # rescue is recorded
    assert not any("detached" in w for w in part["warnings"])
    assert not any("NEB not converged" in w for w in part["warnings"])
    assert part["steps"]["R->P"]["barrier"] == 1.0  # NEB ran on the rescued endpoint
