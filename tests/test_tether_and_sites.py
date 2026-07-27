"""Dissolving tether, adsorption-barrier gate, mid-band desorption, and the
bond-site (``*``) identity check — the desorption-hardening pass.

Deterministic where it matters: the branch logic (activated-adsorption gate,
wrong-site skip, mid-band count) is driven by monkeypatched geometry / barrier
so it doesn't hinge on EMT holding a particular pose; one real EMT run pins the
adsorption barrier as physically ~0 for barrierless O/Pd.
"""


from ase import Atoms

from catpath import neb, pipeline
from catpath.config import Config, SlabConfig
from catpath.network import Network, StateSpec, StepSpec
from catpath.relax import RelaxResult
from catpath.structures import build_slab, place_fragments
from catpath.validate import GeometryReport, binding_site_ok


def _tiny_cfg(tmp_path):
    cfg = Config(name="teth", outdir=str(tmp_path))
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0]
    cfg.search.max_steps = 15
    return cfg


def _far_O():
    return StateSpec("O_far", "O", [{"symbol": "O", "site": "fcc", "height": 9.0}])


def _dummy(atoms):
    return RelaxResult(atoms=atoms, energy=0.0, fmax=0.01, steps=1, converged=True)


# --- adsorption barrier -------------------------------------------------------

def test_adsorption_barrier_barrierless_is_near_zero(tmp_path):
    # O approaching a clean Pd(111) fcc site is (nearly) barrierless; a RELAXED
    # band (not a straight line) must report ~0, not a lateral-clip artifact.
    cfg = _tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    desorbed = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 6.0}])
    bound = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    ab = pipeline._adsorption_barrier(desorbed, bound, cfg)
    assert 0.0 <= ab < 0.4, ab  # barrierless-ish; certainly below bind_ads_barrier_max


def test_activated_adsorption_reported_nonbinding(tmp_path, monkeypatch):
    # A reseat that is geometrically bound but only reached by climbing a big
    # adsorption barrier is genuine activated adsorption -> reported non-binding.
    cfg = _tiny_cfg(tmp_path)
    cfg.search.bind_ads_barrier_max = 0.75
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]

    calls = {"n": 0}

    def fake_geo(atoms, n_slab_, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:  # plain relax detached
            return GeometryReport(ok=False, min_dist=0.0, adsorbate_height=9.0,
                                  reasons=["adsorbate atom 4 detached from slab (9.00 A)"])
        return GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0,
                              reasons=[], anchor_height=2.0)  # reseat geometrically ok

    monkeypatch.setattr(pipeline, "geometry_ok", fake_geo)
    monkeypatch.setattr(pipeline, "_adsorption_barrier", lambda *a, **k: 1.5)  # > max

    _res, _geo, bound, diag = pipeline._relax_state_bound(_far_O(), slab, n_slab, cfg, seed=0)
    assert bound is False                       # activated -> not spontaneously binding
    assert diag["reseated"] is False
    assert diag["ads_barrier"] == 1.5           # the barrier that made it untrusted


def test_low_adsorption_barrier_reseats_bound(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]
    calls = {"n": 0}

    def fake_geo(atoms, n_slab_, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return GeometryReport(ok=False, min_dist=0.0, adsorbate_height=9.0,
                                  reasons=["adsorbate atom 4 detached from slab (9.00 A)"])
        return GeometryReport(ok=True, min_dist=1.0, adsorbate_height=2.0,
                              reasons=[], anchor_height=2.0)

    monkeypatch.setattr(pipeline, "geometry_ok", fake_geo)
    monkeypatch.setattr(pipeline, "_adsorption_barrier", lambda *a, **k: 0.05)  # barrierless

    _res, _geo, bound, diag = pipeline._relax_state_bound(_far_O(), slab, n_slab, cfg, seed=0)
    assert bound is True and diag["reseated"] is True
    assert diag["ads_barrier"] == 0.05


def test_dissolving_ramp_runs_each_stage(tmp_path, monkeypatch):
    # the reseat relaxes once per ramp stage (adiabatic k->0), not a single drop.
    cfg = _tiny_cfg(tmp_path)
    cfg.search.bind_tether_ramp = [1.0, 0.5, 0.0]
    slab = build_slab(cfg.slab)
    n_slab = slab.info["n_slab"]
    relax_calls = {"n": 0}
    real_relax = pipeline.relax

    def counting_relax(*a, **k):
        relax_calls["n"] += 1
        return real_relax(*a, **k)

    monkeypatch.setattr(pipeline, "relax", counting_relax)
    start = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 3.0}])
    _res, _geo = pipeline._dissolve_reseat(start, n_slab, cfg, 7.5, 2.0,
                                         cfg.search.bind_tether_ramp)
    assert relax_calls["n"] == 3  # one relax per ramp fraction


# --- bond-site (*) identity ---------------------------------------------------

def test_binding_site_ok_accepts_correct_and_flags_flip():
    slab = [(0, 0, 0), (2.75, 0, 0), (0, 2.75, 0), (2.75, 2.75, 0)]
    ph = [1.8, 3.0]  # N is the * (low); O up
    correct = Atoms("PdPdPdPdNO",
                    positions=slab + [(1.375, 1.375, 2.0), (1.375, 1.375, 3.1)])
    flipped = Atoms("PdPdPdPdNO",
                    positions=slab + [(1.375, 1.375, 3.1), (1.375, 1.375, 2.0)])
    ok, _ = binding_site_ok(correct, 4, ph)
    assert ok
    bad_ok, reasons = binding_site_ok(flipped, 4, ph)
    assert not bad_ok and any("wrong-site" in r for r in reasons)


def test_symmetric_swap_is_not_wrong_site():
    # two equivalent O's of an NO2*-like fragment swapping is fine (same element).
    slab = [(0, 0, 0), (2.75, 0, 0), (0, 2.75, 0), (2.75, 2.75, 0)]
    ph = [2.0, 2.4, 2.4]  # N is *; two O's
    atoms = Atoms("PdPdPdPdNOO",
                  positions=slab + [(1.375, 1.375, 2.0), (2.0, 1.4, 2.4), (0.8, 1.4, 2.4)])
    ok, _ = binding_site_ok(atoms, 4, ph)
    assert ok


def test_wrong_site_endpoint_skips_neb_and_warns(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    reactant = StateSpec("R", "R", [{"symbol": "N", "site": "fcc", "height": 1.8},
                                    {"symbol": "O", "site": "fcc", "height": 3.0}])
    product = StateSpec("P", "P", [{"symbol": "N", "site": "hcp", "height": 1.8, "dx": 1.0},
                                   {"symbol": "O", "site": "hcp", "height": 3.0, "dx": 1.0}])
    net = Network(slab_cfg=cfg.slab, steps=[StepSpec("R->P", reactant, product)])
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    ok_geo = GeometryReport(ok=True, min_dist=1.0, adsorbate_height=3.1,
                            reasons=[], anchor_height=2.0)

    def flipped_bound(state, slab_, n_slab_, cfg_, seed):
        # build the state but bind through O (swap N/O z) -> wrong-site
        atoms = state.build(slab_)
        pos = atoms.get_positions()
        pos[[n_slab_, n_slab_ + 1]] = pos[[n_slab_ + 1, n_slab_]]
        atoms.set_positions(pos)
        return _dummy(atoms), ok_geo, True, None

    monkeypatch.setattr(pipeline, "_relax_state_bound", flipped_bound)
    monkeypatch.setattr(pipeline, "neb_barrier",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("NEB must not run on a mis-bound endpoint")))

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    assert part["steps"]["R->P"]["barrier"] is None
    assert any("wrong-site" in w for w in part["warnings"]), part["warnings"]


# --- mid-band desorption ------------------------------------------------------

def test_count_desorbed_flags_floating_interior_image():
    slab = [(0, 0, 0), (2.75, 0, 0), (0, 2.75, 0), (2.75, 2.75, 0)]
    bound = Atoms("PdPdPdPdO", positions=slab + [(1.375, 1.375, 2.0)])
    gone = Atoms("PdPdPdPdO", positions=slab + [(1.375, 1.375, 9.0)])
    images = [bound, bound, gone, bound, bound]  # endpoints excluded; 1 interior gone
    assert neb._count_desorbed(images, n_slab=4) == 1


def test_neb_barrier_reports_desorbed_images(monkeypatch):
    slab = [(0, 0, 0), (2.75, 0, 0), (0, 2.75, 0), (2.75, 2.75, 0)]
    bound = Atoms("PdPdPdPdO", positions=slab + [(1.375, 1.375, 2.0)])
    gone = Atoms("PdPdPdPdO", positions=slab + [(1.375, 1.375, 9.0)])
    band = [bound, bound, gone, bound, bound]

    def fake_attempt(*a, **k):
        return [0.0, 0.3, 0.9, 0.3, 0.0], True, band

    monkeypatch.setattr(neb, "_neb_attempt", fake_attempt)
    res = neb.neb_barrier(bound, bound, make_calc=lambda: None, n_slab=4)
    assert res.desorbed_images == 1
