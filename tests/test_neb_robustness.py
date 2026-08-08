"""Two NEB robustness features (see pipeline.run_one_seed / neb.neb_barrier):

* mid-band detachment: an interior band image can fly off the slab even when
  both endpoints bind -- `_mid_band_detachment` catches it with the same
  `geometry_ok` criterion the endpoint bind pre-flight already uses.
* auto-retry: a non-converged NEB gets ONE more attempt with double the step
  budget (band density / fmax untouched) before falling back to today's
  non-converged flagging.

Both are driven by a monkeypatched ``pipeline.neb_barrier`` (SimpleNamespace
stand-ins for ``BarrierResult``) so the tests are fast and deterministic --
matching the style of ``test_bind_preflight.py``.
"""

from types import SimpleNamespace

from autocatpath import pipeline
from autocatpath.config import Config, SlabConfig
from autocatpath.network import Network, StateSpec, StepSpec
from autocatpath.relax import RelaxResult
from autocatpath.structures import build_slab, place_fragments


def tiny_cfg(tmp_path):
    cfg = Config(name="nebrobust", outdir=str(tmp_path))
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0]
    cfg.search.max_steps = 15
    cfg.search.neb_images = 2
    cfg.search.neb_max_steps = 8
    return cfg


def one_step_network(slab_cfg):
    reactant = StateSpec("R", "R", [{"symbol": "O", "site": "fcc", "height": 2.0}])
    product = StateSpec("P", "P", [{"symbol": "O", "site": "hcp", "height": 2.0, "dx": 1.0}])
    return Network(slab_cfg=slab_cfg, steps=[StepSpec("R->P", reactant, product)])


def _dummy_result(atoms):
    return RelaxResult(atoms=atoms, energy=0.0, fmax=0.01, steps=1, converged=True)


def _bound_band(n_slab, slab, n=4):
    """A band of well-bound images (all adsorbate atoms close to the slab)."""
    return [place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
            for _ in range(n)]


# --- _mid_band_detachment: unit tests -----------------------------------------

def _bound_image(slab):
    return place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])


def test_mid_band_detachment_flags_flying_interior_image():
    slab = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))
    n_slab = len(slab)
    flown = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 9.0}])
    band = [_bound_image(slab), _bound_image(slab), flown,
            _bound_image(slab), _bound_image(slab)]  # image 3 of 5 detached
    bar = SimpleNamespace(images=band)

    hit = pipeline._mid_band_detachment(bar, n_slab)
    assert hit is not None
    k, n, d = hit
    assert (k, n) == (3, 5)
    assert d > 3.5  # geometry_ok's default max_ads_height


def test_mid_band_detachment_none_when_whole_band_bound():
    slab = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))
    n_slab = len(slab)
    band = _bound_band(n_slab, slab, n=4)
    bar = SimpleNamespace(images=band)
    assert pipeline._mid_band_detachment(bar, n_slab) is None


def test_mid_band_detachment_ignores_endpoints():
    """A detached *endpoint* (index 1 or N) is the bind pre-flight's job, not
    this interior-only check."""
    slab = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))
    n_slab = len(slab)
    flown = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 9.0}])
    band = [flown, _bound_image(slab), _bound_image(slab), flown]  # only endpoints detached
    bar = SimpleNamespace(images=band)
    assert pipeline._mid_band_detachment(bar, n_slab) is None


def test_mid_band_detachment_no_images_is_none():
    # a stub BarrierResult without an `images` attribute at all
    bar = SimpleNamespace(barrier=1.0, delta_e=0.1, converged=True)
    assert pipeline._mid_band_detachment(bar, n_slab=10) is None


# --- run_one_seed: mid-band flag wiring ----------------------------------------

def test_run_one_seed_flags_low_confidence_on_mid_band_detachment(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    slab = build_slab(cfg.slab)
    flown = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 9.0}])
    band = [_bound_image(slab), flown, _bound_image(slab)]  # 1 interior image, detached

    def fake_neb(*a, **k):
        return SimpleNamespace(barrier=1.5, delta_e=0.2, converged=True, images=band)

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert entry["barrier"] == 1.5  # value is kept, not discarded
    assert entry.get("low_confidence") is True
    assert any("detached mid-band" in w and "barrier untrustworthy" in w
               for w in part["warnings"]), part["warnings"]


def test_run_one_seed_no_low_confidence_when_band_stays_bound(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    slab = build_slab(cfg.slab)
    band = _bound_band(slab.info["n_slab"], slab, n=3)

    def fake_neb(*a, **k):
        return SimpleNamespace(barrier=1.5, delta_e=0.2, converged=True, images=band)

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]
    assert not entry.get("low_confidence")
    assert not any("detached mid-band" in w for w in part["warnings"])


# --- run_one_seed: non-convergence auto-retry ----------------------------------

def test_run_one_seed_retries_once_with_doubled_steps_and_uses_converged_result(
    tmp_path, monkeypatch
):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    calls = []

    def fake_neb(reactant, product, make_calc, n_images, fmax, max_steps, retries, **kw):
        calls.append({"max_steps": max_steps, "n_images": n_images,
                      "fmax": fmax, "retries": retries,
                      "n_slab": kw.get("n_slab"), "tether": kw.get("tether")})
        converged = len(calls) == 2  # first attempt fails, retry converges
        return SimpleNamespace(barrier=2.0 if converged else 9.0,
                               delta_e=0.3, converged=converged, images=[])

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]

    assert len(calls) == 2                       # exactly one retry
    assert calls[0]["max_steps"] == cfg.search.neb_max_steps
    assert calls[1]["max_steps"] == cfg.search.neb_max_steps * 2   # steps doubled
    assert calls[1]["n_images"] == calls[0]["n_images"]            # band unchanged
    assert calls[1]["fmax"] == calls[0]["fmax"]                    # fmax unchanged
    assert calls[1]["retries"] == 0               # the retry itself doesn't chain-retry
    # the retry inherits the dissolving-tether band settle -- an untethered retry
    # would reintroduce the mid-band desorption the tether exists to prevent
    assert calls[1]["n_slab"] == calls[0]["n_slab"] is not None
    assert calls[1]["tether"] == calls[0]["tether"] is not None

    assert entry["barrier"] == 2.0                # uses the retry's (converged) result
    assert not entry.get("low_confidence")         # info-level, not low_confidence
    assert any("converged on retry" in w for w in part["warnings"]), part["warnings"]
    assert not any(w == "R->P seed=0 NEB not converged" for w in part["warnings"])


def test_run_one_seed_keeps_existing_flagging_when_retry_still_fails(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    calls = []

    def fake_neb(reactant, product, make_calc, n_images, fmax, max_steps, retries, **kw):
        calls.append(max_steps)
        return SimpleNamespace(barrier=3.0, delta_e=0.1, converged=False, images=[])

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    entry = part["steps"]["R->P"]

    assert len(calls) == 2  # first attempt + one retry, then give up
    assert calls[1] == calls[0] * 2
    assert entry["barrier"] == 3.0  # last (most refined) attempt's value, same as today
    warns = part["warnings"]
    assert warns.count("R->P seed=0 NEB not converged") == 1  # unchanged, not duplicated
    assert not any("converged on retry" in w for w in warns)


def test_run_one_seed_auto_retry_opt_out(tmp_path, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    cfg.search.neb_auto_retry = False
    net = one_step_network(cfg.slab)
    monkeypatch.setattr(pipeline, "_build_net", lambda c, log=lambda *a, **k: None: net)

    calls = []

    def fake_neb(reactant, product, make_calc, n_images, fmax, max_steps, retries, **kw):
        calls.append(max_steps)
        return SimpleNamespace(barrier=3.0, delta_e=0.1, converged=False, images=[])

    monkeypatch.setattr(pipeline, "neb_barrier", fake_neb)

    part = pipeline.run_one_seed(cfg, seed=0, log=lambda *a, **k: None)
    assert len(calls) == 1  # opted out -- no retry attempt at all
    assert any("NEB not converged" in w for w in part["warnings"])
