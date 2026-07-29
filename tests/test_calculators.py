"""Backend registry + the safe `auto` resolver.

Installed-ness is monkeypatched so the selection logic is tested without needing
the heavy ML stacks; only EMT is actually constructed.
"""

import pytest

from autocatpath import calculators as C
from autocatpath.config import MLIPConfig


def _only(installed: set[str], monkeypatch):
    """Pretend exactly `installed` ML backends are importable."""
    monkeypatch.setattr(C, "_installed",
                        lambda b: b == "emt" or b in installed)


def test_resolve_passes_concrete_backends_through():
    for b in ("emt", "mace", "chgnet", "fairchem", "grace"):
        assert C.resolve_backend(b) == b


def test_auto_picks_best_installed_in_preference_order(monkeypatch):
    _only({"chgnet", "grace"}, monkeypatch)      # mace absent -> next in AUTO_ORDER
    assert C.resolve_backend("auto") == "grace"  # grace precedes chgnet
    _only({"mace", "chgnet"}, monkeypatch)
    assert C.resolve_backend("auto") == "mace"   # mace wins when present


def test_auto_errors_when_no_ml_installed(monkeypatch):
    _only(set(), monkeypatch)                     # only emt available
    with pytest.raises(RuntimeError) as e:
        C.resolve_backend("auto")
    assert "no ML potential installed" in str(e.value)
    assert "autocatpath[mace]" in str(e.value)         # actionable install hint


def test_available_backends_shape():
    av = C.available_backends()
    assert av["emt"] is True
    assert set(av) == {"emt", "mace", "chgnet", "fairchem", "grace"}


def test_dtype_defaults_to_float64():
    assert MLIPConfig().dtype == "float64"


def test_dtype_parsed_from_config():
    from autocatpath.config import Config

    cfg = Config.from_dict(
        {"substrate": "NO", "target": "NH3",
         "mlip": {"backend": "mace", "dtype": "float32"}}
    )
    assert cfg.mlip.dtype == "float32"


def test_calculator_cache_keys_on_dtype(monkeypatch):
    """float32 and float64 must not share a cached calculator — the dtype is
    part of the memo key, so a screening (float32) run never reuses an
    accurate (float64) instance."""
    C.reset_calculator_cache()
    loads: list[str] = []
    monkeypatch.setattr(C, "_load", lambda backend, cfg: loads.append(cfg.dtype) or object())
    C.make_calculator(MLIPConfig(backend="mace", dtype="float64"))
    C.make_calculator(MLIPConfig(backend="mace", dtype="float64"))  # cache hit
    C.make_calculator(MLIPConfig(backend="mace", dtype="float32"))  # distinct key
    assert loads == ["float64", "float32"]
    C.reset_calculator_cache()


def test_emt_element_guard():
    # emt supports the metal + N/O/H; rejects something it has no params for
    C.check_supported({"Pd", "N", "O", "H"}, MLIPConfig(backend="emt"))
    with pytest.raises(ValueError):
        C.check_supported({"Fe"}, MLIPConfig(backend="emt"))


def test_ml_backends_have_no_element_restriction(monkeypatch):
    _only({"mace"}, monkeypatch)
    # auto -> mace (universal): any element set is accepted
    C.check_supported({"Fe", "Ru", "W", "N"}, MLIPConfig(backend="auto"))


def test_uninstalled_backend_gives_actionable_error():
    # chgnet isn't installed in this env -> friendly RuntimeError, not ImportError
    with pytest.raises(RuntimeError) as e:
        C.make_calculator(MLIPConfig(backend="chgnet"))
    assert "autocatpath[chgnet]" in str(e.value)


def test_ml_calculator_is_memoized(monkeypatch):
    """An ML backend loads once and is reused for the same (backend, model,
    device, task) — the load, not the force eval, dominates a run."""
    calls = {"n": 0}
    monkeypatch.setattr(C, "_load", lambda backend, cfg: calls.__setitem__("n", calls["n"] + 1) or object())
    C.reset_calculator_cache()
    try:
        cfg = MLIPConfig(backend="mace", model="medium", device="cuda")
        a = C.make_calculator(cfg)
        b = C.make_calculator(cfg)
        assert a is b               # same instance reused
        assert calls["n"] == 1      # model built exactly once

        # a differing cfg field is a distinct cache entry
        c = C.make_calculator(MLIPConfig(backend="mace", model="medium", device="cpu"))
        assert c is not a and calls["n"] == 2

        # cache=False always rebuilds
        d = C.make_calculator(cfg, cache=False)
        assert d is not a and calls["n"] == 3
    finally:
        C.reset_calculator_cache()


def test_emt_calculator_is_never_cached():
    """EMT is cheap + stateless — always hand back a fresh instance (keeps the
    light/CI path free of process-global state)."""
    C.reset_calculator_cache()
    a = C.make_calculator(MLIPConfig(backend="emt"))
    b = C.make_calculator(MLIPConfig(backend="emt"))
    assert a is not b
