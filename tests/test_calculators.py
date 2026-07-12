"""Backend registry + the safe `auto` resolver.

Installed-ness is monkeypatched so the selection logic is tested without needing
the heavy ML stacks; only EMT is actually constructed.
"""

import pytest

from catpath import calculators as C
from catpath.config import MLIPConfig


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
    assert "catpath[mace]" in str(e.value)         # actionable install hint


def test_available_backends_shape():
    av = C.available_backends()
    assert av["emt"] is True
    assert set(av) == {"emt", "mace", "chgnet", "fairchem", "grace"}


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
    assert "catpath[chgnet]" in str(e.value)
