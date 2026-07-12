"""NEB barrier search, including the non-convergence auto-retry.

We drive the retry with a monkeypatched single-attempt runner (no calculator) so
the escalation logic is tested deterministically and fast; a small real EMT run
checks the happy path.
"""

from catpath import neb
from catpath.calculators import make_calculator
from catpath.config import MLIPConfig, SlabConfig
from catpath.structures import build_slab, place_fragments


def test_retry_escalates_until_converged(monkeypatch):
    calls = []

    def fake_attempt(reactant, product, make_calc, n_images, fmax, max_steps, climb):
        calls.append((n_images, max_steps))
        converged = len(calls) == 3          # only the 3rd attempt "converges"
        return [0.0, 0.5, 0.2], converged

    monkeypatch.setattr(neb, "_neb_attempt", fake_attempt)
    r = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0))
    res = neb.neb_barrier(r, r, make_calc=lambda: None,
                          n_images=4, max_steps=10, retries=3)
    assert res.converged and res.method == "ci-neb+retry2"
    assert len(calls) == 3
    # each retry uses a denser band and a bigger step budget
    assert [c[0] for c in calls] == [4, 6, 9]      # ~1.5x images
    assert [c[1] for c in calls] == [10, 20, 40]   # 2x steps


def test_no_retry_when_first_attempt_converges(monkeypatch):
    def fake_attempt(*a, **k):
        return [0.0, 0.3, 0.1], True

    monkeypatch.setattr(neb, "_neb_attempt", fake_attempt)
    r = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0))
    res = neb.neb_barrier(r, r, make_calc=lambda: None, retries=3)
    assert res.converged and res.method == "ci-neb"


def test_returns_last_attempt_when_never_converges(monkeypatch):
    seen = []

    def fake_attempt(reactant, product, make_calc, n_images, fmax, max_steps, climb):
        seen.append(n_images)
        return [0.0, 0.9, 0.4], False

    monkeypatch.setattr(neb, "_neb_attempt", fake_attempt)
    r = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0))
    res = neb.neb_barrier(r, r, make_calc=lambda: None,
                          n_images=3, retries=2)
    assert not res.converged and res.method == "ci-neb+retry2"
    assert len(seen) == 3                            # 1 + 2 retries


def test_real_neb_runs_on_emt():
    slab = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))
    r = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    p = place_fragments(slab, [{"symbol": "O", "site": "hcp", "height": 2.0, "dx": 1.0}])
    res = neb.neb_barrier(r, p, make_calc=lambda: make_calculator(MLIPConfig()),
                          n_images=3, max_steps=25, retries=1)
    assert res.barrier >= 0.0 and len(res.images_energy) >= 5
