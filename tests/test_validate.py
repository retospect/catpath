import numpy as np

from catpath.config import SlabConfig
from catpath.structures import build_slab, place_fragments
from catpath.validate import geometry_ok, is_similar, rmsd, sanitize_smiles


def slab():
    return build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))


def test_sanitize_smiles():
    assert sanitize_smiles("CCO") is True          # ethanol
    assert sanitize_smiles("c1ccccc1") is True      # benzene
    assert sanitize_smiles("C(C)(C)(C)(C)C") is False  # 5-valent carbon
    assert sanitize_smiles("not a molecule") is False


def test_geometry_detects_clash():
    s = slab()
    n0 = len(s)
    a = place_fragments(s, [
        {"symbol": "O", "site": "fcc", "height": 2.0},
        {"symbol": "O", "site": "fcc", "height": 2.0},  # same spot -> clash
    ])
    rep = geometry_ok(a, n0)
    assert not rep.ok
    assert any("too close" in r for r in rep.reasons)


def test_geometry_detects_detachment():
    s = slab()
    n0 = len(s)
    a = place_fragments(s, [{"symbol": "O", "site": "fcc", "height": 9.0}])
    rep = geometry_ok(a, n0)
    assert not rep.ok
    assert any("detached" in r for r in rep.reasons)


def test_geometry_ok_for_reasonable():
    s = slab()
    n0 = len(s)
    a = place_fragments(s, [{"symbol": "O", "site": "fcc", "height": 1.8}])
    rep = geometry_ok(a, n0)
    assert rep.ok


def test_rmsd_zero_and_positive():
    s = slab()
    n0 = len(s)
    a = place_fragments(s, [{"symbol": "N", "site": "fcc", "height": 2.0},
                            {"symbol": "O", "site": "fcc", "height": 3.2}])
    assert rmsd(a, a, n0) == 0.0
    b = a.copy()
    p = b.get_positions()
    p[n0] += np.array([0.5, 0.0, 0.0])
    b.set_positions(p)
    assert rmsd(a, b, n0) > 0.0
    assert is_similar(a, a, n0, rmsd_thresh=0.1)
