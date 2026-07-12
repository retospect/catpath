import numpy as np
from ase.constraints import FixAtoms

from catpath.config import SlabConfig
from catpath.structures import (
    build_slab,
    place_fragments,
    poses,
    rattle_adsorbate,
    symbols_of,
)


def small_slab():
    return build_slab(SlabConfig(element="Pd", size=(2, 2, 3), vacuum=8.0, fix_layers=1))


def test_slab_atom_count_and_constraints():
    slab = small_slab()
    assert len(slab) == 2 * 2 * 3
    assert slab.info["n_slab"] == len(slab)
    cons = slab.constraints
    assert any(isinstance(c, FixAtoms) for c in cons)
    # exactly one layer (2x2 = 4 atoms) frozen
    fixed = np.concatenate([c.index for c in cons if isinstance(c, FixAtoms)])
    assert len(fixed) == 4


def test_place_fragments_preserves_order():
    slab = small_slab()
    n0 = len(slab)
    a = place_fragments(slab, [
        {"symbol": "N", "site": "fcc", "height": 1.8},
        {"symbol": "O", "site": "fcc", "height": 3.0},
    ])
    assert len(a) == n0 + 2
    assert a.get_chemical_symbols()[n0:] == ["N", "O"]


def test_rattle_only_moves_adsorbate():
    slab = small_slab()
    n0 = len(slab)
    a = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    r = rattle_adsorbate(a, n0, seed=1, amplitude=0.2)
    # slab atoms unchanged, adsorbate moved
    assert np.allclose(a.get_positions()[:n0], r.get_positions()[:n0])
    assert not np.allclose(a.get_positions()[n0:], r.get_positions()[n0:])


def test_rattle_is_deterministic():
    slab = small_slab()
    n0 = len(slab)
    a = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    r1 = rattle_adsorbate(a, n0, seed=7)
    r2 = rattle_adsorbate(a, n0, seed=7)
    assert np.allclose(r1.get_positions(), r2.get_positions())


def test_poses_count_and_symbols():
    slab = small_slab()
    ps = poses(slab, ["N", "O"], count=4, seed=0)
    assert len(ps) == 4
    assert symbols_of(ps[0]) >= {"Pd", "N", "O"}
