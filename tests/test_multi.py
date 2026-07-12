"""Multi-substrate run -> union-column energy map (fast, tiny slab, EMT)."""

import numpy as np

from catpath.config import Config, SlabConfig
from catpath.multi import run_multi, write_multi


def tiny(tmp_path):
    cfg = Config(name="mu", outdir=str(tmp_path))
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 15
    cfg.search.neb_images = 2
    cfg.search.neb_max_steps = 8
    # two DIFFERENT networks -> the energy-map columns must be a union
    cfg.substrates = [
        {"substrate": "NO", "target": "NO3", "network": "oxidation"},
        {"substrate": "NO", "target": "N+O", "network": "ammonia", "reagents": []},
    ]
    return cfg


def test_substrate_runs_normalises_dicts(tmp_path):
    cfg = tiny(tmp_path)
    specs = cfg.substrate_runs()
    assert [s.network for s in specs] == ["oxidation", "ammonia"]
    assert specs[1].reagents == []          # explicit dict override survives


def test_multi_union_columns_and_outputs(tmp_path):
    cfg = tiny(tmp_path)
    multi = run_multi(cfg, log=lambda *a, **k: None)

    assert multi.matrix.shape == (2, len(multi.col_labels))
    assert len(multi.per_run) == 2
    # columns are the UNION of both networks' states
    assert {"NO3", "NO@top"} <= set(multi.col_labels)
    # oxidation row has no NO@top; dissociation row has no NO3 -> genuine NaNs
    assert np.isnan(multi.matrix).any()
    # every row is referenced to its own root -> contains a (near) zero
    for row in multi.matrix:
        assert np.nanmin(np.abs(row)) < 1e-6

    outdir = write_multi(cfg, multi, log=lambda *a, **k: None)
    for f in ["energy_map.png", "energy_map.csv", "multi.json"]:
        assert (outdir / f).exists(), f
