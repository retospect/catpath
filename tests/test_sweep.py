"""Multi-surface sweep -> multi-row energy map (fast, tiny slab, EMT)."""

import numpy as np

from catpath.config import Config, SlabConfig
from catpath.sweep import run_sweep, write_sweep


def tiny(tmp_path):
    cfg = Config(name="sw", outdir=str(tmp_path))
    cfg.network = "oxidation"  # fast linear chain for the sweep test
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 15
    cfg.search.neb_images = 2
    cfg.search.neb_max_steps = 8
    return cfg


def test_sweep_matrix_shape_and_outputs(tmp_path):
    cfg = tiny(tmp_path)
    elements = ["Pd", "Cu"]
    sweep = run_sweep(cfg, elements, log=lambda *a, **k: None)

    # rows = elements, cols = pathway states
    assert sweep.matrix.shape[0] == len(elements)
    assert sweep.matrix.shape[1] == len(sweep.col_labels)
    assert sweep.row_labels == ["NO@Pd", "NO@Cu"]
    # each row is referenced to its own first state -> first column is 0
    assert np.allclose(sweep.matrix[:, 0], 0.0)

    outdir = write_sweep(cfg, sweep, log=lambda *a, **k: None)
    for f in ["energy_map.png", "energy_map.csv", "sweep.json"]:
        assert (outdir / f).exists(), f
