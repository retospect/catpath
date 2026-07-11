"""Fast end-to-end integration test on a tiny slab with the EMT backend."""

import numpy as np

from atosim.config import Config
from atosim.neb import linear_barrier
from atosim.calculators import make_calculator
from atosim.config import MLIPConfig, SlabConfig
from atosim.structures import build_slab, place_fragments
from atosim import pipeline


def tiny_cfg(tmp_path):
    cfg = Config(name="itest", outdir=str(tmp_path))
    cfg.network = "oxidation"  # fast linear chain for the integration test
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 20
    cfg.search.neb_images = 3
    cfg.search.neb_max_steps = 15
    return cfg


def test_linear_barrier_runs():
    slab = build_slab(SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1))
    r = place_fragments(slab, [{"symbol": "O", "site": "fcc", "height": 2.0}])
    p = place_fragments(slab, [{"symbol": "O", "site": "hcp", "height": 2.0, "dx": 1.0}])
    res = linear_barrier(r, p, make_calc=lambda: make_calculator(MLIPConfig()), n_images=5)
    assert res.barrier >= 0.0
    assert np.isfinite(res.e_ts)
    assert len(res.images_energy) == 5


def test_pipeline_end_to_end(tmp_path):
    cfg = tiny_cfg(tmp_path)
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    # four states, two edges
    assert set(res.node_energies) == {"NO+O", "NO2", "NO2+O", "NO3"}
    assert len(res.edges) == 2
    # two seeds -> real spread available (n=2)
    for est in res.node_energies.values():
        assert est.n == 2
    outdir = pipeline.write_outputs(cfg, res, log=lambda *a, **k: None)
    for f in ["results.json", "graph.json", "graph.png", "energy_map.png",
              "nodes.csv", "edges.csv", "config.snapshot.yaml"]:
        assert (outdir / f).exists(), f
