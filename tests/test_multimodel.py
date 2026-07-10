"""Multi-model x multi-seed aggregation plumbing (EMT)."""

from atosim.config import Config, MLIPConfig, SlabConfig
from atosim import pipeline, provenance


def test_specs_single_and_multi():
    assert MLIPConfig(backend="mace", model="small").specs() == [("mace", "small")]
    s = MLIPConfig(backend="mace", models=["small", "medium"]).specs()
    assert s == [("mace", "small"), ("mace", "medium")]
    # explicit backend:model syntax
    s2 = MLIPConfig(models=["mace:small", "emt:"]).specs()
    assert s2 == [("mace", "small"), ("emt", None)]


def tiny(tmp_path):
    cfg = Config(name="mm", outdir=str(tmp_path))
    cfg.network = "oxidation"
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 12
    cfg.search.neb_images = 2
    cfg.search.neb_max_steps = 6
    return cfg


def test_multimodel_pools_model_times_seed(tmp_path):
    cfg = tiny(tmp_path)
    cfg.mlip.models = ["a", "b"]  # backend emt ignores the model name
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    # 2 models x 2 seeds = 4 samples per level
    for est in res.node_energies.values():
        assert est.n == 4
    assert res.models == ["emt:a", "emt:b"]
    # substrate reference -> root level is exactly 0
    assert abs(res.node_energies[res.pathway[0]].mean) < 1e-9


def test_provenance_caption_and_methods(tmp_path):
    cfg = tiny(tmp_path)
    cfg.mlip.models = ["a", "b"]
    res = pipeline.run(cfg, log=lambda *a, **k: None)
    cap = provenance.caption(cfg, res)
    assert "2 model(s) x 2 seed(s) = 4 samples" in cap
    md = provenance.methods_text(cfg, res)
    assert "Methods" in md and "NEB" in md
    pm = provenance.png_metadata(cfg, res)
    assert "Software" in pm and "Description" in pm
