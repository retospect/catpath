"""Cross-model state comparison: energy-interval column packing + plotting,
and the states-only relax path (EMT)."""

import json

from pytest import approx

from atosim import pipeline
from atosim.config import Config, MLIPConfig, SlabConfig
from atosim.viz import _anchor_shift, _assign_columns, _ordered_names, compare_boxplot


def test_interval_packing_shifts_only_on_overlap():
    # three mutually overlapping spans -> three columns
    cols, n = _assign_columns([(0.0, 1.0), (0.5, 1.5), (0.8, 1.8)], pad=0.0)
    assert n == 3 and len(set(cols)) == 3
    # three disjoint spans -> one shared column
    cols, n = _assign_columns([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], pad=0.0)
    assert n == 1 and set(cols) == {0}
    # a column frees up and is reused (min columns = max overlap depth = 2)
    cols, n = _assign_columns([(0.0, 1.0), (0.5, 1.5), (2.0, 3.0)], pad=0.0)
    assert n == 2


def test_pad_forces_a_shift():
    # spans just touch: no overlap at pad 0, but padding pushes them apart
    assert _assign_columns([(0.0, 1.0), (1.0, 2.0)], pad=0.0)[1] == 1
    assert _assign_columns([(0.0, 1.0), (1.0, 2.0)], pad=0.2)[1] == 2


def test_anchor_shift_pins_anchor_to_zero_per_model():
    runs = [
        {"model": "a", "states": {"NO": [-0.6, -0.6], "NH3": [-2.7]}},
        {"model": "b", "states": {"NO": [-1.8], "NH3": [-1.9]}},
    ]
    out = _anchor_shift(runs, "NO")
    assert out[0]["states"]["NO"] == [0.0, 0.0]
    assert out[1]["states"]["NO"] == [0.0]
    # the shift is a per-model constant -> reaction energy NO->NH3 is preserved
    assert out[0]["states"]["NH3"][0] == approx(-2.1)  # -2.7 - (-0.6)
    assert out[1]["states"]["NH3"][0] == approx(-0.1)  # -1.9 - (-1.8)


def test_ordered_names_follows_reaction_order():
    runs = [{"model": "m", "order": ["NO", "N+O", "NH3"],
             "states": {"NH3": [0], "NO": [0], "N+O": [0]}}]
    assert _ordered_names(runs) == ["NO", "N+O", "NH3"]  # substrate first


def test_compare_boxplot_writes_png(tmp_path):
    runs = [
        {"model": "mace", "states": {"NO": [0.0, 0.01], "N+O": [-0.2, -0.18]}},
        {"model": "chgnet", "states": {"NO": [0.0, -0.01], "N+O": [-0.15, -0.16]}},
        {"model": "fairchem", "states": {"NO": [0.0], "N+O": [-0.22]}},
    ]
    out = tmp_path / "cmp.png"
    compare_boxplot(runs, out, title="t")
    assert out.exists() and out.stat().st_size > 0


def _tiny_states_cfg():
    cfg = Config(name="s", substrate="NO", target="NO3", network="oxidation")
    cfg.slab = SlabConfig(size=(2, 2, 3), vacuum=8.0, fix_layers=1)
    cfg.mlip = MLIPConfig(backend="emt")
    cfg.search.seeds = [0, 1]
    cfg.search.max_steps = 15
    return cfg


def test_run_states_substrate_reference_zeros_root():
    cfg = _tiny_states_cfg()
    data = pipeline.run_states(cfg, log=lambda *a, **k: None, reference="substrate")
    assert data["model"] == "emt" and data["reference"] == "substrate"
    root = data["order"][0]
    assert all(v == 0.0 for v in data["states"][root])  # root references to zero
    assert all(len(v) == 2 for v in data["states"].values())  # one sample per seed
    json.dumps(data)  # must be JSON-serialisable (written to disk)


def test_run_states_formation_reference_is_composition_aware():
    cfg = _tiny_states_cfg()
    data = pipeline.run_states(cfg, log=lambda *a, **k: None, reference="formation")
    assert data["reference"] == "formation"
    # NO2 has one more O than NO -> a formation energy, so the root is NOT zero
    root = data["order"][0]
    assert not all(v == 0.0 for v in data["states"][root])
    assert all(len(v) == 2 for v in data["states"].values())
