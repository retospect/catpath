from pathlib import Path

from atosim.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.slab.element == "Pd"
    assert cfg.substrates == ["NO"]  # defaults to [substrate]
    assert len(cfg.search.seeds) >= 3


def test_from_dict_normalises_size():
    cfg = Config.from_dict({"slab": {"size": [2, 2, 3]}})
    assert cfg.slab.size == (2, 2, 3)


def test_snapshot_roundtrip(tmp_path: Path):
    cfg = Config(name="x", substrate="NO", target="NO3")
    p = tmp_path / "snap.yaml"
    cfg.snapshot(p)
    back = Config.from_yaml(p)
    assert back.name == "x"
    assert back.target == "NO3"
    assert back.slab.element == cfg.slab.element


def test_auto_block_parses():
    cfg = Config.from_dict({"network": "auto",
                            "auto": {"max_extra": 3, "prune_energy": 0.2}})
    assert cfg.auto.max_extra == 3
    assert cfg.auto.prune_energy == 0.2
    assert cfg.auto.max_states == 600      # untouched default
    assert Config().auto.prune_energy is None  # off by default
