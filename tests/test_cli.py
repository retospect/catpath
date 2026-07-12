"""CLI: trivial no-file input and hint-rich errors (for humans and LLMs)."""

from argparse import Namespace

import pytest

from catpath.cli import _load


def _ns(**kw):
    base = dict(config=None, substrate=None, target=None, element=None, network=None,
                backend=None, device=None, models=None, seeds=None, reagents=None,
                name=None, outdir=None)
    base.update(kw)
    return Namespace(**base)


def test_chemistry_flags_need_no_config_file():
    c = _load(_ns(substrate="CO", target="CH4", element="Ni", network="auto", seeds="0,1"))
    assert c.substrate == "CO" and c.target == "CH4"
    assert c.slab.element == "Ni" and c.network == "auto"
    assert c.substrates == ["CO"] and c.search.seeds == [0, 1]


def test_missing_config_gives_actionable_hint():
    with pytest.raises(SystemExit) as e:
        _load(_ns(config="does-not-exist.yaml"))
    msg = str(e.value)
    assert "not found" in msg and "--substrate" in msg   # points at the no-file path


def test_unknown_network_lists_choices():
    with pytest.raises(SystemExit) as e:
        _load(_ns(network="bogus"))
    assert "ammonia" in str(e.value) and "auto" in str(e.value)


def test_bad_seeds_gives_hint():
    with pytest.raises(SystemExit) as e:
        _load(_ns(seeds="a,b"))
    assert "integers" in str(e.value)


def test_malformed_yaml_gives_quote_hint(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("substrate: [unclosed\n")
    with pytest.raises(SystemExit) as e:
        _load(_ns(config=str(f)))
    assert "YAML" in str(e.value)
