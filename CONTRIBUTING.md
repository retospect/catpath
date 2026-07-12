# Contributing to catpath

Thanks for your interest! catpath is a small project — currently maintained by
**Reto Stamm** — so the process is light. Issues and pull requests are welcome.

## Development setup

catpath uses [`uv`](https://docs.astral.sh/uv/). The default **`emt`** backend is
pure numpy/ASE (no torch/GPU), so the whole test suite runs anywhere:

```bash
git clone https://github.com/retospect/catpath
cd catpath
uv sync --extra dev
```

## Before you open a PR

Both of these must pass — they're exactly what CI runs:

```bash
uv run ruff check src tests    # lint (the project is NOT ruff-formatted; match the surrounding hand style)
uv run pytest                  # full suite, on the EMT backend
```

- **Tests run on EMT.** Keep new tests calculator-light: use `emt` (or
  monkeypatch), tiny slabs, and few seeds so CI stays fast and torch-free.
- **ML backends are optional and mutually exclusive.** `mace`, `chgnet`,
  `fairchem`, and `grace` have conflicting dependencies — install **one per
  environment** (`uv sync --extra mace`), and don't add code paths that import
  more than one at once. New backends go in `calculators.py` behind a lazy import.
- **Style.** Match the existing hand-formatting (aligned inline comments, ~100
  col). `ruff check` is enforced; `ruff format` is intentionally *not* applied.
- Keep changes focused; add a test for new behavior; update `docs/` and
  `examples/` when you change the config schema or CLI.

## Reporting issues

Please include: the config YAML (or a minimal repro), the backend and device,
the command you ran, and the full error / warning output. `runs/<name>/methods.md`
and `config.snapshot.yaml` capture most of what's needed.

## Releases

Tag `vX.Y.Z` and cut a GitHub Release; `.github/workflows/workflow.yml` publishes
to PyPI via OIDC trusted publishing (no tokens).

## License

By contributing you agree that your contributions are licensed under
**GPL-3.0-or-later**, the project's license.

## Acknowledgements

catpath was requested by **Muhammad Umer**, whose help shaping what it should do
got it off the ground — thank you.

Built with **Claude Code** (Anthropic), with research assistance from
**Perplexity**.
