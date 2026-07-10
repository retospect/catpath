"""Run several substrates (each its own network) in one job.

Where :mod:`atosim.sweep` varies the *surface* (same network across metals),
this varies the *molecule*: each entry is a distinct ``(substrate, target,
network, reagents)`` and gets its own full run and per-run artifacts.  The rows
are then stacked into one **substrate x intermediate energy map** whose columns
are the *union* of every run's pathway states -- cells are ``NaN`` where a
substrate never visits that state, and the ``*`` (rate-limiting) marker already
tolerates the gaps.

Because different networks have different states, the shared column axis is a
union rather than an intersection; this keeps every run's rate-limiting state
visible on one comparable grid.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import Results, run, write_outputs
from .viz import energy_map


@dataclass
class MultiResult:
    matrix: np.ndarray                 # rows x union-cols of relative energies (eV)
    row_labels: list[str]
    col_labels: list[str]
    per_run: dict[str, Results] = field(default_factory=dict)


def run_multi(base: Config, log=print) -> MultiResult:
    specs = base.substrate_runs()
    per_run: dict[str, Results] = {}
    row_energy: list[dict[str, float]] = []
    labels: list[str] = []
    cols: list[str] = []           # union, preserving first-seen order
    seen: set[str] = set()

    for spec in specs:
        cfg = copy.deepcopy(base)
        cfg.substrate = spec.substrate
        cfg.target = spec.target
        cfg.network = spec.network
        cfg.reagents = spec.reagents
        cfg.substrates = [spec.substrate]      # single run; no recursion
        cfg.name = spec.name or f"{base.name}_{spec.substrate}_{spec.network}"
        log(f"=== multi: {spec.substrate} -> {spec.target} "
            f"({spec.network}, reagents={spec.reagents}) ===")
        res = run(cfg, log=log)
        write_outputs(cfg, res, log=log)       # per-substrate artifacts too

        ref = res.node_energies[res.pathway[0]].mean
        row_energy.append({c: res.node_energies[c].mean - ref for c in res.pathway})
        labels.append(spec.name or f"{spec.substrate}->{spec.target}")
        for c in res.pathway:
            if c not in seen:
                seen.add(c)
                cols.append(c)
        per_run[cfg.name] = res

    matrix = np.full((len(row_energy), len(cols)), np.nan)
    for i, energies in enumerate(row_energy):
        for j, c in enumerate(cols):
            if c in energies:
                matrix[i, j] = energies[c]
    return MultiResult(matrix, labels, cols, per_run)


def write_multi(base: Config, multi: MultiResult, log=print) -> Path:
    outdir = Path(base.outdir) / f"{base.name}_multi"
    outdir.mkdir(parents=True, exist_ok=True)

    energy_map(
        multi.matrix, multi.row_labels, multi.col_labels,
        outdir / "energy_map.png",
        title=f"{base.name}: multi-substrate energy map",
    )
    with open(outdir / "energy_map.csv", "w") as f:
        f.write("row," + ",".join(multi.col_labels) + "\n")
        for label, row in zip(multi.row_labels, multi.matrix):
            cells = ["" if np.isnan(v) else f"{v:.4f}" for v in row]
            f.write(label + "," + ",".join(cells) + "\n")

    summary = {"name": base.name, "columns": multi.col_labels, "rows": []}
    for label, row in zip(multi.row_labels, multi.matrix):
        if np.all(np.isnan(row)):
            continue
        peak = int(np.nanargmax(row))
        summary["rows"].append({
            "row": label,
            "rate_limiting_state": multi.col_labels[peak],
            "peak_rel_energy_eV": float(row[peak]),
            "energies_eV": {c: (None if np.isnan(v) else float(v))
                            for c, v in zip(multi.col_labels, row)},
        })
    (outdir / "multi.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote multi-substrate outputs to {outdir}")
    return outdir
