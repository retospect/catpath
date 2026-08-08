"""Sweep the same reaction network across several surfaces/substrates.

Produces the multi-row **substrate x intermediate energy map**: one row per
surface element, columns are the shared pathway states, each row referenced to
its own starting state, and the highest-energy (rate-limiting) state starred.

This is catalyst screening - e.g. NO -> NO3 on Pd vs Pt vs Cu vs Ni - and works
with the EMT backend today (Pd, Pt, Cu, Ni, Ag, Au, Al all supported).
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
class SweepResult:
    matrix: np.ndarray                 # rows x cols of relative energies (eV)
    row_labels: list[str]
    col_labels: list[str]
    per_element: dict[str, Results] = field(default_factory=dict)


def run_sweep(base: Config, elements: list[str], log=print) -> SweepResult:
    rows: list[list[float]] = []
    labels: list[str] = []
    per_element: dict[str, Results] = {}
    cols: list[str] | None = None

    for el in elements:
        cfg = copy.deepcopy(base)
        cfg.slab.element = el
        cfg.name = f"{base.name}_{el}"
        log(f"=== sweep: {base.substrate} -> {base.target} on {el} ===")
        res = run(cfg, log=log)
        write_outputs(cfg, res, log=log)  # per-element artifacts too

        if cols is None:
            cols = res.pathway
        ref = res.node_energies[res.pathway[0]].mean
        rows.append([res.node_energies[c].mean - ref for c in cols])
        labels.append(f"{base.substrate}@{el}")
        per_element[el] = res

    return SweepResult(np.array(rows), labels, cols or [], per_element)


def write_sweep(base: Config, sweep: SweepResult, log=print) -> Path:
    outdir = Path(base.outdir) / f"{base.name}_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    energy_map(
        sweep.matrix, sweep.row_labels, sweep.col_labels,
        outdir / "energy_map.png",
        title=f"{base.substrate} -> {base.target}: catalyst screen",
    )
    with open(outdir / "energy_map.csv", "w") as f:
        f.write("row," + ",".join(sweep.col_labels) + "\n")
        for label, row in zip(sweep.row_labels, sweep.matrix):
            f.write(label + "," + ",".join(f"{v:.4f}" for v in row) + "\n")

    # per-row rate-limiting (starred) state, for quick machine-readable ranking
    summary = {"substrate": base.substrate, "target": base.target,
               "columns": sweep.col_labels, "rows": []}
    for label, row in zip(sweep.row_labels, sweep.matrix):
        peak = int(np.nanargmax(row))
        summary["rows"].append({
            "row": label,
            "rate_limiting_state": sweep.col_labels[peak],
            "peak_rel_energy_eV": float(row[peak]),
            "energies_eV": {c: float(v) for c, v in zip(sweep.col_labels, row)},
        })
    (outdir / "sweep.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote sweep to {outdir}")
    return outdir
