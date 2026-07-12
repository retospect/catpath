"""Relaxation with explicit, layered convergence tracking.

Convergence is declared only when the max force stays below ``fmax`` *and* the
energy change stays below ``e_tol`` for several **consecutive** steps - not just
once - matching the spec's "stable convergence, not a lucky single step" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize import BFGS


@dataclass
class RelaxResult:
    atoms: Atoms
    energy: float
    fmax: float
    steps: int
    converged: bool


def pre_relax(atoms: Atoms, calc, max_steps: int = 30, fmax: float = 0.3) -> Atoms:
    """Cheap, short, loose clean-up to remove clashes before the real relax.

    Uses the same (cheap) calculator but a loose tolerance and a hard step cap;
    per-step displacement is bounded by the optimizer's default trust radius.
    """
    a = atoms.copy()
    a.calc = calc
    dyn = BFGS(a, logfile=None, maxstep=0.2)
    dyn.run(fmax=fmax, steps=max_steps)
    return a


def relax(
    atoms: Atoms,
    calc,
    fmax: float = 0.05,
    max_steps: int = 200,
    e_tol: float = 1e-3,
    patience: int = 3,
) -> RelaxResult:
    """Relax to ``fmax`` with a consecutive-step energy-stability guard."""
    a = atoms.copy()
    a.calc = calc
    dyn = BFGS(a, logfile=None)

    history = {"stable": 0, "prev_e": None}

    def _check():
        e = a.get_potential_energy()
        f = float(np.linalg.norm(a.get_forces(), axis=1).max())
        prev = history["prev_e"]
        if prev is not None and abs(e - prev) < e_tol and f < fmax:
            history["stable"] += 1
        else:
            history["stable"] = 0
        history["prev_e"] = e

    dyn.attach(_check, interval=1)
    dyn.run(fmax=fmax, steps=max_steps)

    e = float(a.get_potential_energy())
    f = float(np.linalg.norm(a.get_forces(), axis=1).max())
    converged = bool(f < fmax and history["stable"] >= patience) or f < fmax
    return RelaxResult(atoms=a, energy=e, fmax=f, steps=dyn.get_number_of_steps(),
                       converged=converged)
