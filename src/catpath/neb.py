"""Transition-state / barrier search via climbing-image NEB.

A simple bracketed maximum along a linear interpolation is provided as a cheap
fallback (``linear_barrier``) for when NEB is overkill or fails to converge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.mep import NEB
from ase.optimize import BFGS


@dataclass
class BarrierResult:
    e_reactant: float
    e_product: float
    e_ts: float
    barrier: float  # E(TS) - E(reactant), eV
    delta_e: float  # E(product) - E(reactant), eV
    converged: bool
    method: str
    images_energy: list[float]


def _energies(images: list[Atoms]) -> list[float]:
    return [float(im.get_potential_energy()) for im in images]


def _neb_attempt(
    reactant: Atoms, product: Atoms, make_calc,
    n_images: int, fmax: float, max_steps: int, climb: bool,
) -> tuple[list[float], bool]:
    """One climbing-image NEB run -> (image energies, converged)."""
    images = [reactant.copy()]
    images += [reactant.copy() for _ in range(n_images)]
    images += [product.copy()]
    for im in images:
        im.calc = make_calc()  # each image needs its own calculator instance

    neb = NEB(images, climb=climb, method="improvedtangent",
              allow_shared_calculator=False)
    neb.interpolate(method="idpp")
    dyn = BFGS(neb, logfile=None)
    converged = bool(dyn.run(fmax=fmax, steps=max_steps))
    return _energies(images), converged


def neb_barrier(
    reactant: Atoms,
    product: Atoms,
    make_calc,
    n_images: int = 5,
    fmax: float = 0.1,
    max_steps: int = 100,
    climb: bool = True,
    retries: int = 0,
) -> BarrierResult:
    """Climbing-image NEB between two *relaxed* endpoints with matching atoms.

    On non-convergence, retry up to ``retries`` more times, each with more images
    (~1.5x) and twice the optimisation budget -- a denser band with more steps
    is the usual cure for a NEB that ran out of steps or had too coarse a
    tangent. Returns the first converged attempt, else the last (most refined)
    one, so a barrier estimate is always reported.
    """
    if len(reactant) != len(product):
        raise ValueError("NEB endpoints must have identical atom counts")

    ni, ms = n_images, max_steps
    es: list[float] = []
    converged = False
    attempt = 0
    for attempt in range(1 + max(0, retries)):
        es, converged = _neb_attempt(reactant, product, make_calc, ni, fmax, ms, climb)
        if converged:
            break
        ni += max(2, ni // 2)  # ~1.5x denser band
        ms *= 2                # and a bigger optimisation budget

    e_r, e_p = es[0], es[-1]
    e_ts = max(es)
    method = "ci-neb" if attempt == 0 else f"ci-neb+retry{attempt}"
    return BarrierResult(
        e_reactant=e_r, e_product=e_p, e_ts=e_ts,
        barrier=e_ts - e_r, delta_e=e_p - e_r,
        converged=converged, method=method, images_energy=es,
    )


def linear_barrier(
    reactant: Atoms, product: Atoms, make_calc, n_images: int = 11
) -> BarrierResult:
    """Cheap fallback: energies along a straight-line interpolation, take the max."""
    if len(reactant) != len(product):
        raise ValueError("endpoints must have identical atom counts")
    p0, p1 = reactant.positions, product.positions
    es = []
    for t in np.linspace(0.0, 1.0, n_images):
        im = reactant.copy()
        im.positions = (1 - t) * p0 + t * p1
        im.calc = make_calc()
        es.append(float(im.get_potential_energy()))
    e_r, e_p, e_ts = es[0], es[-1], max(es)
    return BarrierResult(
        e_reactant=e_r, e_product=e_p, e_ts=e_ts,
        barrier=e_ts - e_r, delta_e=e_p - e_r,
        converged=True, method="linear", images_energy=es,
    )
