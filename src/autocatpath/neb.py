"""Transition-state / barrier search via climbing-image NEB.

A simple bracketed maximum along a linear interpolation is provided as a cheap
fallback (``linear_barrier``) for when NEB is overkill or fails to converge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.constraints import Hookean
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
    desorbed_images: int = 0  # interior band images whose adsorbate desorbed


def _energies(images: list[Atoms]) -> list[float]:
    return [float(im.get_potential_energy()) for im in images]


def _tether_image(im: Atoms, n_slab: int, k: float, rt: float) -> list:
    """One Hookean per adsorbate atom of ``im``, anchored to its nearest slab atom
    (mirrors ``pipeline._tether_constraints`` but local to keep NEB import-clean)."""
    pos = im.get_positions()
    slab_pos = pos[:n_slab]
    cons = []
    for i in range(n_slab, len(im)):
        j = int(np.argmin(np.linalg.norm(slab_pos - pos[i], axis=1)))
        cons.append(Hookean(a1=i, a2=j, k=k, rt=rt))
    return cons


def _make_band(reactant: Atoms, product: Atoms, n_images: int, calc) -> list[Atoms]:
    images = [reactant.copy()]
    images += [reactant.copy() for _ in range(n_images)]
    images += [product.copy()]
    # One calculator shared across the whole band. A serial NEB evaluates images
    # sequentially, and an ASE calculator is stateless across the Atoms it scores
    # (it recomputes on every call), so sharing is correct — and it avoids
    # rebuilding the ML potential (a full torch.load, hundreds of MB to the GPU)
    # once per image, which otherwise dominates wall time.
    for im in images:
        im.calc = calc
    neb = NEB(images, climb=False, method="improvedtangent",
              allow_shared_calculator=True)
    neb.interpolate(method="idpp")
    return images


def _neb_attempt(
    reactant: Atoms, product: Atoms, make_calc,
    n_images: int, fmax: float, max_steps: int, climb: bool,
    tether: dict | None = None, n_slab: int | None = None,
) -> tuple[list[float], bool, list[Atoms]]:
    """One NEB run -> (image energies, converged, images).

    With ``tether`` (``{"k","rt","ramp"}``) and ``n_slab``, the band is first
    settled under DISSOLVING Hookean tethers on the adsorbate atoms — k scaled
    down the ``ramp`` — so no interior image desorbs while the band relaxes, then
    the tethers are dropped for the final (reported) climbing pass on the true,
    untethered PES. Without a tether it is the plain climbing-image NEB.
    """
    calc = make_calc()
    base_reac = [c for c in reactant.constraints if not isinstance(c, Hookean)]
    images = _make_band(reactant, product, n_images, calc)

    if tether and n_slab is not None:
        # adiabatic band settle: interior images pulled toward the surface, spring
        # dissolved to 0 so the final climb runs untethered (see class docstring).
        interior = images[1:-1]
        settle_steps = max(10, max_steps // 3)
        for frac in tether.get("ramp", [1.0, 0.5, 0.25, 0.1]):
            if frac <= 0.0:
                continue
            for im in interior:
                im.set_constraint(
                    list(base_reac)
                    + _tether_image(im, n_slab, tether["k"] * frac, tether["rt"]))
            neb = NEB(images, climb=False, method="improvedtangent",
                      allow_shared_calculator=True)
            BFGS(neb, logfile=None).run(fmax=max(fmax, 0.2), steps=settle_steps)
        for im in interior:  # drop every spring before the reported climb
            im.set_constraint(list(base_reac))

    neb = NEB(images, climb=climb, method="improvedtangent",
              allow_shared_calculator=True)
    dyn = BFGS(neb, logfile=None)
    converged = bool(dyn.run(fmax=fmax, steps=max_steps))
    return _energies(images), converged, images


def _count_desorbed(images: list[Atoms], n_slab: int) -> int:
    """Interior images (endpoints excluded — they are pre-relaxed & gated) whose
    adsorbate desorbed. A garbage-barrier guard: a TS or midpoint that floats off
    the slab means the reported barrier is off a non-surface path."""
    from .validate import geometry_ok

    n = 0
    for im in images[1:-1]:
        geo = geometry_ok(im, n_slab)
        if any("detached" in r for r in geo.reasons):
            n += 1
    return n


def neb_barrier(
    reactant: Atoms,
    product: Atoms,
    make_calc,
    n_images: int = 5,
    fmax: float = 0.1,
    max_steps: int = 100,
    climb: bool = True,
    retries: int = 0,
    n_slab: int | None = None,
    tether: dict | None = None,
) -> BarrierResult:
    """Climbing-image NEB between two *relaxed* endpoints with matching atoms.

    On non-convergence, retry up to ``retries`` more times, each with more images
    (~1.5x) and twice the optimisation budget -- a denser band with more steps
    is the usual cure for a NEB that ran out of steps or had too coarse a
    tangent. Returns the first converged attempt, else the last (most refined)
    one, so a barrier estimate is always reported.

    ``tether`` + ``n_slab`` enable a dissolving-tether band settle before the
    climb (mid-band desorption guard, see :func:`_neb_attempt`). When ``n_slab``
    is given, ``desorbed_images`` counts interior images that still desorbed on
    the final band — a signal the caller turns into an untrusted-barrier warning.
    """
    if len(reactant) != len(product):
        raise ValueError("NEB endpoints must have identical atom counts")

    ni, ms = n_images, max_steps
    es: list[float] = []
    images: list[Atoms] = []
    converged = False
    attempt = 0
    for attempt in range(1 + max(0, retries)):
        es, converged, images = _neb_attempt(
            reactant, product, make_calc, ni, fmax, ms, climb, tether, n_slab)
        if converged:
            break
        ni += max(2, ni // 2)  # ~1.5x denser band
        ms *= 2                # and a bigger optimisation budget

    e_r, e_p = es[0], es[-1]
    e_ts = max(es)
    method = "ci-neb" if attempt == 0 else f"ci-neb+retry{attempt}"
    desorbed = _count_desorbed(images, n_slab) if n_slab is not None else 0
    return BarrierResult(
        e_reactant=e_r, e_product=e_p, e_ts=e_ts,
        barrier=e_ts - e_r, delta_e=e_p - e_r,
        converged=converged, method=method, images_energy=es,
        desorbed_images=desorbed,
    )


def linear_barrier(
    reactant: Atoms, product: Atoms, make_calc, n_images: int = 11
) -> BarrierResult:
    """Cheap fallback: energies along a straight-line interpolation, take the max."""
    if len(reactant) != len(product):
        raise ValueError("endpoints must have identical atom counts")
    p0, p1 = reactant.positions, product.positions
    calc = make_calc()  # one calculator for every interpolation point (see _neb_attempt)
    es = []
    for t in np.linspace(0.0, 1.0, n_images):
        im = reactant.copy()
        im.positions = (1 - t) * p0 + t * p1
        im.calc = calc
        es.append(float(im.get_potential_energy()))
    e_r, e_p, e_ts = es[0], es[-1], max(es)
    return BarrierResult(
        e_reactant=e_r, e_product=e_p, e_ts=e_ts,
        barrier=e_ts - e_r, delta_e=e_p - e_r,
        converged=True, method="linear", images_energy=es,
    )
