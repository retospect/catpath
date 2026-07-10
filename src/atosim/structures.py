"""Build metal slabs and place adsorbates at chemistry-informed sites.

The key invariant for barrier searches: two states connected by a reaction must
have the **same atoms in the same order** so NEB can interpolate between them.
We enforce this by building every state as ``slab + adsorbate_atoms`` where the
adsorbate atoms are appended in a caller-controlled, fixed order.
"""

from __future__ import annotations

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, fcc111
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, reference_states
from ase.eos import EquationOfState

from .config import SlabConfig


def default_lattice(element: str) -> float:
    """ASE reference (experimental) fcc lattice constant, in A."""
    return float(reference_states[atomic_numbers[element]]["a"])


def equilibrium_lattice(element: str, make_calc, a_guess: float | None = None,
                        strain: float = 0.04, n: int = 7) -> float:
    """Fit the potential's equilibrium fcc lattice constant via an EOS scan.

    Removes the built-in epitaxial strain that arises from building a slab at a
    lattice constant the ML/EMT potential does not actually prefer.
    """
    a0 = a_guess or default_lattice(element)
    vols, ens = [], []
    for s in np.linspace(1 - strain, 1 + strain, n):
        at = bulk(element, "fcc", a=a0 * s)  # primitive cell, 1 atom
        at.calc = make_calc()
        vols.append(at.get_volume())
        ens.append(at.get_potential_energy())
    v0 = EquationOfState(vols, ens).fit()[0]
    return float((4 * v0) ** (1 / 3))  # primitive volume = a^3 / 4 for fcc

# High-symmetry adsorption sites on fcc(111), as fractional offsets we resolve
# against ASE's computed site coordinates.
SITE_NAMES = ("ontop", "bridge", "fcc", "hcp")


def build_slab(cfg: SlabConfig) -> Atoms:
    """Return an fcc(111) slab with the bottom ``fix_layers`` layers frozen.

    Uses ``cfg.a`` as the lattice constant when set (e.g. the potential's relaxed
    value); otherwise ASE's default reference constant.
    """
    slab = fcc111(cfg.element, size=cfg.size, vacuum=cfg.vacuum, a=cfg.a)
    slab.pbc = (True, True, True)
    n_per_layer = cfg.size[0] * cfg.size[1]
    z = slab.positions[:, 2]
    order = np.argsort(z)
    fixed = set(order[: cfg.fix_layers * n_per_layer].tolist())
    slab.set_constraint(FixAtoms(indices=sorted(fixed)))
    slab.info["n_slab"] = len(slab)
    return slab


def site_xy(slab: Atoms, site: str) -> np.ndarray:
    """xy coordinates of a named high-symmetry site on the slab."""
    sites = slab.info.get("adsorbate_info", {}).get("sites")
    if sites and site in sites:
        offset = np.asarray(sites[site])
        cell = slab.info["adsorbate_info"]["cell"]
        xy = offset @ cell
        return np.asarray(xy)
    # Fallback: derive from the top-layer atom positions.
    top_z = slab.positions[:, 2].max()
    top = slab.positions[np.abs(slab.positions[:, 2] - top_z) < 0.1]
    a = top[np.lexsort((top[:, 0], top[:, 1]))][0][:2]
    if site == "ontop":
        return a
    # nearest neighbour along +x for bridge/hollow approximations
    nbrs = top[np.argsort(np.linalg.norm(top[:, :2] - a, axis=1))][1:4, :2]
    if site == "bridge":
        return (a + nbrs[0]) / 2
    return (a + nbrs[0] + nbrs[1]) / 3  # fcc/hcp ~ 3-fold hollow centroid


def top_z(slab: Atoms) -> float:
    return float(slab.positions[:, 2].max())


def add_adsorbate_atoms(
    slab: Atoms,
    symbols: list[str],
    site: str = "fcc",
    height: float = 2.0,
    bond: float = 1.2,
    tilt: float = 0.0,
) -> Atoms:
    """Append a small adsorbate (given as a vertical chain of ``symbols``).

    The first symbol sits closest to the surface (the anchor atom).  Extra atoms
    stack above it separated by ``bond``.  ``tilt`` (radians) rotates the chain
    off the surface normal so multi-atom adsorbates are not perfectly vertical.
    """
    atoms = slab.copy()
    atoms.set_constraint(slab.constraints)
    xy = site_xy(slab, site)
    base_z = top_z(slab) + height
    for i, sym in enumerate(symbols):
        dz = i * bond * np.cos(tilt)
        dx = i * bond * np.sin(tilt)
        atoms.append(Atom(sym, position=(xy[0] + dx, xy[1], base_z + dz)))
    return atoms


def place_fragments(slab: Atoms, specs: list[dict]) -> Atoms:
    """Append adsorbate atoms from explicit specs, preserving order.

    Each spec: ``{"symbol": "O", "site": "fcc", "height": 2.0,
    "dx": 0.0, "dy": 0.0}``.  ``dx/dy`` offset the atom from the named site so
    multiple fragments at the same site do not overlap.  Order of ``specs`` is
    the order of appended atoms - callers rely on this for NEB endpoint matching.
    """
    atoms = slab.copy()
    atoms.set_constraint(slab.constraints)
    z0 = top_z(slab)
    for s in specs:
        xy = site_xy(slab, s.get("site", "fcc"))
        x = xy[0] + s.get("dx", 0.0)
        y = xy[1] + s.get("dy", 0.0)
        z = z0 + s.get("height", 2.0)
        atoms.append(Atom(s["symbol"], position=(x, y, z)))
    return atoms


def poses(
    slab: Atoms,
    symbols: list[str],
    count: int,
    seed: int = 0,
    height: float = 2.0,
) -> list[Atoms]:
    """Generate an ensemble of plausible starting poses over sites/tilts.

    Deterministic given ``seed`` (reproducibility is an acceptance criterion).
    """
    rng = np.random.default_rng(seed)
    out = []
    sites = list(SITE_NAMES)
    for k in range(count):
        site = sites[k % len(sites)]
        tilt = float(rng.uniform(0.0, 0.6)) if len(symbols) > 1 else 0.0
        h = height + float(rng.uniform(-0.2, 0.2))
        out.append(add_adsorbate_atoms(slab, symbols, site=site, height=h, tilt=tilt))
    return out


def rattle_adsorbate(atoms: Atoms, n_slab: int, seed: int, amplitude: float = 0.15) -> Atoms:
    """Return a copy with only the adsorbate atoms randomly displaced (seeded).

    Used to build a per-seed pose ensemble that probes robustness of the local
    minimum without moving the (constrained) slab.
    """
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    a.set_constraint(atoms.constraints)
    pos = a.get_positions()
    n_ads = len(atoms) - n_slab
    if n_ads > 0:
        pos[n_slab:] += rng.normal(0.0, amplitude, size=(n_ads, 3))
        a.set_positions(pos)
    return a


def symbols_of(atoms: Atoms) -> set[str]:
    return set(atoms.get_chemical_symbols())
