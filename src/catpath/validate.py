"""Layered validation: RDKit (molecules only), geometry sanity, and similarity.

Per the spec, RDKit understands discrete molecules, *not* periodic surfaces or
adsorbate ``*`` notation - so surface-adsorbed states are judged by geometry and
by cross-seed/model stability instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

# --- RDKit layer (molecule-like inputs/intermediates only) -------------------

def sanitize_smiles(smiles: str) -> bool:
    """True if RDKit can parse and sanitize the molecule (valence, etc.)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return False
    return True


# --- Geometry sanity layer (works for surface + molecule) --------------------

@dataclass
class GeometryReport:
    ok: bool
    min_dist: float          # smallest interatomic distance among adsorbate atoms
    adsorbate_height: float  # WORST (max) adsorbate-atom distance to nearest slab atom
    reasons: list[str]
    anchor_height: float = 0.0        # BEST (min) adsorbate-atom distance to slab
    atom_heights: list[float] = field(default_factory=list)  # per-adsorbate nearest-slab dist
    n_fragments: int = 0              # connected adsorbate fragments
    detached_fragments: int = 0       # fragments whose closest approach exceeds max_ads_height


def _fragments(pos: np.ndarray, ads: list[int], bond_cutoff: float) -> list[list[int]]:
    """Group adsorbate atoms into connected fragments by interatomic bond cutoff.

    A fragment is a set of adsorbate atoms transitively within ``bond_cutoff`` of
    one another — i.e. one chemically bonded molecule/radical. Two adatoms left by
    a dissociation (N* + O*, N* + H*) sit farther apart than a bond, so they land
    in separate fragments and are judged for binding independently.
    """
    parent = {a: a for a in ads}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            if float(np.linalg.norm(pos[ads[i]] - pos[ads[j]])) < bond_cutoff:
                parent[find(ads[i])] = find(ads[j])

    groups: dict[int, list[int]] = {}
    for a in ads:
        groups.setdefault(find(a), []).append(a)
    return list(groups.values())


def geometry_ok(
    atoms: Atoms,
    n_slab: int,
    min_bond: float = 0.7,
    max_bond: float = 3.0,
    max_ads_height: float = 3.5,
    bond_cutoff: float = 1.8,
) -> GeometryReport:
    """Flag clashes and detached/floating adsorbate FRAGMENTS.

    * adsorbate-adsorbate atoms closer than ``min_bond`` -> clash
    * binding is judged **per connected fragment** (atoms within ``bond_cutoff``
      of each other): a fragment counts as desorbed only when its *closest*
      approach to the slab exceeds ``max_ads_height``.

    Judging per fragment (not per atom) is what lets a chemisorbed radical or
    molecule anchored through one heavy atom — N*, O* — with its H's pointing up
    into vacuum count as BOUND: the up-pointing H sits >3.5 A from any metal, but
    the fragment's N/O anchor does not, so the H no longer falsely reads as
    desorption. A fully floated molecule (every atom far) or a dissociated adatom
    that drifted off is still flagged, since each fragment must independently
    reach the surface.
    """
    reasons: list[str] = []
    ads = list(range(n_slab, len(atoms)))
    pos = atoms.get_positions()

    min_dist = np.inf
    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            d = float(np.linalg.norm(pos[ads[i]] - pos[ads[j]]))
            min_dist = min(min_dist, d)
            if d < min_bond:
                reasons.append(f"adsorbate atoms {ads[i]},{ads[j]} too close ({d:.2f} A)")
    if not ads:
        return GeometryReport(ok=not reasons, min_dist=0.0, adsorbate_height=0.0,
                              reasons=reasons)

    slab_pos = pos[:n_slab]
    atom_h = {a: float(np.linalg.norm(slab_pos - pos[a], axis=1).min()) for a in ads}
    heights = [atom_h[a] for a in ads]

    frags = _fragments(pos, ads, bond_cutoff)
    detached = 0
    for frag in frags:
        frag_anchor = min(atom_h[a] for a in frag)
        if frag_anchor > max_ads_height:
            detached += 1
            syms = "".join(atoms[a].symbol for a in frag)
            # keep the literal "detached" — the precis trust-gate counts it.
            reasons.append(
                f"adsorbate fragment {syms} (atoms {min(frag)}-{max(frag)}) "
                f"detached from slab (closest {frag_anchor:.2f} A)"
            )

    return GeometryReport(
        ok=not reasons, min_dist=float(min_dist),
        adsorbate_height=max(heights), reasons=reasons,
        anchor_height=min(heights), atom_heights=heights,
        n_fragments=len(frags), detached_fragments=detached,
    )


def binding_site_ok(
    atoms: Atoms,
    n_slab: int,
    placement_heights: list[float],
    bond_cutoff: float = 1.8,
) -> tuple[bool, list[str]]:
    """Check that each fragment binds the surface through the ``*``-designated atom.

    ``placement_heights[i]`` is the intended height of adsorbate atom ``n_slab+i``
    (from its :class:`~catpath.network.StateSpec` spec); the LOWEST-placed atom in
    a fragment is the one the formula's ``*`` seats at the site. After relaxation
    the atom actually closest to the slab in each fragment must be the SAME ELEMENT
    as that intended anchor — comparing by element (not index) so a swap between
    two equivalent atoms (e.g. the two O's of NO2*) is fine, while a genuine flip
    (NO* rolling onto its O instead of N) is flagged.

    Returns ``(ok, reasons)``; each reason carries the literal ``wrong-site`` so
    the precis trust-gate counts it (a barrier off a mis-bound endpoint is as
    untrustworthy as one off a desorbed one).
    """
    ads = list(range(n_slab, len(atoms)))
    if not ads or len(placement_heights) != len(ads):
        return True, []  # nothing to check / caller gave no intent
    pos = atoms.get_positions()
    slab_pos = pos[:n_slab]
    atom_h = {a: float(np.linalg.norm(slab_pos - pos[a], axis=1).min()) for a in ads}
    height_of = {a: placement_heights[a - n_slab] for a in ads}

    reasons: list[str] = []
    for frag in _fragments(pos, ads, bond_cutoff):
        intended = min(frag, key=lambda a: height_of[a])   # lowest-placed = the *
        actual = min(frag, key=lambda a: atom_h[a])         # relaxed nearest-slab
        want, got = atoms[intended].symbol, atoms[actual].symbol
        if want != got:
            syms = "".join(atoms[a].symbol for a in frag)
            reasons.append(
                f"fragment {syms} (atoms {min(frag)}-{max(frag)}) binds through "
                f"{got} but the * designates {want} — wrong-site")
    return (not reasons), reasons


# --- Similarity layer --------------------------------------------------------

def rmsd(a: Atoms, b: Atoms, n_slab: int) -> float:
    """Kabsch-aligned RMSD over the adsorbate atoms of two states."""
    if len(a) != len(b):
        raise ValueError("RMSD needs matching atom counts")
    pa = a.get_positions()[n_slab:]
    pb = b.get_positions()[n_slab:]
    if len(pa) == 0:
        return 0.0
    pa = pa - pa.mean(0)
    pb = pb - pb.mean(0)
    h = pa.T @ pb
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    aligned = pa @ r.T
    return float(np.sqrt(((aligned - pb) ** 2).sum(1).mean()))


def is_similar(a: Atoms, b: Atoms, n_slab: int, rmsd_thresh: float) -> bool:
    return rmsd(a, b, n_slab) <= rmsd_thresh
