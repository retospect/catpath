"""Layered validation: RDKit (molecules only), geometry sanity, and similarity.

Per the spec, RDKit understands discrete molecules, *not* periodic surfaces or
adsorbate ``*`` notation - so surface-adsorbed states are judged by geometry and
by cross-seed/model stability instead.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    adsorbate_height: float  # min distance of any adsorbate atom to the slab
    reasons: list[str]


def geometry_ok(
    atoms: Atoms,
    n_slab: int,
    min_bond: float = 0.7,
    max_bond: float = 3.0,
    max_ads_height: float = 3.5,
) -> GeometryReport:
    """Flag clashes and detached/floating adsorbates.

    * adsorbate-adsorbate atoms closer than ``min_bond`` -> clash
    * every adsorbate atom must sit within ``max_ads_height`` of some slab atom
      (otherwise it desorbed/floated away)
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
        min_dist = 0.0

    slab_pos = pos[:n_slab]
    worst_height = 0.0
    for a in ads:
        d = float(np.linalg.norm(slab_pos - pos[a], axis=1).min())
        worst_height = max(worst_height, d)
        if d > max_ads_height:
            reasons.append(f"adsorbate atom {a} detached from slab ({d:.2f} A)")

    return GeometryReport(
        ok=not reasons, min_dist=float(min_dist),
        adsorbate_height=worst_height, reasons=reasons,
    )


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
