# Sample structures

Provenance of the bundled sample files (fetched from public databases).

| File | Source | Notes |
|---|---|---|
| `Pd_cod_9008478.cif` | Crystallography Open Database, entry [9008478](https://www.crystallography.net/cod/9008478.html) | fcc Pd bulk cell. Reference for the slab lattice constant. |
| `ethanol.sdf` | PubChem CID [702](https://pubchem.ncbi.nlm.nih.gov/compound/702), 3D record | Discrete molecule — exercises the RDKit validation layer. |

The NO→NO₃/Pd example does **not** require these files — the Pd(111) slab is
built directly by ASE (`ase.build.fcc111`). They are included to demonstrate
reading external structures (ASE for crystals, RDKit for molecules) and as a
provenance anchor for the lattice constant.
