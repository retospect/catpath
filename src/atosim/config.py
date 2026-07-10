"""Run configuration: load from YAML, validate, and snapshot for provenance."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import re

import yaml


class _ChemSafeLoader(yaml.SafeLoader):
    """SafeLoader that does NOT coerce NO/YES/ON/OFF to booleans.

    Chemical labels like ``NO`` are common here; YAML 1.1's implicit bool
    resolver would silently turn ``substrate: NO`` into ``False``.  We strip the
    bool resolver and re-add it restricted to only true/false spellings.
    """


# Drop every existing bool resolver, then keep only true/false/True/False.
_ChemSafeLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for (tag, rx) in resolvers if tag != "tag:yaml.org,2002:bool"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ChemSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _load_yaml(text: str) -> dict:
    return yaml.load(text, Loader=_ChemSafeLoader) or {}


@dataclass
class SlabConfig:
    element: str = "Pd"
    size: tuple[int, int, int] = (3, 3, 4)
    vacuum: float = 10.0
    fix_layers: int = 2  # freeze this many bottom layers


@dataclass
class MLIPConfig:
    backend: str = "emt"  # emt | mace | fairchem
    model: str | None = None  # checkpoint name/path for mace/fairchem
    device: str = "cpu"


@dataclass
class SearchConfig:
    pose_count: int = 4  # adsorption poses per adsorbate
    neb_images: int = 5  # intermediate images (excluding endpoints)
    fmax: float = 0.05  # eV/A convergence on max force (relaxation)
    max_steps: int = 200
    neb_fmax: float = 0.1  # eV/A convergence on the NEB band
    neb_max_steps: int = 80
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    # similarity / acceptance thresholds
    rmsd_thresh: float = 0.7  # A
    energy_thresh: float = 0.05  # eV (~1 kcal/mol) for "same" energy


@dataclass
class Config:
    name: str = "run"
    substrate: str = "NO"  # starting adsorbate label
    target: str = "NO3"  # ending adsorbate label
    network: str = "branching"  # branching | oxidation
    substrates: list[str] = field(default_factory=list)  # rows of the energy map
    slab: SlabConfig = field(default_factory=SlabConfig)
    mlip: MLIPConfig = field(default_factory=MLIPConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    outdir: str = "runs"

    def __post_init__(self) -> None:
        self.slab.size = tuple(self.slab.size)  # type: ignore[assignment]
        # guard against YAML bool coercion / numeric labels leaking through
        self.substrate = str(self.substrate)
        self.target = str(self.target)
        self.substrates = [str(s) for s in self.substrates]
        if not self.substrates:
            self.substrates = [self.substrate]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(_load_yaml(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = copy.deepcopy(data)
        slab = SlabConfig(**data.pop("slab", {}))
        mlip = MLIPConfig(**data.pop("mlip", {}))
        search_data = data.pop("search", {})
        if "size" in search_data:  # tolerate misplacement
            search_data.pop("size")
        search = SearchConfig(**search_data)
        # normalise tuple fields that YAML gives as lists
        slab.size = tuple(slab.size)  # type: ignore[assignment]
        cfg = cls(slab=slab, mlip=mlip, search=search, **data)
        if not cfg.substrates:
            cfg.substrates = [cfg.substrate]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self, path: str | Path) -> None:
        """Write a provenance snapshot so the run is reproducible."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
