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
    a: float | None = None  # lattice constant (A); None -> ASE default or relaxed
    relax_lattice: bool = True  # fit the bulk lattice constant to the chosen potential


@dataclass
class MLIPConfig:
    backend: str = "emt"  # emt | mace | fairchem
    model: str | None = None  # checkpoint name/path for mace/fairchem
    models: list[str] = field(default_factory=list)  # multi-model: each "model" or "backend:model"
    device: str = "cpu"

    def specs(self) -> list[tuple[str, str | None]]:
        """(backend, model) pairs to run. Multi-model when ``models`` is set."""
        if self.models:
            out: list[tuple[str, str | None]] = []
            for m in self.models:
                if ":" in m:
                    b, mm = m.split(":", 1)
                    out.append((b, mm or None))
                else:
                    out.append((self.backend, m))
            return out
        return [(self.backend, self.model)]


@dataclass
class RenderConfig:
    """How to render active-site structure thumbnails/gallery."""

    backend: str = "matplotlib"  # matplotlib (flat, no deps) | povray (ray-traced)
    width: int = 320             # povray canvas width per view (px)
    bonds: bool = True           # draw ball-and-stick bonds (povray)


@dataclass
class SearchConfig:
    pose_count: int = 4  # adsorption poses per adsorbate
    neb_images: int = 5  # intermediate images (excluding endpoints)
    fmax: float = 0.05  # eV/A convergence on max force (relaxation)
    max_steps: int = 200
    neb_fmax: float = 0.1  # eV/A convergence on the NEB band
    neb_max_steps: int = 80
    neb_retries: int = 1  # on non-convergence, retry with a denser band + more steps
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    # similarity / acceptance thresholds
    rmsd_thresh: float = 0.7  # A
    energy_thresh: float = 0.05  # eV (~1 kcal/mol) for "same" energy


@dataclass
class AutoConfig:
    """Controls for ``network: auto`` (rule-guided intermediate autodetection)."""

    max_extra: int = 4  # atom budget = len(substrate atoms) + max_extra
    max_states: int = 600  # safety cap on how many states the explorer generates
    # rough-energy pruning: drop states whose quick (pre-relaxed) energy is more
    # than this many eV above the substrate root, keeping only what still connects
    # root -> target.  None disables it (keep every path to target).
    prune_energy: float | None = None


@dataclass
class SubstrateSpec:
    """One (substrate -> target) network to run in a multi-substrate job."""

    substrate: str
    target: str = ""
    network: str = "ammonia"
    reagents: list[str] | None = None
    name: str = ""  # optional row label / run-folder suffix

    def __post_init__(self) -> None:
        self.substrate = str(self.substrate)
        self.target = str(self.target) if self.target else self.substrate


@dataclass
class Config:
    name: str = "run"
    substrate: str = "NO"  # starting adsorbate label
    target: str = "NH3"  # ending adsorbate label
    network: str = "ammonia"  # ammonia | branching | oxidation
    # which reagent adatoms are available (filters the network branches);
    # None = use the full template (back-compat), [] = reagent-free steps only.
    reagents: list[str] | None = None
    substrates: list = field(default_factory=list)  # rows: labels or SubstrateSpec dicts
    slab: SlabConfig = field(default_factory=SlabConfig)
    mlip: MLIPConfig = field(default_factory=MLIPConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)  # network: auto controls
    outdir: str = "runs"

    def __post_init__(self) -> None:
        self.slab.size = tuple(self.slab.size)  # type: ignore[assignment]
        # guard against YAML bool coercion / numeric labels leaking through
        self.substrate = str(self.substrate)
        self.target = str(self.target)
        if self.reagents is not None:
            self.reagents = [str(r) for r in self.reagents]
        # substrate entries are either bare labels (str) or full spec dicts
        self.substrates = [s if isinstance(s, dict) else str(s)
                           for s in self.substrates]
        if not self.substrates:
            self.substrates = [self.substrate]

    def substrate_runs(self) -> list["SubstrateSpec"]:
        """Normalise ``substrates`` into explicit (substrate, target, network) specs.

        A bare-string entry inherits this config's target/network/reagents; a dict
        entry overrides them per-substrate.  A single-entry result is just the
        ordinary single-substrate run.
        """
        specs: list[SubstrateSpec] = []
        for s in self.substrates:
            if isinstance(s, dict):
                d = dict(s)
                d.setdefault("target", self.target)
                d.setdefault("network", self.network)
                d.setdefault("reagents", self.reagents)
                specs.append(SubstrateSpec(**d))
            else:
                specs.append(SubstrateSpec(substrate=s, target=self.target,
                                           network=self.network,
                                           reagents=self.reagents))
        return specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(_load_yaml(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = copy.deepcopy(data)
        slab = SlabConfig(**data.pop("slab", {}))
        mlip = MLIPConfig(**data.pop("mlip", {}))
        render = RenderConfig(**data.pop("render", {}))
        auto = AutoConfig(**data.pop("auto", {}))
        search_data = data.pop("search", {})
        if "size" in search_data:  # tolerate misplacement
            search_data.pop("size")
        search = SearchConfig(**search_data)
        # normalise tuple fields that YAML gives as lists
        slab.size = tuple(slab.size)  # type: ignore[assignment]
        cfg = cls(slab=slab, mlip=mlip, search=search, render=render, auto=auto, **data)
        if not cfg.substrates:
            cfg.substrates = [cfg.substrate]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self, path: str | Path) -> None:
        """Write a provenance snapshot so the run is reproducible."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
