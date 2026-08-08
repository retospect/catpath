"""Run configuration: load from YAML, validate, and snapshot for provenance."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    backend: str = "emt"  # emt | mace | chgnet | fairchem | grace | auto
    model: str | None = None  # checkpoint / model name (backend-specific)
    models: list[str] = field(default_factory=list)  # multi-model: each "model" or "backend:model"
    device: str = "cpu"
    task: str | None = None  # FAIRChem/UMA task head (defaults to "oc20": adsorbates on metals)

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
    # separate from `neb_retries` above: on a still-non-converged NEB, run ONE more
    # attempt with double the step budget only (same images/fmax) before giving up --
    # see pipeline.run_one_seed. Opt-out for callers that want the raw first-attempt
    # convergence flag reported as-is.
    neb_auto_retry: bool = True
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    # gate the NEB on an endpoint binding pre-flight (see pipeline._relax_state_bound):
    # a desorbed endpoint gets a restrained-then-released reseat attempt before the
    # (expensive) NEB is allowed to run at all. False reproduces pre-preflight behavior.
    bind_preflight: bool = True
    # reseat tether controls (pipeline._relax_state_bound). ``rt`` is the Hookean
    # rest length: the tether only pulls once the adsorbate is farther than this
    # from its nearest slab atom, so it must sit INSIDE the real chemisorption bond
    # range (~1.9-2.1 A for N*/O* on a late transition metal) to pull a desorbing
    # fragment back INTO the well — a longer rt (>2.5) merely leashes runaway
    # desorption at arm's length and releases to re-desorb. More attempts = more
    # fresh poses tried before an endpoint is declared genuinely non-binding.
    bind_tether_k: float = 7.5      # eV/A^2 Hookean spring constant
    bind_tether_rt: float = 2.0     # A rest length (real M-adsorbate bond range)
    bind_reseat_attempts: int = 3   # reseat tries (fresh pose each) before giving up
    # DISSOLVING tether: rather than yank the adsorbate in at full ``k`` and drop
    # the spring in one step (which kicks it straight back off — "leash and
    # re-desorb"), relax through a decreasing schedule of spring constants so the
    # fragment settles into the chemisorption well ADIABATICALLY. Each entry is a
    # fraction of ``bind_tether_k``; the final 0.0 is the unconstrained relax that
    # the endpoint must survive to count as bound. The per-stage energies also
    # trace the adsorption-barrier profile (see ``_relax_state_bound``).
    bind_tether_ramp: list[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.1, 0.0])
    # ADSORPTION barrier: the max energy the fragment must climb as the tether
    # pulls it from its desorbed position into the well (0 if the pull-in is
    # monotonically downhill = barrierless, a spurious desorption legitimately
    # rescued). A reseat that only binds by crossing MORE than this is genuine
    # activated adsorption — the endpoint is reported non-binding (untrusted,
    # excluded from ranking) rather than fed a barrier the site can't really reach.
    bind_ads_barrier_max: float = 0.75  # eV
    # similarity / acceptance thresholds
    rmsd_thresh: float = 0.7  # A
    energy_thresh: float = 0.05  # eV (~1 kcal/mol) for "same" energy
    # SCREENING mode: relax-only cheap thermodynamic tier -- endpoint
    # relaxations (bind preflight included) run exactly as usual, but NEB
    # (and the linear-interpolation barrier fallback) never runs at all. Every
    # edge is left with NO `barrier` key -- never a fabricated 0.0 -- so U_L /
    # span / rate_Ea downstream are honestly thermodynamic-only figures, not a
    # disguised (and wrong) kinetic estimate. See pipeline.run_one_seed /
    # aggregate_partials / graph.build_graph / electrochem._apply_electrochemistry.
    screening: bool = False


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
class ElectrochemistryConfig:
    """Computational hydrogen electrode (CHE) post-processing -- pure
    post-processing over energies already computed, no new relax/NEB calls.
    See ``docs/proposals/pathway-potential-lever.md`` (slice 1) in the
    precis-mcp repo.

    ``U_vs_RHE``: the operating potential (V vs RHE), or the literal string
    ``"optimal"``. catpath always computes both the limiting potential
    (``U_L``) and the span-minimizing potential (``U_opt``, plus its span) --
    they are closed-form and cheap either way; ``U_vs_RHE`` is provenance
    (round-tripped via ``Config.snapshot``) for downstream (precis/explorer)
    to pick a default display potential from.
    ``pH``: optional; only matters for RHE<->SHE display conversion and any
    (currently dormant) decoupled-proton step -- PCET (H-supply) steps are
    pH-independent on the RHE scale.
    ``T``: temperature (K), default 298.15 (standard ambient).
    ``U_window``: optional ``(lo, hi)`` bound (V) for the U_opt search; ``None``
    searches the unbounded closed form.
    """

    U_vs_RHE: float | str = 0.0
    pH: float | None = None
    T: float = 298.15
    U_window: tuple[float, float] | None = None


@dataclass
class SubstrateSpec:
    """One (substrate -> target) network to run in a multi-substrate job."""

    substrate: str
    target: str = ""
    network: str = "ammonia"
    template: str = "parked"  # kind="ammonia" only -- see Config.template
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
    # kind="ammonia" only: "parked" (default, fragment-parking approximation,
    # unchanged behavior) | "coadsorbed" (verify tier -- both dissociation
    # fragments stay in-cell until a product desorbs; see
    # network.build_coadsorbed_ammonia_network). Ignored/invalid for any
    # other network kind.
    template: str = "parked"
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
    # present -> pipeline.write_outputs stamps n_H_rel per node and adds
    # U_L/U_opt/span_at_UL/span_at_Uopt to results.json (slice 1 of the
    # potential-lever proposal); absent (default) -> no CHE post-processing.
    electrochemistry: ElectrochemistryConfig | None = None

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

    def substrate_runs(self) -> list[SubstrateSpec]:
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
                d.setdefault("template", self.template)
                d.setdefault("reagents", self.reagents)
                specs.append(SubstrateSpec(**d))
            else:
                specs.append(SubstrateSpec(substrate=s, target=self.target,
                                           network=self.network,
                                           template=self.template,
                                           reagents=self.reagents))
        return specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls.from_dict(_load_yaml(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
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
        electrochemistry = None
        ec_data = data.pop("electrochemistry", None)
        if ec_data is not None:
            ec_data = dict(ec_data)
            if ec_data.get("U_window") is not None:
                ec_data["U_window"] = tuple(ec_data["U_window"])  # type: ignore[assignment]
            electrochemistry = ElectrochemistryConfig(**ec_data)
        cfg = cls(slab=slab, mlip=mlip, search=search, render=render, auto=auto,
                  electrochemistry=electrochemistry, **data)
        if not cfg.substrates:
            cfg.substrates = [cfg.substrate]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self, path: str | Path) -> None:
        """Write a provenance snapshot so the run is reproducible."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
