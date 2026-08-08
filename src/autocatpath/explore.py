"""Rule-guided intermediate autodetection (the ``network: auto`` backend).

Instead of hand-curating every intermediate (as ``network.py`` does), this module
*generates* the reaction network from a substrate label, a target label, and a
pool of reagent adatoms, by applying three elementary graph-rewrite rules to a
molecular graph of the adsorbate:

* **dissociate** (barriered step) - break one heavy-heavy bond, splitting a
  fragment in two.  Atoms are conserved; the byproduct rides along as a
  co-adsorbed spectator (mass is conserved on the surface).
* **supply** (barrierless link) - stage one reagent adatom (``+H*`` / ``+O*``)
  from the reservoir next to the adsorbate.  This is the only move that changes
  composition.
* **react** (barriered step) - bond the staged reagent atom to an adsorbate atom
  that has spare valence.  Atoms are conserved.

These compose into dissociation, hydrogenation chains, associative (HNO/NOH) and
water branches -- the same routes as the curated templates -- but derived rather
than typed out.  Pruning: valence limits, an atom budget, and a reachability
filter that keeps only states on a path from the substrate to the target.

**Acyclicity (so the result is a DAG the pipeline can order).** Atom count never
decreases (``supply`` adds; ``dissociate``/``react`` conserve).  At a fixed atom
count no reagent is ever staged (``supply`` is the only way to stage one and it
adds an atom), so the only available move is ``dissociate``, which strictly
*lowers* the bond count.  Hence every edge strictly increases
``(atoms, -bonds)`` lexicographically -> no cycles.

**NEB endpoint alignment.** A reaction step conserves the element multiset, so
ordering each state's atoms by a fixed element priority makes a step's reactant
and product share an identical element sequence -- exactly what NEB interpolation
needs -- without any per-atom bookkeeping.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import networkx as nx

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .config import SlabConfig
from .network import Network, StateSpec, StepSpec

# Max bonds to *other adsorbate atoms* (the surface bond is not counted).
_VALENCE = {"H": 1, "O": 2, "N": 3, "C": 4, "S": 2}
# Fixed element layout/order priority; also the per-state spec ordering that
# guarantees NEB-compatible endpoints (see module docstring).  Lower sorts first.
_PRIORITY = {"C": 0, "N": 1, "O": 2, "S": 3, "H": 9}

# Geometry constants (materialisation).  Endpoints only need to be non-overlapping
# and roughly physical -- the relaxer refines them -- but covalent-radius bond
# lengths and an upward substituent cone land closer to the true minimum, which
# tightens NEB convergence versus flat, fixed-distance placement.
_ADS_OFFSET = 1.15      # A: anchor height above surface, beyond its covalent radius
_CONE_DEG = 66.0        # substituent tilt off the surface normal (degrees); a wide
                        # lean keeps chain tips (N-O-H) under the detachment limit
_FRAG_CLEAR = 2.0       # A: extra clearance between co-adsorbed fragment footprints

_MAX_STATES = 600  # safety cap on exploration breadth


def _cov(el: str) -> float:
    return float(covalent_radii[atomic_numbers[el]])


def _bond_len(a: str, b: str) -> float:
    """Single-bond length ~ sum of covalent radii (e.g. N-H~1.0, N-O~1.4 A)."""
    return _cov(a) + _cov(b)


def _anchor_h(el: str) -> float:
    """Height of a surface-anchored atom: its covalent radius + an adsorption gap."""
    return _cov(el) + _ADS_OFFSET


# --- molecular graphs --------------------------------------------------------

def parse_molecule(label: str) -> nx.Graph:
    """Parse a formula label (``NO``, ``NH3``, ``NO2``, ``H2O``) to a star graph.

    The highest-priority heavy atom is the center; every other atom bonds to it.
    That reproduces the connectivity of the small molecules in play (a diatomic,
    an ``AXn`` oxide, an ``AHn`` hydride) which is all substrate/target labels
    ever are -- richer isomers (HNO vs NOH) arise only from *exploration*.
    """
    counts = _formula_counts(label)
    g = nx.Graph()
    atoms: list[str] = []
    for el, n in counts.items():
        atoms.extend([el] * n)
    for i, el in enumerate(atoms):
        g.add_node(i, el=el)
    if len(atoms) <= 1:
        return g
    heavy = [i for i, el in enumerate(atoms) if el != "H"]
    center = min(heavy or range(len(atoms)),
                 key=lambda i: _PRIORITY.get(atoms[i], 5))
    for i in range(len(atoms)):
        if i != center:
            g.add_edge(center, i)
    return g


def _formula_counts(label: str) -> "Counter[str]":
    counts: Counter[str] = Counter()
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", label):
        if el:
            counts[el] += int(num) if num else 1
    return counts


def _canon(g: nx.Graph, pending: int | None = None) -> tuple:
    """A label-independent canonical key (1-WL color refinement) for dedup.

    Includes whether a reagent atom is currently staged (``pending``), so a
    staged ``NO+H`` and a bonded ``HNO`` never collide.
    """
    colors = {n: g.nodes[n]["el"] + ("*" if n == pending else "") for n in g}
    for _ in range(g.number_of_nodes() + 1):
        sig = {n: (colors[n], tuple(sorted(colors[m] for m in g.neighbors(n))))
               for n in g}
        order = {s: i for i, s in enumerate(sorted(set(sig.values()), key=str))}
        colors = {n: str(order[sig[n]]) for n in g}
    nodes = tuple(sorted(colors.values()))
    edges = tuple(sorted(tuple(sorted((colors[u], colors[v]))) for u, v in g.edges()))
    formula = tuple(sorted(g.nodes[n]["el"] for n in g))
    return (formula, nodes, edges, pending is not None)


@dataclass
class _State:
    g: nx.Graph
    pending: int | None  # node id of a staged (unbonded) reagent atom, or None

    def key(self) -> tuple:
        return _canon(self.g, self.pending)


def _free_valence(g: nx.Graph, n: int) -> int:
    return _VALENCE.get(g.nodes[n]["el"], 0) - g.degree(n)


# --- exploration -------------------------------------------------------------

def _successors(s: _State, reagents: set[str], max_atoms: int):
    """Yield ``(kind, new_State, meta)`` for every rule application to ``s``."""
    g, pend = s.g, s.pending
    if pend is None:
        # dissociate: break a heavy-heavy bond
        for u, v in g.edges():
            if g.nodes[u]["el"] != "H" and g.nodes[v]["el"] != "H":
                h = g.copy()
                h.remove_edge(u, v)
                yield "step", _State(h, None), f"break {g.nodes[u]['el']}-{g.nodes[v]['el']}"
        # supply: stage one reagent atom (only if none staged, budget permitting)
        if g.number_of_nodes() < max_atoms:
            nxt = (max(g.nodes) + 1) if g.number_of_nodes() else 0
            for r in sorted(reagents):
                h = g.copy()
                h.add_node(nxt, el=r)
                yield "link", _State(h, nxt), f"+{r}*"
    else:
        # react: bond the staged atom to any atom with spare valence
        if _free_valence(g, pend) > 0:
            for x in g:
                if x != pend and _free_valence(g, x) > 0:
                    h = g.copy()
                    h.add_edge(pend, x)
                    yield "step", _State(h, None), f"{g.nodes[pend]['el']} binds {g.nodes[x]['el']}"


def _is_goal(g: nx.Graph, target: nx.Graph) -> bool:
    """True if some connected fragment of ``g`` is isomorphic to ``target``."""
    tkey = _canon(target)
    for comp in nx.connected_components(g):
        if _canon(g.subgraph(comp).copy()) == tkey:
            return True
    return False


def _default_reagents(sub: nx.Graph, tgt: nx.Graph) -> set[str]:
    """Reagents needed = elements the target has more of than the substrate."""
    diff = (Counter(tgt.nodes[n]["el"] for n in tgt)
            - Counter(sub.nodes[n]["el"] for n in sub))
    return {e for e, c in diff.items() if c > 0} or {"H"}


def explore(substrate: str, target: str, reagents: set[str] | None = None,
            max_extra: int = 4, max_states: int = _MAX_STATES):
    """Breadth-first rule application from ``substrate`` toward ``target``.

    Returns ``(nodes, edges, root_key, goal_keys)`` where ``nodes`` maps a
    canonical key to its ``_State`` and ``edges`` is a list of
    ``(src_key, dst_key, kind, meta)``.
    """
    sub_g = parse_molecule(substrate)
    tgt_g = parse_molecule(target)
    if reagents is None:
        reagents = _default_reagents(sub_g, tgt_g)
    max_atoms = sub_g.number_of_nodes() + max_extra

    root = _State(sub_g, None)
    rkey = root.key()
    nodes: dict[tuple, _State] = {rkey: root}
    edges: list[tuple] = []
    goals: set[tuple] = set()
    seen_edges: set[tuple] = set()
    queue = [rkey]
    if _is_goal(sub_g, tgt_g):
        goals.add(rkey)
    while queue:
        skey = queue.pop(0)
        for kind, ns, meta in _successors(nodes[skey], reagents, max_atoms):
            nkey = ns.key()
            if nkey not in nodes:
                if len(nodes) >= max_states:  # hard breadth cap: skip new states
                    continue                  # (and the edges into them)
                nodes[nkey] = ns
                if _is_goal(ns.g, tgt_g):
                    goals.add(nkey)
                queue.append(nkey)
            e = (skey, nkey, kind)
            if skey != nkey and e not in seen_edges:
                seen_edges.add(e)
                edges.append((skey, nkey, kind, meta))
    return nodes, edges, rkey, goals


def _prune_to_target(nodes, edges, root_key, goal_keys):
    """Keep only nodes on a path from the root to a goal (drop dead branches)."""
    if not goal_keys:
        return set(nodes), edges
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for a, b, *_ in edges:
        g.add_edge(a, b)
    keep: set = set(goal_keys)
    for gk in goal_keys:
        keep |= nx.ancestors(g, gk)
    if root_key not in keep:  # target unreachable from root -> keep everything
        return set(nodes), edges
    kept_edges = [e for e in edges if e[0] in keep and e[1] in keep]
    return keep, kept_edges


# --- materialisation: molecular graph -> StateSpec placements ----------------

def _hpre(k: int) -> str:   # H's shown before their host atom (H, H2, ...)
    return "" if k == 0 else "H" if k == 1 else f"H{k}"


def _hpost(k: int) -> str:  # H's shown after their host atom (H, H2, ...)
    return "" if k == 0 else "H" if k == 1 else f"H{k}"


def _frag_name(g: nx.Graph, comp) -> str:
    els = sorted(g.nodes[n]["el"] for n in comp)
    if len(comp) == 1:
        return els[0]
    heavies = [n for n in comp if g.nodes[n]["el"] != "H"]
    n_h = len(comp) - len(heavies)
    if len(heavies) == 1:
        e = g.nodes[heavies[0]]["el"]
        return e + ("" if n_h == 0 else "H" if n_h == 1 else f"H{n_h}")
    # a center with identical heavy substituents and no H -> NO2 / NO3
    for c in heavies:
        others = [m for m in heavies if m != c]
        if (n_h == 0 and all(g.has_edge(c, m) for m in others)
                and len({g.nodes[m]["el"] for m in others}) == 1):
            e, oe, k = g.nodes[c]["el"], g.nodes[others[0]]["el"], len(others)
            return f"{e}{oe}{k if k > 1 else ''}"
    # a two-heavy chain with H's -> HNO / NOH (H shown next to its host)
    if len(heavies) == 2:
        a, b = sorted(heavies, key=lambda n: _PRIORITY.get(g.nodes[n]["el"], 5))
        ha = sum(1 for m in g.neighbors(a) if g.nodes[m]["el"] == "H")
        hb = sum(1 for m in g.neighbors(b) if g.nodes[m]["el"] == "H")
        ea, eb = g.nodes[a]["el"], g.nodes[b]["el"]
        return _hpre(ha) + ea + eb + _hpost(hb)
    cnt = Counter(els)
    return "".join(f"{e}{cnt[e] if cnt[e] > 1 else ''}" for e in sorted(cnt))


def _state_name(g: nx.Graph) -> str:
    comps = list(nx.connected_components(g))
    names = sorted((_frag_name(g, c) for c in comps),
                   key=lambda s: (-len(s), s))
    return "+".join(names)


def _hill(g: nx.Graph) -> str:
    cnt = Counter(g.nodes[n]["el"] for n in g)
    return "".join(f"{e}{cnt[e] if cnt[e] > 1 else ''}"
                   for e in sorted(cnt, key=lambda e: _PRIORITY.get(e, 5)))


def _fragment_geom(g: nx.Graph, comp) -> tuple[dict, int]:
    """Local (x, y, z) per atom of ONE fragment, anchor at the xy origin.

    Bond lengths come from covalent radii; substituents fan upward in a cone
    (a lone heavy child stacks vertically, e.g. O over N in N-O).  Returns
    ``(positions, anchor)``.
    """
    sub = g.subgraph(comp)
    anchor = min(comp, key=lambda n: (_PRIORITY.get(g.nodes[n]["el"], 5),
                                      -sub.degree(n), n))
    pos = {anchor: np.array([0.0, 0.0, _anchor_h(g.nodes[anchor]["el"])])}
    depth = {anchor: 0}
    cone = math.radians(_CONE_DEG)
    seen, queue = {anchor}, [anchor]
    while queue:
        p = queue.pop(0)
        kids = [m for m in sub.neighbors(p) if m not in seen]
        for i, m in enumerate(kids):
            b = _bond_len(g.nodes[p]["el"], g.nodes[m]["el"])
            # fan every substituent into an upward cone.  A wide tilt matters for
            # chains (N-O-H): stacking bonds vertically would lift the tip past
            # the ~3.5 A "detached from slab" limit, so we lean instead.
            azi = 2 * math.pi * i / len(kids) + 0.7 * depth[p]
            d = np.array([math.sin(cone) * math.cos(azi),
                          math.sin(cone) * math.sin(azi), math.cos(cone)])
            pos[m] = pos[p] + b * d
            depth[m] = depth[p] + 1
            seen.add(m)
            queue.append(m)
    return pos, anchor


def _footprint(pos: dict, anchor: int) -> float:
    a = pos[anchor][:2]
    return max((float(np.linalg.norm(pos[n][:2] - a)) for n in pos), default=0.0)


def _layout(g: nx.Graph) -> dict:
    """Assign (dx, dy, height, site) to every atom.

    Each fragment is built with :func:`_fragment_geom`, then fragments are spaced
    by their footprints so they cannot overlap.  The primary (largest) fragment
    sits at an fcc hollow; co-adsorbed byproducts / staged reagents go to hcp --
    distinct sites, matching how the curated templates stage extra adatoms.
    """
    comps = sorted(nx.connected_components(g),
                   key=lambda c: (-len(c), sorted(g.nodes[n]["el"] for n in c)))
    geoms = [_fragment_geom(g, c) for c in comps]
    foots = [_footprint(p, a) for p, a in geoms]
    dirs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.7, 0.7), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7)]
    out: dict = {}
    for ci, (pos, _anchor) in enumerate(geoms):
        if ci == 0:
            cx, cy, site = 0.0, 0.0, "fcc"
        else:
            gap = foots[0] + foots[ci] + _FRAG_CLEAR
            ux, uy = dirs[(ci - 1) % len(dirs)]
            cx, cy, site = ux * gap, uy * gap, "hcp"
        for n, p in pos.items():
            out[n] = (cx + float(p[0]), cy + float(p[1]), float(p[2]), site)
    return out


def _materialise(g: nx.Graph) -> StateSpec:
    pos = _layout(g)
    specs = [{"symbol": g.nodes[n]["el"], "site": pos[n][3],
              "height": round(pos[n][2], 3), "dx": round(pos[n][0], 3),
              "dy": round(pos[n][1], 3)} for n in g]
    # order by element priority -> equal-multiset states share atom ordering,
    # so any reaction step's endpoints align for NEB interpolation.
    specs.sort(key=lambda s: _PRIORITY.get(s["symbol"], 5))
    return StateSpec(_state_name(g), _hill(g), specs)


# --- public entry point ------------------------------------------------------

def build_auto_network(slab_cfg: SlabConfig, substrate: str = "NO",
                       target: str | None = None,
                       reagents: list[str] | None = None,
                       max_extra: int = 4,
                       max_states: int = _MAX_STATES) -> Network:
    """Autodetect the reaction network from ``substrate`` -> ``target``.

    ``reagents`` defaults to the elements the target needs more of than the
    substrate (e.g. NO->NH3 needs only H; NO->NO3 needs only O).  ``max_extra``
    is the reagent-atom budget and ``max_states`` caps exploration breadth.
    """
    target = target or substrate
    nodes, edges, rkey, goals = explore(
        substrate, target, set(reagents) if reagents is not None else None,
        max_extra=max_extra, max_states=max_states)
    keep, edges = _prune_to_target(nodes, edges, rkey, goals)

    specs = {k: _materialise(nodes[k].g) for k in keep}
    # disambiguate any name collisions so node ids stay unique
    used: dict[str, int] = {}
    seen_names: dict[str, tuple] = {}
    for k, st in specs.items():
        if st.name in seen_names and seen_names[st.name] != k:
            used[st.name] = used.get(st.name, 1) + 1
            st.name = f"{st.name}#{used[st.name]}"
        seen_names.setdefault(st.name, k)

    steps: list[StepSpec] = []
    links: list[tuple[str, str]] = []
    step_states: set[str] = set()
    for a, b, kind, meta in edges:
        na, nb = specs[a].name, specs[b].name
        if kind == "step":
            steps.append(StepSpec(f"{na}->{nb}", specs[a], specs[b]))
            step_states |= {na, nb}
        else:
            links.append((na, nb))
    # keep only links whose endpoints also appear in a step (Network.states()
    # is derived from steps), matching the curated-network invariant
    links = [(a, b) for a, b in links if a in step_states and b in step_states]
    return Network(slab_cfg, steps=steps, links=links)


def _target_state_name(target: str) -> str:
    """State name of the pure target fragment (e.g. 'NH3') -- used to spot goal
    states like 'NH3+O' by fragment membership."""
    return _state_name(parse_molecule(target))


def prune_by_rough_energy(net: Network, make_calc, target: str, threshold: float,
                          prerelax_steps: int = 40, log=lambda *a, **k: None) -> Network:
    """Drop states whose quick pre-relaxed energy is > ``threshold`` eV above the
    substrate root, then keep only what still connects root -> a target state.

    Deterministic (a short relax, no rattle), so every seed prunes to the SAME
    network and the energy-map columns stay aligned.  If pruning would sever the
    target it is skipped (returns ``net`` unchanged) with a warning -- a run must
    always retain at least one substrate->target path.
    """
    from .relax import pre_relax

    order = net.order()
    if not order:
        return net
    root = order[0]
    states = net.states()
    goal_name = _target_state_name(target)
    goals = {n for n in states if goal_name in n.split("+")}

    slab = net.slab()  # honours an injected prebuilt_slab (the precis structure seam)
    energy: dict[str, float] = {}
    for name, st in states.items():
        atoms = pre_relax(st.build(slab), make_calc(), max_steps=prerelax_steps)
        energy[name] = float(atoms.get_potential_energy())
    e0 = energy[root]
    keep = {n for n, e in energy.items() if e - e0 <= threshold} | {root} | goals

    # restrict to states still on a root -> goal path within the kept set
    g = nx.DiGraph()
    g.add_nodes_from(keep)
    for s in net.steps:
        if s.reactant.name in keep and s.product.name in keep:
            g.add_edge(s.reactant.name, s.product.name)
    for a, b in net.links:
        if a in keep and b in keep:
            g.add_edge(a, b)
    reach = {root} | (nx.descendants(g, root) if root in g else set())
    to_goal: set = set()
    for gl in goals & set(g):
        to_goal |= {gl} | nx.ancestors(g, gl)
    final = reach & to_goal

    if not (goals & final) or root not in final:
        log("warning: rough-energy pruning would sever the target; skipped")
        return net

    steps = [s for s in net.steps
             if s.reactant.name in final and s.product.name in final]
    kept = set()
    for s in steps:
        kept |= {s.reactant.name, s.product.name}
    links = [(a, b) for a, b in net.links if a in kept and b in kept]
    log(f"rough-energy pruning: kept {len(kept)}/{len(states)} states "
        f"(<= {threshold:.2f} eV above root)")
    return Network(net.slab_cfg, steps=steps, links=links)
