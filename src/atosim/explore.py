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

from .config import SlabConfig
from .network import Network, StateSpec, StepSpec

# Max bonds to *other adsorbate atoms* (the surface bond is not counted).
_VALENCE = {"H": 1, "O": 2, "N": 3, "C": 4, "S": 2}
# Fixed element layout/order priority; also the per-state spec ordering that
# guarantees NEB-compatible endpoints (see module docstring).  Lower sorts first.
_PRIORITY = {"C": 0, "N": 1, "O": 2, "S": 3, "H": 9}
# Anchor height (A above the surface) when an atom sits directly on the slab.
_ANCHOR_H = {"C": 1.9, "N": 1.8, "O": 1.7, "S": 1.9, "H": 1.1}

_MAX_STATES = 600  # safety cap on exploration breadth


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
            max_extra: int = 4):
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
    while queue and len(nodes) < _MAX_STATES:
        skey = queue.pop(0)
        for kind, ns, meta in _successors(nodes[skey], reagents, max_atoms):
            nkey = ns.key()
            if nkey not in nodes:
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


def _layout(g: nx.Graph) -> dict:
    """Assign (dx, dy, height) to every atom: fragments on separate slots,
    substituents fanned around their anchor.  Endpoints only need to be
    non-overlapping and roughly physical -- the relaxer refines them."""
    slots = [(0.0, 0.0), (2.6, 0.0), (-2.6, 0.0), (0.0, 2.6),
             (0.0, -2.6), (2.6, 2.6), (-2.6, -2.6), (2.6, -2.6)]
    comps = sorted(nx.connected_components(g),
                   key=lambda c: (-len(c), sorted(g.nodes[n]["el"] for n in c)))
    pos: dict = {}
    for ci, comp in enumerate(comps):
        ox, oy = slots[ci % len(slots)]
        sub = g.subgraph(comp)
        anchor = min(comp, key=lambda n: (_PRIORITY.get(g.nodes[n]["el"], 5),
                                          -sub.degree(n), n))
        pos[anchor] = (ox, oy, _ANCHOR_H.get(g.nodes[anchor]["el"], 1.7))
        seen, queue = {anchor}, [anchor]
        while queue:
            p = queue.pop(0)
            kids = [m for m in sub.neighbors(p) if m not in seen]
            px, py, ph = pos[p]
            for i, m in enumerate(kids):
                ang = 2 * math.pi * i / max(1, len(kids)) + 0.4 * ph
                pos[m] = (px + 0.95 * math.cos(ang), py + 0.95 * math.sin(ang),
                          ph + 0.9)
                seen.add(m)
                queue.append(m)
    return pos


def _materialise(g: nx.Graph) -> StateSpec:
    pos = _layout(g)
    specs = [{"symbol": g.nodes[n]["el"], "site": "fcc",
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
                       max_extra: int = 4) -> Network:
    """Autodetect the reaction network from ``substrate`` -> ``target``.

    ``reagents`` defaults to the elements the target needs more of than the
    substrate (e.g. NO->NH3 needs only H; NO->NO3 needs only O).
    """
    target = target or substrate
    nodes, edges, rkey, goals = explore(
        substrate, target, set(reagents) if reagents is not None else None,
        max_extra=max_extra)
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
