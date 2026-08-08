"""Computational hydrogen electrode (CHE) post-processing over energies a run
already has -- no new relax/NEB calls; thermodynamic-only (no beta symmetry
factor). See ``docs/proposals/pathway-potential-lever.md`` (slice 1) in the
precis-mcp repo for the design of record.

catpath's curated networks model hydrogenation as **supply edges** (``+H*``
staged from a reservoir). Under CHE (Norskov), an applied potential U (V vs
RHE) enters only through that reservoir's chemical potential -- each supplied
H represents H+ + e-, so

    G_node(U) = G_node(0) + n_H_rel(node) * e*U

where ``n_H_rel`` is the node's reservoir-H count *relative to the root
state's own H count* (so a root that already carries H still works), and
``e*U`` in eV is just ``U`` numerically (e = 1 in these units). Every
downstream quantity (limiting potential, span-vs-U) is therefore a max/min of
functions affine in U, so each has a closed-form (not searched) optimum.

Pure functions only: everything here takes plain dicts/lists/strings, no
networkx/Config/pipeline dependency, so it is trivial to unit test and reuse
from either the catpath pipeline or the precis bridge.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable

K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K
T_STANDARD = 298.15  # K -- standard ambient (25 C); the CHE decision, NOT the
# 300 K "round number" some MD/relax conventions default to.

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


# --- H-content parsing --------------------------------------------------

def element_counts(fragment: str) -> dict[str, int]:
    """Parse one chemical-formula fragment (e.g. ``'NH2'``, ``'H2O'``) into
    element -> atom-count.

    Anything from ``'@'`` onward is a site-isomer suffix (e.g. ``'NO@top'``),
    not part of the formula, and is stripped before parsing.
    """
    formula = fragment.split("@", 1)[0]
    counts: dict[str, int] = {}
    for sym, digits in _FORMULA_TOKEN.findall(formula):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(digits) if digits else 1)
    return counts


def n_H(node_id: str) -> int:
    """Total H atoms in a node id, summed over its ``'+'``-split fragment
    labels (e.g. ``'NH2+OH'`` -> 3, ``'NO+H'`` -> 1, ``'NO'`` -> 0)."""
    return sum(element_counts(frag).get("H", 0) for frag in node_id.split("+"))


def n_H_rel(node_id: str, root_id: str) -> int:
    """H atoms absorbed from the reservoir, relative to the root state's own
    H count -- the quantity the CHE shift ``G_node(U) = G_node(0) +
    n_H_rel*eU`` actually uses."""
    return n_H(node_id) - n_H(root_id)


# --- limiting potential --------------------------------------------------

def electrochemical_steps(
    edges: Iterable[tuple[str, str]], root: str
) -> list[tuple[str, str]]:
    """Edges (reactant, product) where n_H_rel increases -- each is one PCET
    (one supplied H+ + e-)."""
    return [(a, b) for a, b in edges if n_H_rel(b, root) > n_H_rel(a, root)]


def limiting_potential(
    node_energy0: dict[str, float],
    edges: Iterable[tuple[str, str]],
    root: str,
) -> float | None:
    """U_L (V vs RHE) = -max over electrochemical steps of dG_step(0) (eV).

    ``None`` if ``edges`` contains no electrochemical (H-supply) step.
    """
    steps = electrochemical_steps(edges, root)
    if not steps:
        return None
    dg = max(node_energy0[b] - node_energy0[a] for a, b in steps)
    return -dg


# --- pathway (mirrors catpath.precis.analysis.reaction_path so span figures
# agree between the catpath-native and precis-served views -- kept local
# rather than imported so this module has no dependency on the precis bridge
# subpackage) -----------------------------------------------------------

def reaction_path(
    edges: Iterable[tuple[str, str]], root: str, target: str
) -> list[str]:
    """Shortest root->target chain across ALL edges (reaction + supply).
    ``[]`` if unreachable."""
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    q: deque[list[str]] = deque([[root]])
    seen = {root}
    while q:
        path = q.popleft()
        if path[-1] == target:
            return path
        for nxt in adj.get(path[-1], []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])
    return []


# --- graph traversal: reachability, path enumeration, required leaves ------

def reachable_from(edges: Iterable[tuple[str, str]], root: str) -> set[str]:
    """Every node reachable from ``root`` by following ``edges`` forward
    (reaction + supply -- the whole graph, not just one path)."""
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


_MAX_PATH_EXPANSIONS = 20_000  # defensive cap on BFS-of-paths queue growth


def enumerate_paths(
    edges: Iterable[tuple[str, str]],
    root: str,
    leaf: str,
    max_paths: int = 64,
    log=lambda *a, **k: None,
) -> list[list[str]]:
    """Every simple root->leaf path across ``edges``, fewest-hops first,
    capped at ``max_paths`` (a branching DAG can have combinatorially many
    routes; logs when the cap binds). BFS-over-partial-paths naturally
    discovers completions in non-decreasing length order, so truncating at
    the cap keeps the cheapest routes."""
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    paths: list[list[str]] = []
    q: deque[list[str]] = deque([[root]])
    expansions = 0
    capped_by_expansions = False
    while q and len(paths) < max_paths:
        path = q.popleft()
        node = path[-1]
        if node == leaf:
            paths.append(path)
            continue
        expansions += 1
        if expansions > _MAX_PATH_EXPANSIONS:
            capped_by_expansions = True
            break
        for nxt in adj.get(node, ()):
            if nxt not in path:  # simple path -- no repeated nodes
                q.append(path + [nxt])
    if len(paths) >= max_paths or capped_by_expansions:
        log(f"enumerate_paths: capped at {len(paths)} paths {root}->{leaf} "
            "(more exist; kept the fewest-hop routes)")
    return paths


def is_reaction_edge(
    a: str, b: str, parking_links: Iterable[tuple[str, str]], root: str
) -> bool:
    """True for a genuine chemical (kinetic) step -- not a declared
    supply/parking link, and not one where n_H_rel changes (a PCET, even on
    the rare chance it isn't declared as a link)."""
    if (a, b) in set(parking_links):
        return False
    return n_H_rel(b, root) == n_H_rel(a, root)


_MAX_SINK_HOPS = 64  # defensive cap on the parked-branch follow-to-sink walk


def _follow_to_sink(
    adj: dict[str, list[str]], start: str, avoid: set[str]
) -> str | None:
    """Follow a single-successor chain from ``start`` to its terminal sink (a
    node with no outgoing edges). ``None`` if the branch forks (>=2
    successors -- a genuine competing-reaction dead-end, e.g. an associative
    fork, not a bookkeeping parking branch), cycles, rejoins ``avoid`` (the
    root->target path), or runs past :data:`_MAX_SINK_HOPS`."""
    seen = {start}
    node = start
    for _ in range(_MAX_SINK_HOPS):
        if node in avoid:
            return None  # rejoins the main path -- not a separate leaf
        nxt = adj.get(node, [])
        if not nxt:
            return node  # terminal sink
        if len(nxt) > 1:
            return None  # forks -- ambiguous, not a single required sink
        node = nxt[0]
        if node in seen:
            return None  # cycle
        seen.add(node)
    return None  # ran past the hop cap -- unresolved, treat as ambiguous


def required_leaves(
    edges: Iterable[tuple[str, str]],
    parking_links: Iterable[tuple[str, str]],
    root: str,
    target: str,
    log=lambda *a, **k: None,
) -> set[str]:
    """The required-leaf set for the DAG span/limiting-potential objective:
    ``{target}`` plus the terminal sink of every genuinely *parked* branch --
    a declared supply/parking link that diverges from the root->target path
    (every root fragment must end SOMEWHERE, e.g. the ammonia network's O
    atom parking off to the water sink). A diverging branch that forks into
    >=2 dead ends (a genuine competing-*reaction* fork, e.g. the associative
    HNO/NOH branch) is ambiguous -- not a single bookkeeping "this atom
    parked here" sink -- and is skipped (logged), same as a branch with no
    sink at all (cycle / too long). ``{target}`` alone if the root->target
    path itself can't be found.
    """
    edges = list(edges)
    path = reaction_path(edges, root, target)
    leaves = {target}
    if not path:
        log(f"required_leaves: no root->target path ({root} -> {target}); "
            "falling back to {target}")
        return leaves
    on_path_nodes = set(path)
    on_path_pairs = set(zip(path, path[1:]))

    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)

    for a, b in parking_links:
        if a not in on_path_nodes or (a, b) in on_path_pairs:
            continue  # not a divergence from the target path
        sink = _follow_to_sink(adj, b, on_path_nodes)
        if sink is None:
            log(f"required_leaves: parking branch {a}->{b} has no unambiguous "
                "sink (fork/cycle/too long); skipping (falls back to target-only "
                "for this branch)")
            continue
        leaves.add(sink)
    return leaves


# --- span-vs-U -------------------------------------------------------------

@dataclass(frozen=True)
class AffineU:
    """A quantity affine in the applied potential U: ``value(U) = intercept +
    slope*U``."""

    intercept: float
    slope: float

    def __call__(self, u: float) -> float:
        return self.intercept + self.slope * u


def _state_levels(
    path: list[str], node_energy0: dict[str, float], root: str
) -> list[AffineU]:
    return [AffineU(node_energy0[s], n_H_rel(s, root)) for s in path]


def _ts_levels(
    path: list[str],
    state_levels: list[AffineU],
    edge_barrier: dict[tuple[str, str], float],
) -> list[AffineU]:
    """Cumulative TS height per step, folded in as a "state" for the span --
    same construction as ``catpath.precis.analysis.energetic_span``
    (``ts_energy = state_e[i] + barrier(i->i+1)``; barriers are U-independent
    under thermodynamic-only CHE, so a TS level keeps its reactant's slope)."""
    out = []
    for i in range(len(path) - 1):
        ea = edge_barrier.get((path[i], path[i + 1]), 0.0)
        sl = state_levels[i]
        out.append(AffineU(sl.intercept + ea, sl.slope))
    return out


def energetic_span_u(
    path: list[str],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
    u: float,
) -> float:
    """The energetic span (Kozuch-Shaik: max climb from any state to any
    *later* TS along the path) at a fixed potential U -- the same algorithm as
    ``catpath.precis.analysis.energetic_span``, generalized so each level is
    evaluated at U via ``G_node(U) = G_node(0) + n_H_rel(node)*U``."""
    if len(path) < 2:
        return 0.0
    state_levels = _state_levels(path, node_energy0, root)
    ts_levels = _ts_levels(path, state_levels, edge_barrier)
    span = 0.0
    min_state = state_levels[0](u)
    for i in range(len(ts_levels)):
        ts = ts_levels[i](u)
        min_state = min(min_state, state_levels[i](u))
        span = max(span, ts - min_state)
    return span


def _path_diff_terms(
    path: list[str],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
) -> list[AffineU]:
    """The ``d_ik(U) = ts_i(U) - state_k(U)`` difference terms (``k<=i``) for
    one path, plus the implicit 0 floor -- exactly the terms
    ``energetic_span_u`` maximizes (``span(U) = max(0, max_{i,k<=i}
    d_ik(U))``: subtracting a running min over ``state_k, k<=i`` is the same
    as maxing over the per-k differences). Each ``d_ik`` is affine in U, so
    ``max`` of these is piecewise-linear convex, and its breakpoints are
    exactly the pairwise crossings of these terms (crossing two *levels*
    directly is a cheaper but unsound shortcut: two ``d_ik``/``d_jl`` terms
    sharing neither their ts nor their state component can cross at a U that
    is not a crossing of any two of the underlying per-state/per-TS levels).
    Shared building block for the single-path (:func:`optimal_span_u`) and
    multi-path/DAG (:func:`optimal_span_dag_u`) optimizers."""
    if len(path) < 2:
        return [AffineU(0.0, 0.0)]
    state_levels = _state_levels(path, node_energy0, root)
    ts_levels = _ts_levels(path, state_levels, edge_barrier)
    diffs: list[AffineU] = [AffineU(0.0, 0.0)]
    for i in range(len(ts_levels)):
        for k in range(i + 1):
            diffs.append(AffineU(
                ts_levels[i].intercept - state_levels[k].intercept,
                ts_levels[i].slope - state_levels[k].slope,
            ))
    return diffs


def _pairwise_crossings(
    terms: list[AffineU], window: tuple[float, float] | None
) -> set[float]:
    """Every U where two of ``terms`` cross (+ the window endpoints, if
    given) -- the candidate breakpoints of ``max(terms)`` (or of any
    finite min/max combination built from them: at any U the combination's
    value equals exactly one term, so every breakpoint is a crossing between
    the term that was active and the one that takes over)."""
    candidates: set[float] = set()
    if window is not None:
        candidates.add(window[0])
        candidates.add(window[1])
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            ti, tj = terms[i], terms[j]
            if ti.slope == tj.slope:
                continue  # parallel -- no crossing
            u_star = (tj.intercept - ti.intercept) / (ti.slope - tj.slope)
            if window is not None and not (window[0] <= u_star <= window[1]):
                continue
            candidates.add(u_star)
    if not candidates:
        candidates.add(0.0)  # degenerate: every term shares one slope (flat)
    return candidates


def optimal_span_u(
    path: list[str],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
    window: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """(U_opt, span_at_Uopt): a potential minimizing the energetic span.

    span(U) is a max of the :func:`_path_diff_terms` (affine in U) ->
    piecewise-linear convex, and its minimizer is a breakpoint of that
    envelope -- a :func:`_pairwise_crossings` candidate. Evaluate the true
    span at every candidate (+ the window endpoints, if given) and take the
    min. O(n^2) difference terms -> O(n^4) pairs, trivial at these graph
    sizes (~10 states).

    Along a path where n_H_rel is monotone (true root->target: it is
    non-decreasing along a reduction path), every term has slope >= 0, so
    span(U) is non-decreasing in U and its unconstrained minimum is attained
    on a half-line ``(-inf, U0]``, not at a unique point -- there is then no
    single "the" minimizer, and the ``U_opt`` returned is *a* potential
    attaining the minimum span, not necessarily ``U0``. Pass ``window`` (a
    plausible operating-potential range) for a physically meaningful
    ``U_opt`` when that degeneracy matters.
    """
    if len(path) < 2:
        return None, None
    diffs = _path_diff_terms(path, node_energy0, edge_barrier, root)
    candidates = _pairwise_crossings(diffs, window)

    best_u: float | None = None
    best_span: float | None = None
    for u in sorted(candidates):
        span = energetic_span_u(path, node_energy0, edge_barrier, root, u)
        if best_span is None or span < best_span:
            best_u, best_span = u, span
    return best_u, best_span


# --- span-vs-U over a DAG (all required leaves, easiest route to each) ------

_MAX_DAG_TERMS = 800  # guards the O(terms^2) pairwise-crossing search


def span_dag_u(
    leaf_paths: dict[str, list[list[str]]],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
    u: float,
) -> float:
    """span_dag(U) = MAX over required leaves of MIN over that leaf's
    enumerated root->leaf paths of the path's :func:`energetic_span_u`.

    Every required leaf (target + each parked fragment's sink -- see
    :func:`required_leaves`) must still be reached for turnover, so the
    worst of them governs the objective; but the mechanism is free to take
    whichever enumerated route to a given leaf is easiest, so within one
    leaf it's the best (min) of its routes. ``0.0`` if there are no leaves
    with any path.
    """
    per_leaf = [
        min(energetic_span_u(p, node_energy0, edge_barrier, root, u) for p in paths)
        for paths in leaf_paths.values() if paths
    ]
    return max(per_leaf) if per_leaf else 0.0


def optimal_span_dag_u(
    leaf_paths: dict[str, list[list[str]]],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
    window: tuple[float, float] | None = None,
    log=lambda *a, **k: None,
) -> tuple[float | None, float | None]:
    """(U_opt, span_at_Uopt) minimizing :func:`span_dag_u`.

    This is min-of-max over the required leaves -- piecewise-linear but NOT
    convex (a min of convex pieces generally isn't). The candidate-evaluation
    approach stays exact regardless: at any U, span_dag(U) equals exactly one
    underlying :func:`_path_diff_terms` difference term (whichever path/leaf
    combination is realizing the outer max/inner min at that instant), so
    every breakpoint of the whole nested structure is still a pairwise
    crossing between two of those terms -- the same fact
    :func:`_pairwise_crossings` relies on, independent of how the terms get
    combined. So: gather the union of difference terms across EVERY
    enumerated path for EVERY leaf, intersect pairwise (+ window endpoints),
    evaluate the true ``span_dag_u`` at each candidate, take the min.

    Guards the O(terms^2) pair count: if the union gets large, subsamples
    each leaf down to its fewest-hop paths (already the
    :func:`enumerate_paths` order) until back under budget, logging.
    ``(None, None)`` if no leaf has any path.
    """
    def _all_terms(lp: dict[str, list[list[str]]]) -> list[AffineU]:
        terms: list[AffineU] = []
        for paths in lp.values():
            for p in paths:
                terms.extend(_path_diff_terms(p, node_energy0, edge_barrier, root))
        return terms

    if not any(paths for paths in leaf_paths.values()):
        return None, None

    all_terms = _all_terms(leaf_paths)
    cap = max((len(paths) for paths in leaf_paths.values()), default=0)
    while len(all_terms) > _MAX_DAG_TERMS and cap > 1:
        cap = max(1, cap // 2)
        log(f"optimal_span_dag_u: {len(all_terms)} difference terms exceeds "
            f"the {_MAX_DAG_TERMS}-term budget; subsampling to the {cap} "
            "fewest-hop path(s) per leaf")
        leaf_paths = {leaf: paths[:cap] for leaf, paths in leaf_paths.items()}
        all_terms = _all_terms(leaf_paths)

    candidates = _pairwise_crossings(all_terms, window)
    best_u: float | None = None
    best_span: float | None = None
    for u in sorted(candidates):
        span = span_dag_u(leaf_paths, node_energy0, edge_barrier, root, u)
        if best_span is None or span < best_span:
            best_u, best_span = u, span
    return best_u, best_span


# --- pH ----------------------------------------------------------------

def u_rhe_to_she(u_rhe: float, ph: float, t: float = T_STANDARD) -> float:
    """Convert a potential from the RHE scale to SHE at a given pH
    (-0.0592 V/pH at 298.15 K) -- display/literature-comparison only; every
    PCET step is pH-independent on RHE, which is the point of that reference
    choice."""
    return u_rhe - (math.log(10) * K_B_EV * t) * ph


def u_she_to_rhe(u_she: float, ph: float, t: float = T_STANDARD) -> float:
    """Inverse of :func:`u_rhe_to_she`."""
    return u_she + (math.log(10) * K_B_EV * t) * ph


def decoupled_ph_shift(n_protons: float, ph: float, t: float = T_STANDARD) -> float:
    """Free-energy shift (eV) from a *decoupled* (non-PCET) proton/hydroxide
    transfer at a given pH.

    A decoupled proton transfer consumes H+ from solution, and
    mu(H+) = mu_std - ln10*kT*pH (protons get scarcer as pH rises), so
    consuming one costs dG = dG_std - mu(H+) shift = dG_std + ln10*kT*pH --
    a *positive* shift per net proton transferred at positive pH (and the
    opposite sign for a net hydroxide transfer, since OH- is produced by
    consuming water at low proton activity).

    ``n_protons``: net H+ transferred to the product minus OH- transferred
    (negative for a net hydroxide step). Dormant for the current templates
    (no decoupled step exists yet -- every H-supply step today is a PCET,
    pH-independent on RHE) but kept affine-safe: it does not depend on U, so
    adding it to a step's dG(0) leaves the U_L / span optimizers unchanged.
    """
    return n_protons * math.log(10) * K_B_EV * t * ph


# --- selectivity: guarded fork probabilities (P_side) -----------------------

def fork_branch_fraction(
    competitors: list[dict], taken: str, t: float = T_STANDARD
) -> float | None:
    """The kinetic branch fraction the ``taken`` product captures at one
    fork: ``exp(-Ea/kT)`` per competing edge with equal prefactors,
    normalized over ``competitors``.

    ``competitors``: every competing REACTION edge out of the fork state, as
    ``{"product": str, "barrier": float | None, "blocked": bool}`` -- caller's
    job to have already excluded supply/parking edges and to have resolved
    ``blocked`` (edge or endpoint flagged low_confidence / wrong-site /
    infeasible). ``None`` ("insufficient data") if any competitor lacks a
    computed barrier or is blocked -- never fabricate a ratio.
    U-independent: fork competitors here are chemical steps (barriers, not
    CHE-shifted by U).
    """
    for e in competitors:
        if e.get("barrier") is None or e.get("blocked"):
            return None
    weights = {e["product"]: math.exp(-e["barrier"] / (K_B_EV * t)) for e in competitors}
    total = sum(weights.values())
    if total <= 0 or taken not in weights:
        return None
    return weights[taken] / total


def p_side(
    target_path: list[str],
    forks: dict[str, list[dict]],
    t: float = T_STANDARD,
) -> float | None:
    """``P_side = 1 - P(main)``: the probability the mechanism is NOT the
    target path, from the kinetic branch fraction (:func:`fork_branch_fraction`)
    at every fork state along ``target_path``.

    ``forks``: fork state -> its competing REACTION edges (already filtered
    to exclude supply/parking edges; states with <2 aren't forks and don't
    need an entry). ``None`` ("insufficient data") if ANY fork along the path
    is missing its guard (a competitor lacks a barrier, or is flagged) --
    the whole selectivity estimate is only as good as its shakiest fork.
    """
    p_main = 1.0
    for a, b in zip(target_path, target_path[1:]):
        competitors = forks.get(a, [])
        if len(competitors) < 2:
            continue
        frac = fork_branch_fraction(competitors, taken=b, t=t)
        if frac is None:
            return None
        p_main *= frac
    return 1.0 - p_main
