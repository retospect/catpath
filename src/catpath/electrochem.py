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


def optimal_span_u(
    path: list[str],
    node_energy0: dict[str, float],
    edge_barrier: dict[tuple[str, str], float],
    root: str,
    window: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """(U_opt, span_at_Uopt): a potential minimizing the energetic span.

    ``energetic_span_u`` computes ``span(U) = max(0, max_{i, k<=i} d_ik(U))``
    where ``d_ik(U) = ts_i(U) - state_k(U)`` (``ts_i`` is step i's cumulative
    TS height, ``state_k`` the k-th state visited so far along the path --
    exactly the running-min construction ``energetic_span_u`` /
    ``catpath.precis.analysis.energetic_span`` use: subtracting a running min
    over ``state_k, k<=i`` is the same as maxing over the per-k differences).
    Each ``d_ik`` is affine in U, so span(U) is a max of affine functions ->
    piecewise-linear convex, and its minimizer is a breakpoint of that
    envelope.

    Breakpoints occur where two *active* ``d_ik`` lines cross. Two such lines
    that share neither their ``ts_i``/``ts_j`` nor their ``state_k``/
    ``state_l`` component can cross at a U that is **not** a crossing of any
    pair of the underlying per-state/per-TS *levels* -- so this builds the
    ``d_ik`` difference terms explicitly (same k<=i enumeration as the span
    itself, plus the implicit 0 floor) and intersects THOSE, rather than the
    cheaper but unsound shortcut of intersecting the levels directly. O(n^2)
    difference terms -> O(n^4) pairs, trivial at these graph sizes (~10
    states). Evaluate the true span at every candidate (+ the window
    endpoints, if a window is given) and take the min.

    Along a path where n_H_rel is monotone (true root->target: it is
    non-decreasing along a reduction path), every ``d_ik`` has slope >= 0, so
    span(U) is non-decreasing in U and its unconstrained minimum is attained
    on a half-line ``(-inf, U0]``, not at a unique point -- there is then no
    single "the" minimizer, and the ``U_opt`` returned is *a* potential
    attaining the minimum span, not necessarily ``U0``. Pass ``window`` (a
    plausible operating-potential range) for a physically meaningful
    ``U_opt`` when that degeneracy matters.
    """
    if len(path) < 2:
        return None, None
    state_levels = _state_levels(path, node_energy0, root)
    ts_levels = _ts_levels(path, state_levels, edge_barrier)

    # d_ik(U) = ts_i(U) - state_k(U) for k <= i -- exactly the terms
    # energetic_span_u maximizes (against an implicit 0 floor, included below
    # as its own flat term so a crossing into that floor is a candidate too).
    diffs: list[AffineU] = [AffineU(0.0, 0.0)]
    for i in range(len(ts_levels)):
        for k in range(i + 1):
            diffs.append(AffineU(
                ts_levels[i].intercept - state_levels[k].intercept,
                ts_levels[i].slope - state_levels[k].slope,
            ))

    candidates: set[float] = set()
    if window is not None:
        candidates.add(window[0])
        candidates.add(window[1])
    for i in range(len(diffs)):
        for j in range(i + 1, len(diffs)):
            di, dj = diffs[i], diffs[j]
            if di.slope == dj.slope:
                continue  # parallel -- no crossing
            u_star = (dj.intercept - di.intercept) / (di.slope - dj.slope)
            if window is not None and not (window[0] <= u_star <= window[1]):
                continue
            candidates.add(u_star)
    if not candidates:
        candidates.add(0.0)  # degenerate: every term shares one slope (flat span)

    best_u: float | None = None
    best_span: float | None = None
    for u in sorted(candidates):
        span = energetic_span_u(path, node_energy0, edge_barrier, root, u)
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
