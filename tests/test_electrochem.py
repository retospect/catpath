"""Tests for the CHE post-processing lever (catpath.electrochem).

See docs/proposals/pathway-potential-lever.md (slice 1, precis-mcp repo) for
the design of record. No relax/NEB calls here -- everything is arithmetic
over synthetic energies.
"""

from __future__ import annotations

import math

import pytest

from catpath import electrochem as ec


# --- n_H parsing ------------------------------------------------------------

def test_n_H_parses_fragment_formulas():
    assert ec.n_H("NO+H") == 1
    assert ec.n_H("HNO") == 1
    assert ec.n_H("NH2+OH") == 3
    assert ec.n_H("NO") == 0
    assert ec.n_H("H2O") == 2
    assert ec.n_H("N+O") == 0
    assert ec.n_H("N+H") == 1


def test_n_H_strips_site_isomer_suffix():
    # 'NO@top' is a site isomer, not a different formula.
    assert ec.n_H("NO@top") == 0
    assert ec.n_H("NH2@top") == 2


def test_n_H_rel_is_relative_to_root():
    assert ec.n_H_rel("NO+H", "NO") == 1
    assert ec.n_H_rel("NO", "NO") == 0
    # a root that already carries H still works (spec decision)
    assert ec.n_H_rel("NH+H", "NH") == 1
    assert ec.n_H_rel("NH", "NH") == 0
    assert ec.n_H_rel("NH", "NH2") == -1


# --- limiting potential ------------------------------------------------

def test_limiting_potential_tiny_network():
    # A -(+H*)-> A+H -(chemical)-> B -(+H*)-> B+H
    node_energy0 = {"A": 0.0, "A+H": 0.3, "B": 0.1, "B+H": 0.5}
    edges = [("A", "A+H"), ("A+H", "B"), ("B", "B+H")]
    # dG(0) of the two PCET steps: 0.3-0.0=0.3 and 0.5-0.1=0.4 -> U_L=-0.4
    assert ec.limiting_potential(node_energy0, edges, root="A") == pytest.approx(-0.4)


def test_limiting_potential_none_without_a_pcet_step():
    node_energy0 = {"A": 0.0, "B": 0.2}
    edges = [("A", "B")]  # no H count change -> not electrochemical
    assert ec.limiting_potential(node_energy0, edges, root="A") is None


def test_electrochemical_steps_filters_by_n_H_increase():
    edges = [("NO", "NO+H"), ("NO", "NO+O"), ("N+H", "NH")]
    assert ec.electrochemical_steps(edges, root="NO") == [("NO", "NO+H")]


# --- span-vs-U (hand-computable 3-state path) --------------------------
#
# N -(+H*, no barrier)-> N+H -(chemical, Ea=0.5)-> NH
#   n_H_rel:  N=0             N+H=1                   NH=1 (chemical conserves H)
#   G(0):     N=0.0           N+H=0.3                 NH=0.1
#
# Levels (intercept, slope): L_N=(0,0)  L_NH1=(0.3,1)  L_NH2=(0.1,1)
# TS(N->N+H)  = L_N   + 0   = (0,0)
# TS(N+H->NH) = L_NH1 + 0.5 = (0.8,1)
#
# span(u) = 0.8+u for u >= -0.3 (running min stays at L_N=0)
#         = 0.5    for u <= -0.3 (running min switches to L_NH1, giving the
#                    bare barrier Ea=0.5, U-independent since N+H and its TS
#                    share a slope) -- global min is 0.5, attained on the
#                    whole ray u <= -0.3.

_PATH = ["N", "N+H", "NH"]
_E0 = {"N": 0.0, "N+H": 0.3, "NH": 0.1}
_BARRIER = {("N", "N+H"): 0.0, ("N+H", "NH"): 0.5}


def test_energetic_span_u_matches_hand_computation():
    assert ec.energetic_span_u(_PATH, _E0, _BARRIER, "N", 0.0) == pytest.approx(0.8)
    assert ec.energetic_span_u(_PATH, _E0, _BARRIER, "N", -0.3) == pytest.approx(0.5)
    assert ec.energetic_span_u(_PATH, _E0, _BARRIER, "N", -1.0) == pytest.approx(0.5)
    assert ec.energetic_span_u(_PATH, _E0, _BARRIER, "N", 1.0) == pytest.approx(1.8)


def test_optimal_span_u_finds_the_exact_minimum():
    u_opt, span_opt = ec.optimal_span_u(_PATH, _E0, _BARRIER, "N")
    assert span_opt == pytest.approx(0.5)
    assert u_opt <= -0.3 + 1e-9  # anywhere on the degenerate minimizing ray


def test_optimal_span_u_respects_a_window():
    # within [0, 2] the span is monotone increasing (0.8+u regime only) -> the
    # window-clamped optimum sits at the low endpoint.
    u_opt, span_opt = ec.optimal_span_u(_PATH, _E0, _BARRIER, "N", window=(0.0, 2.0))
    assert u_opt == pytest.approx(0.0)
    assert span_opt == pytest.approx(0.8)


def test_optimal_span_u_short_path_is_none():
    assert ec.optimal_span_u(["N"], _E0, _BARRIER, "N") == (None, None)


def test_optimal_span_u_finds_crossings_between_disjoint_difference_terms():
    # A 4-state path (root + 2 supply steps around 1 chemical step) with
    # slopes 0, 1, 1, 2. The energetic span is max(0, max_{i,k<=i} d_ik(U))
    # for d_ik = ts_i - state_k; d_{1,0} (ts of edge 1, state 0) and d_{2,1}
    # (ts of edge 2, state 1) share neither their ts nor their state
    # component -- their crossing is generally NOT a crossing of any two of
    # the underlying per-state/per-TS *levels*, so a level-crossing-only
    # candidate set can miss it (only the explicit d_ik intersections find
    # it). Regression for exactly that.
    path = ["N", "N+H", "NH", "NH+H"]
    e0 = {"N": 0.0, "N+H": 0.35, "NH": 0.10, "NH+H": 0.55}
    barrier = {("N", "N+H"): 0.0, ("N+H", "NH"): 0.6, ("NH", "NH+H"): 0.0}

    u_opt, span_opt = ec.optimal_span_u(path, e0, barrier, "N", window=(-5.0, 5.0))
    assert u_opt is not None and span_opt is not None

    # the found minimizer is genuinely NOT a crossing of any two of the
    # per-state/per-TS levels -- i.e. it only exists in the (correct) d_ik
    # difference-term candidate set this fix adds, not in the old
    # level-crossing-only candidate set. (Its span value happens to still be
    # reachable by the old candidate set too, via a different, degenerate
    # equal-value point further out -- see the module's non-decreasing-span
    # note -- so this is a candidate-set/necessity regression, not a
    # numerically-observed value regression; the grid cross-check below is
    # the actual correctness guarantee.)
    state_levels = ec._state_levels(path, e0, "N")
    ts_levels = ec._ts_levels(path, state_levels, barrier)
    levels = state_levels + ts_levels
    level_crossings = set()
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            li, lj = levels[i], levels[j]
            if li.slope == lj.slope:
                continue
            level_crossings.add(round((lj.intercept - li.intercept) / (li.slope - lj.slope), 6))
    assert round(u_opt, 6) not in level_crossings

    # brute-force cross-check: a fine grid over the same window must never
    # find a strictly lower span than the closed-form optimum reports -- if
    # the candidate set missed the true minimizer, this would catch it.
    grid_min = min(ec.energetic_span_u(path, e0, barrier, "N", u / 1000.0)
                   for u in range(-5000, 5001))
    assert span_opt <= grid_min + 1e-6
    assert span_opt == pytest.approx(grid_min, abs=2e-3)


# --- RHE <-> SHE ---------------------------------------------------------

def test_rhe_to_she_298K_is_minus_0p0592_per_pH():
    u_she = ec.u_rhe_to_she(0.0, ph=1.0)
    assert u_she == pytest.approx(-0.0592, abs=2e-4)
    assert ec.u_rhe_to_she(0.5, ph=0.0) == pytest.approx(0.5)  # pH=0 -> no shift


def test_she_to_rhe_is_the_inverse():
    u_rhe = 0.2
    ph = 4.5
    u_she = ec.u_rhe_to_she(u_rhe, ph)
    assert ec.u_she_to_rhe(u_she, ph) == pytest.approx(u_rhe, abs=1e-9)


def test_decoupled_ph_shift_is_dormant_safe():
    assert ec.decoupled_ph_shift(2, ph=0.0) == 0.0
    # consuming a proton from solution costs MORE at higher pH (protons are
    # scarcer) -> a net proton transfer to the product is a positive shift;
    # a net hydroxide transfer (n_protons < 0) is the opposite sign.
    assert ec.decoupled_ph_shift(1, ph=7.0) > 0.0
    assert ec.decoupled_ph_shift(-1, ph=7.0) < 0.0


# --- integration: ammonia template network, synthetic energies -----------

def test_electrochemistry_integration_ammonia_template():
    from catpath.config import Config, ElectrochemistryConfig, SlabConfig
    from catpath.graph import build_graph
    from catpath.network import build_network
    from catpath.pipeline import Results, _apply_electrochemistry
    from catpath.uncertainty import Estimate

    net = build_network(SlabConfig(), kind="ammonia")
    root = net.order()[0]
    assert root == "NO"

    def synth_energy(name: str) -> float:
        # a downhill-ish synthetic profile (cheaper per absorbed H) plus a
        # small perturbation so it is not perfectly flat.
        return -0.2 * ec.n_H_rel(name, root) + 0.05 * len(name.split("+"))

    node_energies = {
        name: Estimate(mean=synth_energy(name), std=0.01, n=3, values=[])
        for name in net.states()
    }
    edges = [
        {"name": s.name, "reactant": s.reactant.name, "product": s.product.name,
         "barrier": Estimate(mean=0.4, std=0.0, n=3, values=[]),
         "delta_e": Estimate(mean=0.0, std=0.0, n=3, values=[])}
        for s in net.steps
    ]
    results = Results(node_energies=node_energies, edges=edges,
                      pathway=net.order(), links=list(net.links))

    ref = node_energies[root].mean
    all_edges = list(edges)
    for a, b in results.links:
        if not any(e["reactant"] == a and e["product"] == b for e in all_edges):
            all_edges.append({"name": f"{a}->{b}", "reactant": a, "product": b,
                              "kind": "supply"})
    g = build_graph(node_energies, all_edges, energy_ref=ref)

    cfg = Config(name="che_itest", target="NH3")
    cfg.electrochemistry = ElectrochemistryConfig(U_vs_RHE="optimal")

    che = _apply_electrochemistry(g, cfg, results, log=lambda *a, **k: None)

    assert che is not None
    for n, data in g.nodes(data=True):
        assert data["n_H"] == ec.n_H_rel(n, root)

    assert che["U_L"] is not None and math.isfinite(che["U_L"])
    assert che["U_opt"] is not None and math.isfinite(che["U_opt"])
    assert che["span_at_UL"] is not None and che["span_at_UL"] >= 0.0
    assert che["span_at_Uopt"] is not None and che["span_at_Uopt"] >= 0.0
    # U_opt minimizes span by construction -> never worse than at U_L
    assert che["span_at_Uopt"] <= che["span_at_UL"] + 1e-9


def test_electrochemistry_absent_config_is_a_no_op():
    from catpath.config import Config, SlabConfig
    from catpath.graph import build_graph
    from catpath.network import build_network
    from catpath.pipeline import Results, _apply_electrochemistry
    from catpath.uncertainty import Estimate

    net = build_network(SlabConfig(), kind="oxidation")
    node_energies = {name: Estimate(mean=0.0, std=0.0, n=1, values=[])
                     for name in net.states()}
    edges = [{"name": s.name, "reactant": s.reactant.name, "product": s.product.name,
             "barrier": Estimate(mean=0.1, std=0.0, n=1, values=[]),
             "delta_e": Estimate(mean=0.0, std=0.0, n=1, values=[])}
            for s in net.steps]
    results = Results(node_energies=node_energies, edges=edges,
                      pathway=net.order(), links=list(net.links))
    g = build_graph(node_energies, edges, energy_ref=0.0)

    cfg = Config(name="no_che")  # electrochemistry left at its default None
    assert _apply_electrochemistry(g, cfg, results) is None
    assert "n_H" not in next(iter(g.nodes(data=True)))[1]
