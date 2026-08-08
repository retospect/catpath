"""Tests for the CHE post-processing lever (autocatpath.electrochem).

See docs/proposals/pathway-potential-lever.md (slice 1, precis-mcp repo) for
the design of record. No relax/NEB calls here -- everything is arithmetic
over synthetic energies.
"""

from __future__ import annotations

import math

import pytest

from autocatpath import electrochem as ec


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


def test_limiting_potential_full_dag_picks_up_an_off_target_path_step():
    # A -(+H*, dG=0.1)-> A+H -(chemical)-> AB      [target path]
    # A -(+H*, dG=0.5)-> X+H                        [off-target-path branch]
    node_energy0 = {"A": 0.0, "A+H": 0.1, "AB": 0.05, "X+H": 0.5}
    edges = [("A", "A+H"), ("A", "X+H"), ("A+H", "AB")]

    target_path = ec.reaction_path(edges, "A", "AB")
    assert target_path == ["A", "A+H", "AB"]
    path_edges = list(zip(target_path, target_path[1:]))
    u_l_path_only = ec.limiting_potential(node_energy0, path_edges, "A")
    assert u_l_path_only == pytest.approx(-0.1)

    # full-DAG U_L must also see the off-path A->X+H step (dG=0.5, the real
    # constraint -- both branches must proceed for turnover).
    reachable = ec.reachable_from(edges, "A")
    reachable_edges = [(a, b) for a, b in edges if a in reachable]
    u_l_dag = ec.limiting_potential(node_energy0, reachable_edges, "A")
    assert u_l_dag == pytest.approx(-0.5)
    assert u_l_dag < u_l_path_only


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


# --- required leaves (root fragments must all end somewhere) ---------------

def test_required_leaves_ammonia_template():
    from autocatpath.config import SlabConfig
    from autocatpath.network import build_network

    net = build_network(SlabConfig(), kind="ammonia")
    root = net.order()[0]
    assert root == "NO"
    all_edges = [(s.reactant.name, s.product.name) for s in net.steps] + list(net.links)

    # target NH3: the O atom (from the N/O dissociation) parks off to its own
    # H2O sink via the (N+O -> O+H) link -- both fragments must land.
    leaves = ec.required_leaves(all_edges, net.links, root, "NH3")
    assert leaves == {"NH3", "H2O"}


def test_required_leaves_falls_back_to_target_on_ambiguous_branch():
    # A parking link into a branch that FORKS into two dead ends (a genuine
    # competing-reaction fork, not a single bookkeeping sink) is skipped.
    edges = [("A", "B"), ("A", "X"), ("X", "Y1"), ("X", "Y2")]
    parking_links = [("A", "B"), ("A", "X")]
    leaves = ec.required_leaves(edges, parking_links, "A", "B")
    assert leaves == {"B"}


# --- span-vs-U over a DAG (worst-of-required-leaves, easiest route each) ---

def test_span_dag_exceeds_target_span_when_the_parked_branch_is_worse():
    # target path N -> N+H -> NH has a small chemical barrier (0.2); the
    # parked branch N -> N+O -> OX has a much bigger one (0.9) -- both
    # fragments must still get somewhere, so the DAG objective must see it.
    node_energy0 = {"N": 0.0, "N+H": 0.1, "NH": 0.05, "N+O": 0.0, "OX": 0.0}
    edge_barrier = {("N", "N+H"): 0.0, ("N", "N+O"): 0.0,
                    ("N+H", "NH"): 0.2, ("N+O", "OX"): 0.9}
    edges = [("N", "N+H"), ("N", "N+O"), ("N+H", "NH"), ("N+O", "OX")]
    parking_links = [("N", "N+H"), ("N", "N+O")]

    target_path = ec.reaction_path(edges, "N", "NH")
    leaves = ec.required_leaves(edges, parking_links, "N", "NH")
    assert leaves == {"NH", "OX"}
    leaf_paths = {leaf: ec.enumerate_paths(edges, "N", leaf) for leaf in leaves}

    span_target = ec.energetic_span_u(target_path, node_energy0, edge_barrier, "N", 0.0)
    span_dag = ec.span_dag_u(leaf_paths, node_energy0, edge_barrier, "N", 0.0)
    assert span_target == pytest.approx(0.3)
    assert span_dag == pytest.approx(0.9)
    assert span_dag > span_target


def test_span_dag_takes_the_easier_of_two_routes_to_one_leaf():
    # two routes A->AH@1->AHZ (barrier 0.5) and A->AH@2->AHZ (barrier 0.1)
    # converge on the SAME leaf ("@..." is the existing site-isomer suffix,
    # stripped by n_H parsing, so both intermediates carry n_H=1) -- the
    # mechanism takes the easier (min) one, not the harder.
    node_energy0 = {"A": 0.0, "AH@1": 0.0, "AH@2": 0.0, "AHZ": 0.0}
    edge_barrier = {("A", "AH@1"): 0.0, ("A", "AH@2"): 0.0,
                    ("AH@1", "AHZ"): 0.5, ("AH@2", "AHZ"): 0.1}
    path1 = ["A", "AH@1", "AHZ"]
    path2 = ["A", "AH@2", "AHZ"]

    s1 = ec.energetic_span_u(path1, node_energy0, edge_barrier, "A", 0.0)
    s2 = ec.energetic_span_u(path2, node_energy0, edge_barrier, "A", 0.0)
    assert s1 == pytest.approx(0.5) and s2 == pytest.approx(0.1)

    span_dag = ec.span_dag_u({"AHZ": [path1, path2]}, node_energy0, edge_barrier, "A", 0.0)
    assert span_dag == pytest.approx(min(s1, s2))
    assert span_dag < max(s1, s2)


def test_optimal_span_dag_u_no_leaves_is_none():
    assert ec.optimal_span_dag_u({}, {}, {}, "A") == (None, None)
    assert ec.optimal_span_dag_u({"L": []}, {}, {}, "A") == (None, None)


# --- P_side: guarded fork probabilities -------------------------------------

def test_p_side_hand_computed_two_edge_fork():
    kT = ec.K_B_EV * ec.T_STANDARD
    forks = {"A": [
        {"product": "B", "barrier": 0.20, "blocked": False},
        {"product": "C", "barrier": 0.35, "blocked": False},
    ]}
    expected_p_main = math.exp(-0.20 / kT) / (math.exp(-0.20 / kT) + math.exp(-0.35 / kT))

    assert ec.fork_branch_fraction(forks["A"], taken="B") == pytest.approx(expected_p_main)
    assert ec.p_side(["A", "B"], forks) == pytest.approx(1.0 - expected_p_main)


def test_p_side_ignores_states_with_fewer_than_two_competitors():
    forks = {"A": [{"product": "B", "barrier": 0.2, "blocked": False}]}  # no real fork
    assert ec.p_side(["A", "B"], forks) == pytest.approx(0.0)


def test_p_side_none_when_a_competitor_barrier_missing():
    forks = {"A": [
        {"product": "B", "barrier": 0.2, "blocked": False},
        {"product": "C", "barrier": None, "blocked": False},
    ]}
    assert ec.fork_branch_fraction(forks["A"], taken="B") is None
    assert ec.p_side(["A", "B"], forks) is None


def test_p_side_none_when_a_competitor_is_blocked():
    forks = {"A": [
        {"product": "B", "barrier": 0.2, "blocked": False},
        {"product": "C", "barrier": 0.3, "blocked": True},  # low_confidence/wrong-site/etc.
    ]}
    assert ec.p_side(["A", "B"], forks) is None


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
    from autocatpath.config import Config, ElectrochemistryConfig, SlabConfig
    from autocatpath.graph import build_graph
    from autocatpath.network import build_network
    from autocatpath.pipeline import Results, _apply_electrochemistry
    from autocatpath.uncertainty import Estimate

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
    # U_opt minimizes span_dag by construction -> never worse than at U_L
    assert che["span_at_Uopt"] <= che["span_at_UL"] + 1e-9

    # target-path-only diagnostic figure, present alongside the DAG headline
    assert che["span_target_at_Uopt"] is not None and che["span_target_at_Uopt"] >= 0.0
    # every barrier here is the same constant (0.4 eV) with no low_confidence
    # flags, so P_side is fully guarded -- a real number in [0, 1], not None.
    assert che["P_side"] is not None and 0.0 <= che["P_side"] <= 1.0
    assert che["T"] == pytest.approx(298.15)


def test_electrochemistry_absent_config_is_a_no_op():
    from autocatpath.config import Config, SlabConfig
    from autocatpath.graph import build_graph
    from autocatpath.network import build_network
    from autocatpath.pipeline import Results, _apply_electrochemistry
    from autocatpath.uncertainty import Estimate

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
