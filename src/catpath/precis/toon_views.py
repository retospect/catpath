"""TOON renderings of pathway data — the LLM-facing tables.

Uses precis's own ``format.toon`` serialiser (braced ``{col⇥col}`` header,
TAB-separated homogeneous rows) so pathway output reads exactly like ``search``.
Numbers are pre-formatted to strings (2 dp) — TOON renders floats via ``repr``
otherwise, which spends tokens on ``0.7400000001``.

Imports precis, so this module is handler-side (not part of the precis-free
``runner``/``analysis``/``text_views`` set).
"""

from __future__ import annotations

from typing import Any

from precis.format import toon

from . import analysis


def _e(x: Any) -> str:
    return "" if x is None else f"{float(x):+.2f}"  # signed (relative energies)


def _b(x: Any) -> str:
    return "" if x is None else f"{float(x):.2f}"  # barriers (positive)


def _conf(low: Any) -> str:
    return "low" if low else "ok"


def _roots(meta: dict[str, Any]) -> tuple[str, str]:
    return analysis.roots(meta.get("graph") or {}, meta.get("results", {}))


# ── single-pathway tables ───────────────────────────────────────────────
def intermediates_toon(meta: dict[str, Any]) -> str:
    graph = meta.get("graph") or {}
    nm = {n["id"]: n for n in graph.get("nodes", [])}
    order = meta.get("results", {}).get("pathway", list(nm))
    rows = [
        {
            "state": s,
            "rel_eV": _e(nm.get(s, {}).get("rel_energy")),
            "std": _b(nm.get(s, {}).get("energy_std")),
            "conf": _conf(nm.get(s, {}).get("low_confidence")),
        }
        for s in order
        if s in nm
    ]
    return toon.dump(rows, schema=["state", "rel_eV", "std", "conf"])


def steps_toon(meta: dict[str, Any]) -> str:
    graph = meta.get("graph") or {}
    rows = [
        {
            "reaction": f'{e["source"]}→{e["target"]}',
            "Ea_eV": _b(e.get("barrier")),
            "std": _b(e.get("barrier_std")),
            "dE_eV": _e(e.get("delta_e")),
            "conf": _conf(e.get("low_confidence")),
        }
        for e in analysis._reaction_edges(graph)
    ]
    return toon.dump(rows, schema=["reaction", "Ea_eV", "std", "dE_eV", "conf"])


def warnings_toon(meta: dict[str, Any]) -> str:
    warns = meta.get("warnings") or []
    if not warns:
        return "no warnings — states/barriers converged and within tolerance."
    rows = [{"warning": w} for w in warns]
    return toon.dump(rows, schema=["warning"])


def analysis_text(meta: dict[str, Any]) -> str:
    graph = meta.get("graph") or {}
    root, target = _roots(meta)
    r = meta.get("results", {})
    n = r.get("n_samples", "?")
    models = ",".join(r.get("models", [])) or r.get("backend", "?")

    rl = analysis.rate_limiting(graph, root, target)
    span = analysis.energetic_span(graph, root, target)
    head = [f"{root} → {target}  ({models}, {n} samples)", ""]
    if rl:
        flag = "  [LOW CONFIDENCE]" if rl["low_confidence"] else ""
        head.append(
            f"rate-limiting: {rl['step']}   Ea = {_b(rl['ea'])} ± {_b(rl['std'])} eV{flag}"
        )
        if rl["low_confidence"]:
            head.append("  → spread exceeds tolerance; escalate this step's fidelity "
                        "before trusting it.")
    if span is not None:
        head.append(f"energetic span (whole-path apparent barrier): {_b(span)} eV")
    head.append("")

    ranked = analysis.barriers_ranked(graph)
    brows = [
        {"reaction": s["reaction"], "Ea_eV": _b(s["ea"]), "std": _b(s["std"]), "conf": s["conf"]}
        for s in ranked
    ]
    head.append("barriers (descending):")
    head.append(toon.dump(brows, schema=["reaction", "Ea_eV", "std", "conf"]))

    sel = analysis.selectivity(graph, root, target)
    if len(sel) > 1:  # only meaningful when the root branches
        srows = [
            {
                "entry_step": s["entry_step"],
                "entry_Ea": _b(s["entry_ea"]),
                "role": "target-path" if s["on_target_path"] else "competing",
            }
            for s in sel
        ]
        head += ["", "selectivity (first steps out of root, lowest entry wins):",
                 toon.dump(srows, schema=["entry_step", "entry_Ea", "role"])]
    return "\n".join(head)


# ── cross-candidate compare (interleaved profile) ───────────────────────
def compare_toon(candidates: list[dict[str, Any]]) -> str:
    """`candidates`: [{slug, lever, graph, root, target}]. Rows = candidates.
    When they share a network, columns interleave state(rel eV) + ‡(barrier Eₐ)
    along the reaction coordinate; always: RATE (max step), SPAN, conf. Sorted
    by RATE ascending (best first)."""
    if not candidates:
        return "no computed candidates to compare."

    profiles = []
    for c in candidates:
        path, cols = analysis.profile_positions(c["graph"], c["root"], c["target"])
        summ = analysis.summarize(c["graph"], c["root"], c["target"])
        profiles.append({"c": c, "path": path, "cols": cols, "summ": summ})

    paths = {tuple(p["path"]) for p in profiles}
    aligned = len(paths) == 1 and all(p["path"] for p in profiles)

    def _row_scalars(p: dict[str, Any]) -> dict[str, Any]:
        rl = p["summ"]["rate_limiting"] or {}
        return {
            "cand": p["c"]["slug"],
            "lever": p["c"].get("lever", ""),
            "RATE": _b(rl.get("ea")),
            "SPAN": _b(p["summ"]["span"]),
            "conf": _conf(rl.get("low_confidence")),
        }

    if not aligned:
        rows = [_row_scalars(p) for p in profiles]
        rows.sort(key=lambda r: (r["RATE"] == "", r["RATE"]))
        note = "# networks differ — scalar comparison only (RATE = rate-limiting Eₐ)\n"
        return note + toon.dump(rows, schema=["cand", "lever", "RATE", "SPAN", "conf"])

    # aligned: build interleaved columns from the shared coordinate.
    template = profiles[0]["cols"]
    legend, col_names = [], []
    for col in template:
        if col["kind"] == "state":
            col_names.append(col["label"])
        else:
            col_names.append(col["pos"])
            legend.append(f'{col["pos"]} {col["label"]}')
    schema = ["cand", "lever", *col_names, "RATE", "SPAN", "conf"]

    rows = []
    for p in profiles:
        row = _row_scalars(p)
        for col, name in zip(p["cols"], col_names):
            row[name] = _e(col["value"]) if col["kind"] == "state" else _b(col["value"])
        rows.append(row)
    rows.sort(key=lambda r: (r["RATE"] == "", r["RATE"]))

    head = "# ‡ = step barrier Eₐ; state cols = rel eV vs root.  " + "  ".join(legend)
    return head + "\n" + toon.dump(rows, schema=schema)
