"""Tests for the precis-mcp bridge (``catpath.precis``).

Two layers:

* the **pure runner** (``catpath.precis.runner``) — needs only catpath's own
  deps, so it always runs in CI;
* the **handler** (``catpath.precis.handler.PathwayHandler``) — needs
  ``precis-mcp`` installed *and* a test Postgres at ``PRECIS_TEST_PG_URL``,
  so it skips in a bare catpath checkout.
"""

from __future__ import annotations

import os
import pathlib

import pytest

SMOKE = """
name: bridge_smoke
substrate: "NO"
target: "NO3"
network: oxidation
slab: {element: Pd, size: [2, 2, 3], vacuum: 8.0, fix_layers: 1, relax_lattice: false}
mlip: {backend: emt}
search: {seeds: [0], neb_images: 3, neb_max_steps: 15, neb_retries: 0, max_steps: 40, pose_count: 2}
"""

_MIG = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src/catpath/precis/migrations/0001_pathway_kind.sql"
)


# --- pure runner: catpath-only, always runs -----------------------------
def test_runner_emt_smoke_and_determinism() -> None:
    from catpath.precis import runner

    art = runner.run_pathway_from_yaml(SMOKE)
    r = art["results_json"]
    assert r["backend"] == "emt"
    assert r["nodes"] and r["edges"], "network produced no states/steps"
    assert art["methods_md"].startswith("# Methods")
    assert art["graph_json"]["nodes"] and art["graph_json"]["links"]
    # every state's relaxed geometry is harvested for slice-1 ingest
    assert set(art["structures_extxyz"]) == set(r["nodes"])
    # deterministic content address for an unchanged config
    assert runner.run_pathway_from_yaml(SMOKE)["content_key"] == art["content_key"]


def test_content_key_discriminates_config() -> None:
    from catpath.precis import runner

    base = {"name": "x", "mlip": {"backend": "emt"}}
    assert runner.content_key(base) != runner.content_key(
        {"name": "x", "mlip": {"backend": "mace"}}
    )
    assert runner.content_key(base) == runner.content_key(dict(base))


def test_chem_safe_yaml_keeps_NO_a_string() -> None:
    # YAML 1.1 would coerce bare NO -> False; catpath's loader must not.
    from catpath.precis import runner

    art = runner.run_pathway_from_yaml(SMOKE)
    assert art["config"]["substrate"] == "NO"


def test_network_topology_and_mermaid_no_compute() -> None:
    # The "argue before you compute" surface: build the network (rule-based, no
    # ML) and render it as text + mermaid.
    from catpath.precis import runner
    from catpath.precis.text_views import (
        graph_to_mermaid,
        topology_to_mermaid,
        topology_to_text,
    )

    topo = runner.network_topology(_yaml_dict(SMOKE))
    assert topo["states"] and topo["steps"]
    assert "NO3" in {s["name"] for s in topo["states"]}  # target intermediate present
    assert all("composition" in s for s in topo["states"])

    mer = topology_to_mermaid(topo)
    assert mer.startswith("flowchart LR") and "-->" in mer
    txt = topology_to_text(topo)
    assert "Intermediates" in txt and "Elementary steps" in txt

    # a computed run renders with energies + barriers
    gmer = graph_to_mermaid(runner.run_pathway_from_yaml(SMOKE)["graph_json"])
    assert "Ea" in gmer and "eV" in gmer


# Branching network — connected (supply bridges), so it has a real root→target
# path and exercises the interleaved profile / compare. The linear `oxidation`
# SMOKE is disconnected (rate-limiting still works via max-over-edges fallback).
BRANCH = """
name: branch_smoke
substrate: "NO"
target: "NO3"
network: branching
slab: {element: Pd, size: [2, 2, 3], vacuum: 8.0, fix_layers: 1, relax_lattice: false}
mlip: {backend: emt}
search: {seeds: [0], neb_images: 3, neb_max_steps: 12, neb_retries: 0, max_steps: 30, pose_count: 2}
"""


def test_analysis_over_computed_graph() -> None:
    from catpath.precis import analysis, runner

    art = runner.run_pathway_from_yaml(BRANCH)
    g, res = art["graph_json"], art["results_json"]
    root, target = analysis.roots(g, res)
    assert root in {n["id"] for n in g["nodes"]}  # root is a real node, not the label

    rl = analysis.rate_limiting(g, root, target)
    assert rl and rl["ea"] is not None and "→" in rl["step"]

    span = analysis.energetic_span(g, root, target)
    assert span is not None and span >= 0.0

    ranked = analysis.barriers_ranked(g)
    eas = [r["ea"] for r in ranked]
    assert eas == sorted(eas, reverse=True)  # highest barrier first

    path, cols = analysis.profile_positions(g, root, target)
    assert path and cols[0]["kind"] == "state"
    assert any(c["kind"] == "ts" for c in cols)  # ≥1 barrier column on the path


def test_toon_views_and_aligned_compare() -> None:
    pytest.importorskip("precis")  # toon_views uses precis.format.toon
    from catpath.precis import analysis, runner, toon_views

    a1 = runner.run_pathway_from_yaml(BRANCH)
    meta = {"graph": a1["graph_json"], "results": a1["results_json"],
            "warnings": a1["warnings"]}
    assert toon_views.intermediates_toon(meta).startswith("{state")
    assert "Ea_eV" in toon_views.steps_toon(meta)
    ana = toon_views.analysis_text(meta)
    assert "rate-limiting" in ana and "barriers (descending)" in ana

    # aligned interleaved compare: two candidates sharing the branching network
    a2 = runner.run_pathway_from_yaml(BRANCH.replace("element: Pd", "element: Pt"))

    def _cand(slug: str, art: dict, el: str) -> dict:
        g, res = art["graph_json"], art["results_json"]
        r, t = analysis.roots(g, res)
        return {"slug": slug, "lever": el, "graph": g, "root": r, "target": t}

    out = toon_views.compare_toon([_cand("pd", a1, "Pd"), _cand("pt", a2, "Pt")])
    assert "RATE" in out and "SPAN" in out
    assert "‡" in out  # aligned → barrier columns present (not the scalar fallback)
    assert "pd" in out and "pt" in out


# --- handler: needs precis-mcp + a test DB ------------------------------
_DSN = os.environ.get("PRECIS_TEST_PG_URL")


def _apply_migration(store) -> None:
    lines = [ln for ln in _MIG.read_text().splitlines() if not ln.strip().startswith("--")]
    body = "\n".join(lines).replace("BEGIN;", "").replace("COMMIT;", "").strip()
    with store.tx() as conn:
        conn.execute(body)


def _yaml_dict(text: str) -> dict:
    from catpath.config import _load_yaml

    return _load_yaml(text)


class _FakeCtx:
    """Minimal precis DispatchContext stand-in for testing a job dispatcher."""

    def __init__(self, store, params) -> None:
        self.store = store
        self.meta = {"params": params}
        self.status = None
        self.failure = None
        self.events: list = []
        self.set_meta_kw: dict = {}

    def record_failure(self, msg) -> None:
        self.failure = msg

    def append_chunk(self, kind, text) -> None:
        self.events.append((kind, text))

    def set_meta(self, **kw) -> None:
        self.set_meta_kw.update(kw)

    def set_status(self, s) -> None:
        self.status = s


def test_pathway_skill_discoverable() -> None:
    pytest.importorskip("precis")  # skill served via the precis.skills entry point
    import precis.handlers.skill as sk

    sk._SKILLS_MAP_CACHE = None  # re-scan so the installed plugin root is seen
    body = sk._load_skills_map().get("precis-pathway-help", "")
    assert body and "view='compare'" in body and "rate-limiting" in body


@pytest.mark.skipif(not _DSN, reason="needs PRECIS_TEST_PG_URL + precis-mcp")
def test_catpath_explore_dispatch_writes_back() -> None:
    pytest.importorskip("precis")
    from precis.store import Store
    from precis.store.types import BlockInsert

    from catpath.precis import job, runner

    store = Store.connect(_DSN)
    try:
        _apply_migration(store)
        slug = "dispatch-test"
        if store.get_ref(kind="pathway", id=slug):
            store.soft_delete_ref(store.get_ref(kind="pathway", id=slug).id)
        eff = runner.effective_config(_yaml_dict(SMOKE), force_backend="emt")
        with store.tx() as c:
            ref = store.insert_ref(
                kind="pathway", slug=slug, title="t",
                meta={"content_key": runner.content_key(eff), "status": "computing"}, conn=c,
            )
            store.insert_blocks(
                ref.id, [BlockInsert(pos=0, text="placeholder",
                                     meta={"chunk_kind": "pathway_body"})], conn=c,
            )
        ctx = _FakeCtx(store, {"pathway_ref_id": ref.id, "config": _yaml_dict(SMOKE),
                               "force_backend": "emt", "target_node": "spark"})
        job._dispatch(ctx, job.SPEC)

        assert ctx.failure is None, ctx.failure
        assert ctx.status == "succeeded"
        got = store.get_ref(kind="pathway", id=slug)
        assert got.meta["results"]["nodes"], "results not written back"
        assert got.meta["produced_by"] == "catpath_explore"
        assert got.meta["ran_on"] == "spark"
        blocks = store.list_blocks_for_ref(got.id)
        assert blocks[0].text.startswith("# Methods")
        store.soft_delete_ref(ref.id)
    finally:
        store.close()


@pytest.mark.skipif(not _DSN, reason="needs PRECIS_TEST_PG_URL + precis-mcp")
def test_put_routes_to_pinned_node(monkeypatch) -> None:
    pytest.importorskip("precis")
    from precis.dispatch import Hub
    from precis.store import Store

    monkeypatch.setenv("PRECIS_CATPATH_ENABLED", "1")
    monkeypatch.setenv("PRECIS_CATPATH_ROUTE_NODE", "spark")
    from catpath.precis import PathwayHandler

    store = Store.connect(_DSN)
    try:
        _apply_migration(store)
        hub = Hub(store=store)
        h = PathwayHandler(hub=hub)
        h._register_with(hub)  # registers pathway → hub.kinds (can_own_jobs)

        slug = "route-test"
        if store.get_ref(kind="pathway", id=slug):
            h.delete(id=slug)

        r = h.put(id="route_test", text=SMOKE)
        assert "dispatched catpath compute" in r.body and "spark" in r.body, r.body

        ref = store.get_ref(kind="pathway", id=slug)
        assert ref.meta["status"] == "computing"
        assert ref.meta["route_node"] == "spark"

        # A catpath_explore job was minted, parented on the pathway ref (only
        # possible because pathway.can_own_jobs → JOB_PARENT_KINDS extension),
        # over ssh_node, pinned to spark.
        with store.pool.connection() as c:
            rows = c.execute(
                "SELECT ref_id, meta FROM refs WHERE kind='job' AND parent_id=%s "
                "AND deleted_at IS NULL",
                (ref.id,),
            ).fetchall()
        assert rows, "no catpath_explore job minted"
        job_id, jmeta = rows[0]
        assert jmeta["job_type"] == "catpath_explore"
        assert jmeta["executor"] == "ssh_node"
        assert jmeta["params"]["target_node"] == "spark"

        store.soft_delete_ref(job_id)
        store.soft_delete_ref(ref.id)
    finally:
        store.close()


@pytest.mark.skipif(not _DSN, reason="needs PRECIS_TEST_PG_URL + precis-mcp")
def test_handler_roundtrip() -> None:
    pytest.importorskip("precis")
    from precis.dispatch import Hub
    from precis.store import Store

    os.environ["PRECIS_CATPATH_ENABLED"] = "1"
    os.environ.pop("PRECIS_CATPATH_ROUTE_NODE", None)  # in-process path
    from catpath.precis import PathwayHandler

    store = Store.connect(_DSN)
    try:
        _apply_migration(store)
        hub = Hub(store=store)
        h = PathwayHandler(hub=hub)
        h._register_with(hub)

        slug = "bridge-smoke"
        h.delete(id=slug) if store.get_ref(kind="pathway", id=slug) else None

        put = h.put(id="bridge_smoke", text=SMOKE)
        assert f"created pathway '{slug}'" in put.body, put.body

        ref = store.get_ref(kind="pathway", id=slug)
        assert ref is not None and ref.meta["results"]["nodes"]
        assert ref.meta["backend_forced"] == "emt"
        blocks = store.list_blocks_for_ref(ref.id)
        assert blocks and blocks[0].text.startswith("# Methods")

        assert "States (relative energy" in h.get(id=slug, view="profile").body
        assert "Ea=" in h.get(id=slug, view="network").body
        assert "substrate" in h.get(id=slug, view="config").body

        # regen cache-hit
        assert "unchanged (cache hit" in h.put(id="bridge_smoke", text=SMOKE).body

        h.delete(id=slug)
        assert store.get_ref(kind="pathway", id=slug) is None
    finally:
        store.close()


@pytest.mark.skipif(not _DSN, reason="needs PRECIS_TEST_PG_URL + precis-mcp")
def test_preview_no_compute(monkeypatch) -> None:
    pytest.importorskip("precis")
    from precis.dispatch import Hub
    from precis.store import Store

    monkeypatch.setenv("PRECIS_CATPATH_ENABLED", "1")
    monkeypatch.delenv("PRECIS_CATPATH_ROUTE_NODE", raising=False)
    from catpath.precis import PathwayHandler

    store = Store.connect(_DSN)
    try:
        _apply_migration(store)
        hub = Hub(store=store)
        h = PathwayHandler(hub=hub)
        h._register_with(hub)
        slug = "preview-test"
        if store.get_ref(kind="pathway", id=slug):
            h.delete(id=slug)

        r = h.put(id="preview_test", text=SMOKE, mode="preview")
        assert "previewed" in r.body and "mermaid" in r.body, r.body

        ref = store.get_ref(kind="pathway", id=slug)
        assert ref.meta["status"] == "preview"
        assert ref.meta["topology"]["steps"], "topology not stored"
        # views render on a preview (no results yet)
        assert "flowchart" in h.get(id=slug, view="mermaid").body
        assert "Intermediates" in h.get(id=slug, view="intermediates").body

        h.delete(id=slug)
    finally:
        store.close()


@pytest.mark.skipif(not _DSN, reason="needs PRECIS_TEST_PG_URL + precis-mcp")
def test_compare_view(monkeypatch) -> None:
    pytest.importorskip("precis")
    from precis.dispatch import Hub
    from precis.store import Store

    monkeypatch.setenv("PRECIS_CATPATH_ENABLED", "1")
    monkeypatch.delenv("PRECIS_CATPATH_ROUTE_NODE", raising=False)
    from catpath.precis import PathwayHandler

    store = Store.connect(_DSN)
    try:
        _apply_migration(store)
        hub = Hub(store=store)
        h = PathwayHandler(hub=hub)
        h._register_with(hub)
        for slug in ("cmp-pd", "cmp-pt"):
            if store.get_ref(kind="pathway", id=slug):
                h.delete(id=slug)

        h.put(id="cmp_pd", text=BRANCH)  # element Pd, branching (connected)
        h.put(id="cmp_pt", text=BRANCH.replace("element: Pd", "element: Pt"))

        out = h.get(id="cmp-pd", view="compare").body
        assert "RATE" in out and "SPAN" in out, out
        assert "cmp-pd" in out and "cmp-pt" in out, out  # both candidates present

        h.delete(id="cmp-pd")
        h.delete(id="cmp-pt")
    finally:
        store.close()


@pytest.mark.skipif(not _DSN, reason="needs precis-mcp")
def test_handler_gated_off_by_default() -> None:
    pytest.importorskip("precis")
    from precis.dispatch import Hub, InitError
    from precis.store import Store

    os.environ.pop("PRECIS_CATPATH_ENABLED", None)
    from catpath.precis import PathwayHandler

    store = Store.connect(_DSN)
    try:
        with pytest.raises(InitError):
            PathwayHandler(hub=Hub(store=store))
    finally:
        store.close()
