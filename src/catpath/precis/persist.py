"""Persist a catpath run artifact onto a `pathway` ref.

Shared by the two write paths so they stay identical:

* the handler's **in-process** `put` (slice 0), and
* the **`catpath_explore`** job dispatch (slice 1, runs on the pinned node).

Imports only precis's public Store surface (no catpath deps), so it is cheap to
import inside the job dispatcher.
"""

from __future__ import annotations

from typing import Any

BODY_KIND = "pathway_body"


def pathway_meta(artifact: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `refs.meta` payload for a pathway ref: the authoritative config +
    snapshot, the reaction graph, the pooled-uncertainty results, warnings."""
    meta: dict[str, Any] = {
        "content_key": artifact["content_key"],
        "catpath_version": artifact["catpath_version"],
        "config": artifact["config"],
        "config_snapshot_yaml": artifact["config_snapshot_yaml"],
        "results": artifact["results_json"],
        "graph": artifact["graph_json"],
        "warnings": artifact["warnings"],
        "n_structures": len(artifact["structures_extxyz"]),
        "status": "ready",
    }
    if extra:
        meta.update(extra)
    return meta


def pathway_title(artifact: dict[str, Any]) -> str:
    r = artifact["results_json"]
    el = artifact["config"].get("slab", {}).get("element", "?")
    return f"{r['substrate']} → {r['target']} on {el}"


def persist_result(
    store: Any,
    ref_id: int,
    artifact: dict[str, Any],
    *,
    pathway_slug: str | None = None,
    ingest: bool = True,
    extra_meta: dict[str, Any] | None = None,
    conn: Any = None,
) -> None:
    """Stamp the pathway ref's meta and (re)write its methods body chunk, and
    (slice 1b) ingest each relaxed intermediate as a `structure` ref linked back
    to the pathway. The ref must already exist. Runs in its own transaction
    unless a `conn` is supplied.

    Structure ingest runs *before* the meta stamp (it opens its own
    transactions via `structure_save`) so the resulting `{state → ref_id}` map
    lands in the same meta. Best-effort — an ingest failure never blocks the
    core result write-back."""
    extra = dict(extra_meta or {})
    if ingest and pathway_slug and artifact.get("structures_extxyz"):
        try:
            from .ingest import ingest_intermediates

            extra["structure_refs"] = ingest_intermediates(
                store, ref_id, pathway_slug, artifact["content_key"],
                artifact["structures_extxyz"],
            )
        except Exception:
            pass  # native ingest is additive; keep the pathway result regardless

    meta = pathway_meta(artifact, extra=extra)

    def _do(c: Any) -> None:
        store.stamp_ref_meta(ref_id, meta, conn=c)
        store.replace_body_chunk(ref_id, artifact["methods_md"], chunk_kind=BODY_KIND, conn=c)

    if conn is not None:
        _do(conn)
    else:
        with store.tx() as c:
            _do(c)
