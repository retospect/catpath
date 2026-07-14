"""The ``pathway`` kind — a precis-mcp plugin handler.

Slice 0 (dark, ``PRECIS_CATPATH_ENABLED``): a `pathway` ref owns a catpath
reaction-network run. ``put`` takes the config YAML as the body, runs the
catpath pipeline **in-process on EMT** (cheap, qualitative), and persists:

* the ``methods.md`` paragraph as the embedded/citable body chunk
  (``chunk_kind='pathway_body'``),
* the reaction graph + pooled-uncertainty ``results.json`` + the provenance
  config snapshot in ``meta``.

Regen is content-addressed: re-``put``ting an unchanged config is a no-op
cache hit. Fan-out across ``(model, seed)`` and heavy backends move to the
precis compute lane in later slices (see
``docs/design/catpath-integration.md`` in precis-mcp). Native `structure`
refs per intermediate (the ``pathway-node`` link) are slice 1 — deferred
here because link relations are a closed `Relation` Literal in precis core
and need a core edit to extend.
"""

from __future__ import annotations

import os
import re
from typing import Any, ClassVar

from precis.dispatch import Hub, InitError
from precis.errors import BadInput
from precis.protocol import Handler, KindSpec
from precis.response import Response
from precis.store.types import BlockInsert

from .persist import BODY_KIND, pathway_meta, pathway_title, persist_result

#: When set, `put` routes the compute to a `catpath_explore` job pinned to this
#: node instead of running catpath in-process (slice 1). The gateway sets it to
#: the GPU node's PRECIS_NODE (e.g. 'spark'); unset → in-process EMT (slice 0).
_ROUTE_NODE_ENV = "PRECIS_CATPATH_ROUTE_NODE"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PathwayHandler(Handler):
    spec: ClassVar[KindSpec] = KindSpec(
        kind="pathway",
        title="Reaction pathway (catpath)",
        description=(
            "A catalyst reaction-network exploration (catpath): give it a "
            "surface, substrate, and target as a YAML config body; it relaxes "
            "every intermediate and finds NEB barriers, reporting energies "
            "with pooled uncertainty (low-confidence flagged, not faked). "
            "put(kind='pathway', id='<name>', text='<config yaml>') runs it; "
            "get(kind='pathway', id='<name>', view='network'|'profile'|"
            "'methods'|'config'). Slice 0 runs EMT in-process (qualitative). "
            "See precis-pathway-help."
        ),
        supports_get=True,
        supports_put=True,
        supports_delete=True,
        supports_tag=True,
        is_numeric=False,
        role="artifact",
        corpus_role="none",
        # Own the derived catpath_explore compute job (ADR 0044 compute lane),
        # so `put` can route the run to the pinned node instead of running it
        # in-process. Requires precis KindSpec.can_own_jobs (>= 8.22).
        can_own_jobs=True,
        views=("network", "profile", "methods", "config"),
    )

    def __init__(self, *, hub: Hub) -> None:
        # Gated dark: the kind only appears when explicitly enabled, so
        # the slice merges without exposing an in-process compute path by
        # default. (Mirrors PRECIS_SANDBOX_ENABLED / PRECIS_CLASSIFY_ENABLED.)
        if os.environ.get("PRECIS_CATPATH_ENABLED", "") not in ("1", "true", "True"):
            raise InitError(
                "pathway kind is off; set PRECIS_CATPATH_ENABLED=1 to enable"
            )
        # catpath (ase/rdkit/networkx) is a hard dep of catpath[precis], but
        # guard so a broken env drops the kind cleanly instead of crashing boot.
        try:
            from . import runner  # noqa: F401
        except Exception as e:  # pragma: no cover - env-dependent
            raise InitError(f"catpath pipeline unavailable: {e}") from e
        _ = hub

    # -- put -------------------------------------------------------------
    def put(  # type: ignore[override]
        self,
        *,
        id: str | int | None = None,
        text: str | None = None,
        tags: list[str] | None = None,
        **_kw: Any,
    ) -> Response:
        from . import runner

        if not text or not text.strip():
            raise BadInput(
                "pathway needs the config YAML as the body (text=)",
                next=(
                    "put(kind='pathway', id='no_to_no3_pd', "
                    "text='substrate: \"NO\"\\ntarget: \"NO3\"\\n...')"
                ),
            )

        route_node = os.environ.get(_ROUTE_NODE_ENV) or None
        # Routed: run the config's own backend on the pinned node. In-process:
        # force EMT (the gateway has no ML backend, keeps the put cheap).
        force = None if route_node else "emt"

        # Parse once (chem-safe) to derive the slug + the effective content key,
        # so an unchanged config short-circuits before the expensive run.
        try:
            from ..config import _load_yaml

            raw = _load_yaml(text)
            effective = runner.effective_config(raw, force_backend=force)
            key = runner.content_key(effective)
            slug = self._slugify(id or effective.get("name") or "pathway")
        except BadInput:
            raise
        except Exception as e:
            raise BadInput(f"could not parse pathway config: {e}") from e

        store = self.hub.store
        existing = store.get_ref(kind="pathway", id=slug)
        if (
            existing is not None
            and existing.meta.get("content_key") == key
            and existing.meta.get("status") != "computing"
        ):
            return Response(
                body=(
                    f"pathway '{slug}' unchanged (cache hit {key[:12]}); "
                    "nothing to recompute."
                )
            )

        if route_node:
            return self._dispatch_job(
                store, slug, raw, effective, key, route_node, force, tags, existing
            )

        # In-process on EMT (slice 0). Synchronous — keep demo configs small.
        artifact = runner.run_pathway_from_yaml(text, force_backend="emt")
        with store.tx() as conn:
            if existing is None:
                ref = store.insert_ref(
                    kind="pathway",
                    slug=slug,
                    title=pathway_title(artifact),
                    meta=pathway_meta(artifact, extra={"backend_forced": "emt", "slice": 0}),
                    conn=conn,
                )
                store.insert_blocks(
                    ref.id,
                    [BlockInsert(pos=0, text=artifact["methods_md"],
                                 meta={"chunk_kind": BODY_KIND})],
                    conn=conn,
                )
                ref_id, verb = ref.id, "created"
            else:
                ref_id, verb = existing.id, "regenerated"
                persist_result(store, ref_id, artifact,
                               extra_meta={"backend_forced": "emt", "slice": 0}, conn=conn)
            for t in tags or []:
                self._add_tag(store, ref_id, t, conn)

        return Response(body=self._put_summary(slug, verb, artifact))

    def _dispatch_job(
        self,
        store: Any,
        slug: str,
        raw_config: dict[str, Any],
        effective: dict[str, Any],
        key: str,
        node: str,
        force: str | None,
        tags: list[str] | None,
        existing: Any,
    ) -> Response:
        """Route the compute: ensure the pathway ref exists (status=computing),
        then mint a `catpath_explore` job pinned to `node` (compute lane, ADR
        0044). The node's ssh_node worker claims it and runs catpath there,
        writing the result back onto this ref."""
        seed_meta = {
            "content_key": key,
            "config": effective,
            "status": "computing",
            "route_node": node,
            "slice": 1,
        }
        placeholder = (
            f"# {slug}\n\ncatpath compute dispatched to **{node}** "
            f"(cache_key {key[:12]}). Results will replace this on completion."
        )
        with store.tx() as conn:
            if existing is None:
                ref = store.insert_ref(
                    kind="pathway",
                    slug=slug,
                    title=f"pathway {slug} (computing)",
                    meta=seed_meta,
                    conn=conn,
                )
                store.insert_blocks(
                    ref.id,
                    [BlockInsert(pos=0, text=placeholder, meta={"chunk_kind": BODY_KIND})],
                    conn=conn,
                )
                ref_id = ref.id
            else:
                ref_id = existing.id
                store.stamp_ref_meta(ref_id, seed_meta, conn=conn)
            for t in tags or []:
                self._add_tag(store, ref_id, t, conn)

        from precis.handlers.job import JobHandler

        job = JobHandler(hub=self.hub).put(
            job_type="catpath_explore",
            executor="ssh_node",
            parent_id=ref_id,
            idem_key=f"catpath_explore:{key}",
            params={
                "pathway_ref_id": ref_id,
                "config": raw_config,
                "force_backend": force,  # None → run the config's own backend
                "content_key": key,
                "target_node": node,
            },
        )
        return Response(
            body=(
                f"dispatched catpath compute for '{slug}' to {node} "
                f"(cache_key {key[:12]}). {job.body}\n"
                f"Track: get(kind='pathway', id='{slug}') — status 'computing' "
                "until the job writes back."
            )
        )

    # -- get -------------------------------------------------------------
    def get(  # type: ignore[override]
        self,
        *,
        id: str | int | None = None,
        view: str | None = None,
        **_kw: Any,
    ) -> Response:
        if id is None:
            raise BadInput(
                "pathway get needs an id (the pathway slug)",
                next="get(kind='pathway', id='no_to_no3_pd')",
            )
        store = self.hub.store
        ref = store.get_ref(kind="pathway", id=str(id))
        if ref is None:
            raise BadInput(f"no pathway '{id}'")

        meta = ref.meta or {}
        if meta.get("status") == "computing":
            node = meta.get("route_node", "?")
            return Response(
                body=f"pathway '{id}' is computing on {node} "
                f"(cache_key {str(meta.get('content_key', ''))[:12]}). Check back shortly."
            )
        v = (view or "").lower()
        if v == "config":
            return Response(body=meta.get("config_snapshot_yaml", "(no config)"))
        if v == "methods":
            blocks = store.list_blocks_for_ref(ref.id)
            body = "\n\n".join(b.text for b in blocks if b.text)
            return Response(body=body or "(no methods)")
        if v == "network":
            return Response(body=self._render_network(ref.title, meta))
        # default / "profile"
        return Response(body=self._render_profile(ref.title, meta))

    # -- delete ----------------------------------------------------------
    def delete(  # type: ignore[override]
        self, *, id: str | int | None = None, **_kw: Any
    ) -> Response:
        if id is None:
            raise BadInput("pathway delete needs an id")
        store = self.hub.store
        ref = store.get_ref(kind="pathway", id=str(id))
        if ref is None:
            raise BadInput(f"no pathway '{id}'")
        store.soft_delete_ref(ref.id)
        return Response(body=f"deleted pathway '{id}'")

    # -- tag -------------------------------------------------------------
    def tag(  # type: ignore[override]
        self,
        *,
        id: str | int | None = None,
        add: list[str] | None = None,
        remove: list[str] | None = None,
        **_kw: Any,
    ) -> Response:
        if id is None:
            raise BadInput("pathway tag needs an id")
        store = self.hub.store
        ref = store.get_ref(kind="pathway", id=str(id))
        if ref is None:
            raise BadInput(f"no pathway '{id}'")
        with store.tx() as conn:
            for t in add or []:
                self._add_tag(store, ref.id, t, conn)
            for t in remove or []:
                self._remove_tag(store, ref.id, t, conn)
        return Response(body=f"tagged pathway '{id}'")

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _slugify(name: Any) -> str:
        s = _SLUG_RE.sub("-", str(name).strip().lower()).strip("-")
        return s or "pathway"

    @staticmethod
    def _put_summary(slug: str, verb: str, artifact: dict[str, Any]) -> str:
        r = artifact["results_json"]
        n_nodes = len(r["nodes"])
        n_edges = len(r["edges"])
        low = sum(1 for e in r["edges"] if e["barrier"].get("low_confidence"))
        warns = len(artifact["warnings"])
        return (
            f"{verb} pathway '{slug}': {n_nodes} states, {n_edges} steps "
            f"({r['n_samples']} samples, backend {r['backend']}). "
            f"{low} low-confidence barrier(s), {warns} warning(s). "
            f"get(kind='pathway', id='{slug}', view='profile')"
        )

    @staticmethod
    def _render_profile(title: str, meta: dict[str, Any]) -> str:
        r = meta.get("results", {})
        lines = [f"# {title}", ""]
        ref_name = r.get("pathway", [None])[0]
        lines.append(f"Energy reference: {r.get('energy_reference', '?')}")
        lines.append("")
        lines.append("## States (relative energy, eV)")
        nodes = r.get("nodes", {})
        for name in r.get("pathway", []):
            est = nodes.get(name, {})
            rel = est.get("mean", float("nan"))
            std = est.get("std", 0.0)
            flag = "  [LOW CONFIDENCE]" if est.get("low_confidence") else ""
            here = "  ←root" if name == ref_name else ""
            lines.append(f"  {name:<16} {rel:+.3f} ± {std:.3f}{flag}{here}")
        warns = meta.get("warnings") or []
        if warns:
            lines += ["", "## Warnings", *(f"  - {w}" for w in warns)]
        return "\n".join(lines)

    @staticmethod
    def _render_network(title: str, meta: dict[str, Any]) -> str:
        r = meta.get("results", {})
        lines = [f"# {title} — reaction network", ""]
        for e in r.get("edges", []):
            b = e.get("barrier", {})
            d = e.get("delta_e", {})
            flag = "  [LOW CONFIDENCE]" if b.get("low_confidence") else ""
            lines.append(
                f"  {e['reactant']} → {e['product']}: "
                f"Ea={b.get('mean', float('nan')):.3f}±{b.get('std', 0.0):.3f}  "
                f"ΔE={d.get('mean', float('nan')):+.3f}±{d.get('std', 0.0):.3f}{flag}"
            )
        return "\n".join(lines)

    def _add_tag(self, store: Any, ref_id: int, raw: str, conn: Any) -> None:
        from precis.store import Tag

        store.add_tag(ref_id, Tag.parse_strict(raw, kind="pathway"),
                      set_by="agent", conn=conn)

    def _remove_tag(self, store: Any, ref_id: int, raw: str, conn: Any) -> None:
        from precis.store import Tag

        store.remove_tag(ref_id, Tag.parse_strict(raw, kind="pathway"), conn=conn)
