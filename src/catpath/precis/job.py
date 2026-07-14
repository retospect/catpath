"""``catpath_explore`` job_type — run a catpath reaction-network exploration on
the pinned compute node and write the result back onto its `pathway` ref.

This is the routing seam (slice 1). The `pathway` handler mints one of these
jobs (`meta.executor='ssh_node'`, `meta.params.target_node=<node>`, parented on
the pathway ref via the compute lane, ADR 0044). The `ssh_node` worker pass on
that node — and only that node, per the target_node claim gate — claims it and
invokes `dispatch()` here, which runs catpath **in-process** (catpath[precis] +
the ML backend are installed in that node's worker venv) and persists the
artifact. So the heavy relax/NEB runs where the hardware is; the gateway only
mints the job.

Registered via the ``precis.job_types`` entry point
(``catpath_explore = catpath.precis.job:load``). Needs no host capability
(``REQUIRES`` is empty); the node pin does the routing.
"""

from __future__ import annotations

import logging
from typing import Any

from precis.workers.job_types import JobTypeSpec

log = logging.getLogger(__name__)

NAME = "catpath_explore"

_PARAMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # The pathway ref the result is written back onto.
        "pathway_ref_id": {"type": "integer"},
        # The authoritative config (parsed YAML) to run.
        "config": {"type": "object"},
        # Backend override; null → run the config's own mlip.backend.
        "force_backend": {"type": ["string", "null"]},
        # Content address (matches the handler's regen cache key).
        "content_key": {"type": "string"},
        # The node this job pins itself to (claim gate → runs here).
        "target_node": {"type": ["string", "null"]},
    },
    "required": ["pathway_ref_id", "config"],
    "additionalProperties": True,
}

COMPATIBLE_EXECUTORS = frozenset({"ssh_node"})
#: catpath needs no special host capability (empty ⊆ any executor's PROVIDES);
#: the target_node pin routes it to the box with catpath + the backend installed.
REQUIRES: frozenset[str] = frozenset()
DESCRIPTION = (
    "Run a catpath reaction-network exploration on the pinned node; "
    "write the graph + pooled-uncertainty result back onto the pathway ref."
)


def _dispatch(ctx: Any, spec: Any) -> None:
    """Plugin dispatcher invoked by ``ssh_node`` for a claimed job. Runs the
    catpath pipeline in-process on this node and persists the artifact onto the
    pathway ref. ``ctx`` is a precis DispatchContext (store / meta / append_chunk
    / set_meta / set_status / record_failure)."""
    params = (ctx.meta or {}).get("params") or {}
    try:
        pathway_ref_id = int(params["pathway_ref_id"])
        config = dict(params["config"])
    except (KeyError, TypeError, ValueError) as exc:
        ctx.record_failure(f"catpath_explore: malformed params ({exc})")
        return
    force_backend = params.get("force_backend")
    backend = force_backend or (config.get("mlip") or {}).get("backend", "?")
    ctx.append_chunk(
        "job_event",
        f"catpath_explore: {config.get('name', '?')} backend={backend}",
    )

    try:
        from catpath.precis import runner

        artifact = runner.run_pathway(config, force_backend=force_backend)
    except Exception as exc:  # pragma: no cover - env/compute dependent
        log.warning("catpath_explore: run failed", exc_info=True)
        ctx.record_failure(f"catpath_explore: run failed: {exc}")
        return

    try:
        from catpath.precis.persist import persist_result

        persist_result(
            ctx.store,
            pathway_ref_id,
            artifact,
            extra_meta={"produced_by": NAME, "slice": 1, "ran_on": params.get("target_node")},
        )
    except Exception as exc:
        log.warning("catpath_explore: persist failed", exc_info=True)
        ctx.record_failure(f"catpath_explore: persist failed: {exc}")
        return

    r = artifact["results_json"]
    ctx.set_meta(content_key=artifact["content_key"], n_states=len(r["nodes"]))
    ctx.append_chunk(
        "job_summary",
        f"catpath: {len(r['nodes'])} states, {len(r['edges'])} steps "
        f"({r['n_samples']} samples, backend {r['backend']}) → pathway #{pathway_ref_id}.",
    )
    ctx.set_status("succeeded")


def _run(*_a: Any, **_k: Any) -> Any:
    raise NotImplementedError("catpath_explore runs via dispatch(), not run()")


SPEC = JobTypeSpec(
    name=NAME,
    params_schema=_PARAMS_SCHEMA,
    compatible_executors=COMPATIBLE_EXECUTORS,
    requires=REQUIRES,
    description=DESCRIPTION,
    run=_run,
    dispatch=_dispatch,
)


def load() -> JobTypeSpec:
    return SPEC


__all__ = ["SPEC", "load", "NAME"]
