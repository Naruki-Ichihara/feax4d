"""Persistence helpers for feax4d optimisation results.

:func:`optimize` already streams a ParaView history (``history.xdmf`` /
``.csv`` / ``.png``).  This module persists the things needed to *reload* a
run programmatically:

* ``design.npz`` — the final design vector ``x_opt`` plus the 8 unpacked node
  fields and shape metadata (so a design can be reloaded without rerunning).
* ``result.json`` — a human-readable summary (best objective, stop reason,
  iteration count, …) and the full :class:`OptimizeConfig` it came from.

Use :func:`save_result` to write them and :func:`load_design` /
:func:`load_summary` to read them back.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as onp

from feax4d.shell import N_FIELDS


_FIELD_NAMES = (
    "rho0", "x1_0", "x2_0", "x3_0",
    "rho1", "x1_1", "x2_1", "x3_1",
)


def _jsonable(obj):
    """Recursively convert a value to something JSON-serialisable.

    Dataclasses → dicts, Paths → str, callables → ``"<callable>"``, lists/tuples
    recurse; anything else that is not a JSON-native scalar/dict is summarised
    by its type name (e.g. a feax ``Mesh`` or numpy array).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if callable(obj):
        return "<callable>"
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    return f"<{type(obj).__name__}>"


def split_design(x_opt, n_nodes):
    """Split a flat design vector into the 8 named node fields."""
    x = onp.asarray(x_opt).reshape(-1)
    return {
        name: x[k * n_nodes:(k + 1) * n_nodes]
        for k, name in enumerate(_FIELD_NAMES)
    }


def save_result(result, config=None, out_dir=None, save_fields=True) -> Path:
    """Persist an :class:`OptimizeResult` to ``design.npz`` + ``result.json``.

    Parameters
    ----------
    result : OptimizeResult
        The object returned by :func:`feax4d.optimize`.
    config : OptimizeConfig, optional
        The config the run came from — serialised into ``result.json``.
    out_dir : path-like, optional
        Target directory.  Defaults to ``config.output_dir`` if given, else
        ``result.xdmf_path``'s parent.
    save_fields : bool
        Also store the 8 unpacked node fields in ``design.npz`` (convenient
        for post-processing without re-splitting).

    Returns
    -------
    Path
        The output directory.
    """
    if out_dir is None:
        out_dir = getattr(config, "output_dir", None) or Path(result.xdmf_path).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── design.npz ──
    x = onp.asarray(result.x_opt).reshape(-1)
    payload = {
        "x_opt": x,
        "n_nodes": onp.asarray(result.n_nodes),
        "n_fields": onp.asarray(N_FIELDS),
    }
    if save_fields:
        payload.update(split_design(x, result.n_nodes))
    onp.savez(out_dir / "design.npz", **payload)

    # ── result.json ──
    summary = {
        "best_obj": float(result.best_obj),
        "best_iter": int(result.best_iter),
        "n_iters": int(result.n_iters),
        "stop_reason": result.stop_reason,
        "denom": float(result.denom),
        "n_nodes": int(result.n_nodes),
        "xdmf_path": str(result.xdmf_path),
    }
    doc = {
        "summary": summary,
        "config": _jsonable(config) if config is not None else None,
        "final": {
            "obj": result.history["obj"][-1] if result.history["obj"] else None,
            "mean_density": result.history["vol"][-1] if result.history["vol"] else None,
        },
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(doc, f, indent=2)

    return out_dir


def load_design(path):
    """Load a saved ``design.npz``.

    ``path`` may be the ``design.npz`` file or the directory containing it.
    Returns ``(x_opt, meta)`` where ``meta`` holds ``n_nodes``, ``n_fields``
    and the 8 named node fields (if they were saved).
    """
    path = Path(path)
    if path.is_dir():
        path = path / "design.npz"
    data = onp.load(path)
    x_opt = data["x_opt"]
    meta = {k: data[k] for k in data.files if k != "x_opt"}
    if "n_nodes" in meta:
        meta["n_nodes"] = int(meta["n_nodes"])
    if "n_fields" in meta:
        meta["n_fields"] = int(meta["n_fields"])
    return x_opt, meta


def load_summary(path):
    """Load a saved ``result.json`` (dir or file path) as a dict."""
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    with open(path) as f:
        return json.load(f)
