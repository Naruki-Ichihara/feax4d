"""3D visualisation of the printed toolpaths.

:func:`plot_print_layers` renders the assembled print-layer structure (the
``layers`` returned by :func:`feax4d.fibre_paths_to_gcode` /
:func:`feax4d.svg_to_gcode_polymer_fill`) as 3D polylines: each layer is drawn
at its Z height in print order, with fibre toolpaths and polymer-infill
toolpaths in different colours.

Coordinates are the fibre/infill node coordinates (mm, with Y already in
physical-up orientation); Z is ``(layer_index + 1) * layer_height``.  Because a
laminate is far wider than it is thick, the Z axis is exaggerated by default so
the stack is legible (set ``z_exaggeration`` to ``1.0`` for true proportions).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as onp


def plot_print_layers(
    layers: Sequence[dict],
    layer_height: float = 0.15,
    fiber_color: str = "#ff7f0e",
    polymer_color: str = "#2ca02c",
    fiber_lw: float = 0.6,
    polymer_lw: float = 0.35,
    polymer_alpha: float = 0.6,
    z_exaggeration: Optional[float] = None,
    layer_gap: Optional[float] = None,
    elev: float = 22.0,
    azim: float = -60.0,
    ax=None,
    title: Optional[str] = None,
    legend: bool = False,
    show_ticks: bool = False,
):
    """Plot assembled print layers in 3D.

    All fibre paths are drawn in a single colour (``fiber_color``); the polymer
    infill is drawn faintly behind them.

    Parameters
    ----------
    layers : list of dict
        Print layers in order, each ``{"fiber": [Path...], "contour": [Path...],
        "type": "P"|"F"}`` (the ``layers`` entry of the g-code result).
    layer_height : float
        Default Z increment per print layer (mm); a per-layer ``layer_height``
        of ``0.0`` (coplanar fibre/polymer) is honoured.
    fiber_color : str
        Colour for all fibre toolpaths (default orange).
    z_exaggeration : float, optional
        Multiplier applied to Z for display.  ``None`` auto-scales so the stack
        height is ~30% of the in-plane span; ``1.0`` keeps true proportions.
    layer_gap : float, optional
        If given, the visual Z spacing (in plot units) between successive print
        layers is fixed to this value (coplanar fibre/polymer share a level),
        overriding ``z_exaggeration``.  Use it to spread the layers apart.
    show_ticks : bool
        Show axis tick marks and number labels (default False — a clean view).
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axes to draw into (a new figure is created otherwise).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    if ax is None:
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    n_layers = len(layers)
    xmin = ymin = onp.inf
    xmax = ymax = -onp.inf

    def _bounds(nodes):
        nonlocal xmin, xmax, ymin, ymax
        xmin = min(xmin, nodes[:, 0].min()); xmax = max(xmax, nodes[:, 0].max())
        ymin = min(ymin, nodes[:, 1].min()); ymax = max(ymax, nodes[:, 1].max())

    # First pass: in-plane bounds (for the auto Z exaggeration / box aspect).
    for ld in layers:
        for p in list(ld.get("fiber", [])) + list(ld.get("contour", [])):
            _bounds(onp.asarray(p.nodes))
    if not onp.isfinite(xmin):
        xmin = ymin = 0.0
        xmax = ymax = 1.0
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    total_z = max(sum(
        (layer_height if ld.get("layer_height") is None else ld.get("layer_height"))
        for ld in layers), 1e-9)
    if z_exaggeration is None:
        z_exaggeration = max(1.0, (0.3 * span) / total_z)

    n_fiber = n_poly = 0
    z_cum = 0.0          # cumulative honoured layer height (z_exaggeration mode)
    level = -1           # discrete level index (layer_gap mode)
    z_max = 0.0
    for ld in layers:
        # Honor an explicit per-layer height (0.0 ⇒ coplanar with the previous
        # layer), matching the g-code generator's Z handling.
        lh = ld.get("layer_height")
        lh = layer_height if lh is None else lh
        if layer_gap is not None:
            if lh > 0:
                level += 1
            z = max(level, 0) * layer_gap
        else:
            z_cum += lh
            z = z_cum * z_exaggeration
        z_max = max(z_max, z)
        # Polymer infill (faint, behind) first.
        for p in ld.get("contour", []):
            n = onp.asarray(p.nodes)
            ax.plot(n[:, 0], n[:, 1], z, color=polymer_color, lw=polymer_lw,
                    alpha=polymer_alpha)
            n_poly += 1
        # Fibre paths: single colour.
        for p in ld.get("fiber", []):
            n = onp.asarray(p.nodes)
            ax.plot(n[:, 0], n[:, 1], z, color=fiber_color, lw=fiber_lw)
            n_fiber += 1

    if title is None:
        title = (f"Print paths — {n_layers} layers "
                 f"({n_fiber} fibre, {n_poly} polymer toolpaths)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, max(z_max, 1e-9)))
    if show_ticks:
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z")
    else:
        # Clean view: no tick marks, no number labels, no grid.
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.grid(False)

    if legend:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color=fiber_color, lw=1.6, label="fibre"),
            Line2D([0], [0], color=polymer_color, lw=1.4,
                   alpha=min(1.0, polymer_alpha + 0.3), label="polymer"),
        ], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


def plot_print_paths(result: dict, **kwargs):
    """Plot the 3D print paths from a g-code result dict.

    ``result`` is the dict returned by :func:`feax4d.fibre_paths_to_gcode`
    (with ``polymer_fill=True``) or :func:`feax4d.svg_to_gcode_polymer_fill`;
    it must carry the ``layers`` / ``layer_height`` entries.  Extra keyword
    arguments are forwarded to :func:`plot_print_layers`.
    """
    if "layers" not in result:
        raise KeyError(
            "result has no 'layers' — use fibre_paths_to_gcode(..., "
            "polymer_fill=True) or svg_to_gcode_polymer_fill(...)."
        )
    return plot_print_layers(result["layers"],
                             layer_height=result.get("layer_height", 0.15),
                             **kwargs)


def plot_print_layers_plotly(
    layers: Sequence[dict],
    layer_height: float = 0.15,
    fiber_color: str = "#ff7f0e",
    polymer_color: str = "#2ca02c",
    fiber_width: float = 3.0,
    polymer_width: float = 1.2,
    polymer_opacity: float = 0.6,
    z_exaggeration: Optional[float] = None,
    layer_gap: Optional[float] = None,
    height: int = 650,
    title: Optional[str] = None,
    show_ticks: bool = False,
):
    """Interactive 3D print-path view as a **plotly** figure (mouse-rotatable).

    Same content as :func:`plot_print_layers` but rendered with plotly, which
    is interactive inline in Colab / Jupyter (drag to rotate, scroll to zoom).
    Returns a ``plotly.graph_objects.Figure``; call ``.show()`` to display.

    Requires ``plotly`` (pre-installed on Colab; ``pip install plotly`` locally).
    """
    import plotly.graph_objects as go

    # In-plane bounds + true stacked height (coplanar layers add 0).
    xmin = ymin = onp.inf
    xmax = ymax = -onp.inf
    for ld in layers:
        for p in list(ld.get("fiber", [])) + list(ld.get("contour", [])):
            n = onp.asarray(p.nodes)
            xmin = min(xmin, n[:, 0].min()); xmax = max(xmax, n[:, 0].max())
            ymin = min(ymin, n[:, 1].min()); ymax = max(ymax, n[:, 1].max())
    if not onp.isfinite(xmin):
        xmin = ymin = 0.0; xmax = ymax = 1.0
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    total_z = max(sum(
        (layer_height if ld.get("layer_height") is None else ld.get("layer_height"))
        for ld in layers), 1e-9)
    if z_exaggeration is None:
        z_exaggeration = max(1.0, (0.3 * span) / total_z)

    traces = []
    z_cum = 0.0
    level = -1
    z_max = 0.0
    for ld in layers:
        lh = ld.get("layer_height")
        lh = layer_height if lh is None else lh
        if layer_gap is not None:
            if lh > 0:
                level += 1
            z = max(level, 0) * layer_gap
        else:
            z_cum += lh
            z = z_cum * z_exaggeration
        z_max = max(z_max, z)
        for p in ld.get("contour", []):
            n = onp.asarray(p.nodes)
            traces.append(go.Scatter3d(
                x=n[:, 0], y=n[:, 1], z=onp.full(n.shape[0], z), mode="lines",
                line=dict(color=polymer_color, width=polymer_width),
                opacity=polymer_opacity, showlegend=False, hoverinfo="skip"))
        for p in ld.get("fiber", []):
            n = onp.asarray(p.nodes)
            traces.append(go.Scatter3d(
                x=n[:, 0], y=n[:, 1], z=onp.full(n.shape[0], z), mode="lines",
                line=dict(color=fiber_color, width=fiber_width),
                showlegend=False, hoverinfo="skip"))

    z_plot = max(z_max, 1e-9)
    m = max(xmax - xmin, ymax - ymin, z_plot, 1e-9)
    # Clean axes by default: hide tick numbers, ticks and grid.
    ax_kw = {} if show_ticks else dict(
        showticklabels=False, ticks="", showgrid=False, zeroline=False, title="")
    fig = go.Figure(traces)
    fig.update_layout(
        height=height, title=title or "Print paths (drag to rotate)",
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(**({"title": "x [mm]"} if show_ticks else {}), **ax_kw),
            yaxis=dict(**({"title": "y [mm]"} if show_ticks else {}), **ax_kw),
            zaxis=dict(**({"title": "z"} if show_ticks else {}), **ax_kw),
            aspectmode="manual",
            aspectratio=dict(x=(xmax - xmin) / m, y=(ymax - ymin) / m, z=z_plot / m),
        ),
    )
    return fig


def plot_print_paths_plotly(result: dict, **kwargs):
    """Interactive plotly 3D print-path view from a g-code result dict.

    See :func:`plot_print_layers_plotly`.  ``result`` must carry the
    ``layers`` / ``layer_height`` entries (``fibre_paths_to_gcode`` /
    ``svg_to_gcode_polymer_fill`` output).
    """
    if "layers" not in result:
        raise KeyError(
            "result has no 'layers' — use fibre_paths_to_gcode(..., "
            "polymer_fill=True) or svg_to_gcode_polymer_fill(...)."
        )
    return plot_print_layers_plotly(result["layers"],
                                    layer_height=result.get("layer_height", 0.15),
                                    **kwargs)
