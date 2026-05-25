"""Fibre-path → Fibrifier (9T Labs) G-code stage.

Thin wrapper over the vendored :mod:`feax4d.fibrifier` backend that turns the
fibre-path SVGs written by :func:`feax4d.generate_fibre_paths`
(``fibre_paths_layer{i}.svg``) into a single multi-layer Fibrifier-format
G-code file.

* :func:`svg_to_gcode` — direct pass-through to the backend
  (``svg_to_fibrifier_gcode``); takes one or more SVG paths.
* :func:`fibre_paths_to_gcode` — convenience that collects the per-layer SVGs
  from a fibre-paths directory (in layer order) and feeds them to the backend.

Print parameters live in :class:`FibrifierParams` (and its grouped
sub-dataclasses), re-exported here.
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import List, Optional, Sequence, Union

from feax4d.fibrifier import (
    svg_to_fibrifier_gcode,
    FibrifierGcodeGenerator,
    FibrifierParams,
    FibrifierTemperatureParams,
    FibrifierSpeedParams,
    FibrifierExtrusionParams,
    FibrifierFiberParams,
    FibrifierRetractionParams,
    FibrifierLayer,
    FibrifierModel,
)
from feax4d.fibrifier.path import Path as _FibrePath
from feax4d.fibrifier.fibrifier_gcode import (
    load_and_prepare_paths as _load_and_prepare_paths,
    generate_infill_paths as _generate_infill_paths,
    plot_paths as _plot_paths,
)

# Direct alias — the backend entry point under a shorter feax4d name.
svg_to_gcode = svg_to_fibrifier_gcode


def _svg_has_paths(svg_path) -> bool:
    """True if the SVG contains at least one <polyline>/<path> element."""
    try:
        text = _Path(svg_path).read_text()
    except OSError:
        return False
    return ("<polyline" in text) or ("<path" in text)


def collect_layer_svgs(
    fibre_dir,
    layers: Optional[Sequence[int]] = None,
    skip_empty: bool = True,
) -> List[str]:
    """Collect ``fibre_paths_layer{i}.svg`` files from a directory in order.

    Parameters
    ----------
    fibre_dir : path-like
        Directory written by :func:`feax4d.generate_fibre_paths`.
    layers : sequence of int, optional
        Restrict to these layer indices (default: all found, sorted).
    skip_empty : bool
        Drop SVGs that contain no path elements (e.g. a polymer-only layer
        with no fibres), which would otherwise emit an empty G-code layer.
    """
    fibre_dir = _Path(fibre_dir)
    found = {}
    for p in fibre_dir.glob("fibre_paths_layer*.svg"):
        try:
            idx = int(p.stem.replace("fibre_paths_layer", ""))
        except ValueError:
            continue
        found[idx] = p
    keys = sorted(found) if layers is None else [k for k in layers if k in found]
    svgs = []
    for k in keys:
        if skip_empty and not _svg_has_paths(found[k]):
            continue
        svgs.append(str(found[k]))
    return svgs


def _footprint_contour(svg_path) -> _FibrePath:
    """Closed rectangle Path covering the SVG viewBox (y flipped like fibres).

    This bounds the polymer infill region to the full plate footprint — so
    polymer fills the whole layer area (fibres are laid on the separate fibre
    layer).  No density contour is required or printed.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox", "0 0 0 0").split()
    x0, y0, W, H = (float(v) for v in vb)
    rect = [(x0, y0), (x0 + W, y0), (x0 + W, y0 + H), (x0, y0 + H), (x0, y0)]
    rect = [(x, H - y) for (x, y) in rect]   # same flip as fibrifier _flip_paths_y
    return _FibrePath(path_id=0, nodes=rect, path_type="contour")


def _polymer_infill_for_svg(svg_path, angle_deg, pitch, inset):
    """Polymer infill Path list filling the footprint of one fibre-path SVG."""
    contour = _footprint_contour(svg_path)
    return _generate_infill_paths([contour], angle_deg, pitch, inset)


def svg_to_gcode_polymer_fill(
    svg_paths: Sequence[Union[str, _Path]],
    output_gcode: str = "output_fibrifier.gcode",
    params: Optional[FibrifierParams] = None,
    infill_angle: Union[float, Sequence[float]] = 45.0,
    infill_pitch: float = 0.8,
    infill_inset: float = 0.3,
    layer_print_layers: Union[int, Sequence[int]] = 1,
    polymer_base_layers: int = 0,
    polymer_top_layers: int = 0,
    polymer_in_fiber_layers: bool = True,
    cross_hatch: bool = True,
    coplanar: bool = True,
    fiber_cut: Optional[bool] = None,
    connection_threshold: float = 5.0,
    min_fiber_length: float = 23.73,
    smooth_sigma: float = 3.0,
    decimate_epsilon: float = 0.05,
    flip_y: bool = True,
    preview_path: Optional[str] = None,
    offset_x: Optional[float] = None,
    offset_y: Optional[float] = None,
    layer_height: Optional[float] = None,
):
    """Build Fibrifier g-code where each layer = polymer infill (P) + fibre (F).

    Unlike :func:`svg_to_gcode` (which only emits a polymer layer when the SVG
    carries a density *contour*), this fills the polymer region of every layer
    with infill scan-lines and lays the fibre paths on the adjacent fibre
    layer — i.e. polymer everywhere the fibre isn't.  Only the infill is
    emitted; no perimeter/contour loop is printed.

    Build thickness with ``layer_print_layers`` — the number of physical print
    laminae for **each design layer** (int for all layers, or one value per
    layer, e.g. ``[n0, n1]``).  Each design layer's pattern (its polymer infill
    + its fibres) is stacked ``n_i`` times in Z, so design layer ``i`` becomes a
    coherent thick lamina of thickness ``n_i * layer_height``.  The part stays a
    true bilayer (bottom = layer 0, top = layer 1) — mechanically consistent
    with re-optimising at ``T_LAYER_i = n_i * layer_height`` (so increasing
    thickness is just a thickness change in the optimisation, not a different
    laminate).  With ``cross_hatch`` the polymer infill rotates +90° between
    successive laminae of the same design layer.

    Other parameters mirror :func:`svg_to_gcode`, plus ``infill_angle``
    (scalar or one angle per SVG), ``infill_pitch`` and ``infill_inset``.
    """
    if params is None:
        params = FibrifierParams()
    if offset_x is not None:
        params.offset_x = offset_x
    if offset_y is not None:
        params.offset_y = offset_y
    if layer_height is not None:
        params.layer_height = layer_height
    if fiber_cut is not None:
        params.fiber.fiber_cut = fiber_cut

    import matplotlib.pyplot as plt

    svg_paths = [str(s) for s in svg_paths]
    angles = (list(infill_angle) if isinstance(infill_angle, (list, tuple))
              else [float(infill_angle)] * len(svg_paths))
    counts = (list(layer_print_layers) if isinstance(layer_print_layers, (list, tuple))
              else [int(layer_print_layers)] * len(svg_paths))

    # Preview naming: one PNG per design layer.  Without an explicit
    # ``preview_path`` they are written next to the g-code as
    # ``<svg stem>_preview.png``; with one, as ``<stem>_layer{i}.png``.
    pv_base = _Path(preview_path) if preview_path else _Path(output_gcode)

    def _layer_preview_path(i, svg):
        if preview_path:
            return str(pv_base.with_name(f"{pv_base.stem}_layer{i}.png"))
        return str(pv_base.with_name(f"{_Path(svg).stem}_preview.png"))

    def _infill_at(contour, svg, angle):
        """Polymer infill at one angle: density contour if present, else footprint."""
        if contour:
            return _generate_infill_paths(contour, angle, infill_pitch, infill_inset), "density-limited"
        return _polymer_infill_for_svg(svg, angle, infill_pitch, infill_inset), "footprint"

    # Continuous fibre (no cut) → order the stacked laminae end-to-start by
    # reversing every other lamina, so the climb between layers happens at the
    # same X/Y (a clean Z step) rather than a diagonal jump.
    continuous = not params.fiber.fiber_cut

    def _reverse_fibre(paths):
        return [_FibrePath(path_id=p.path_id, nodes=list(reversed(p.nodes)),
                           path_type=p.path_type) for p in reversed(paths)]

    # Pre-load every SVG once (fibre + polymer contour).
    loaded = []
    for svg in svg_paths:
        fiber, contour, _svg_h = _load_and_prepare_paths(
            svg_path=svg, connection_threshold=connection_threshold,
            min_fiber_length=min_fiber_length, flip_y=flip_y,
            smooth_sigma=smooth_sigma, decimate_epsilon=decimate_epsilon,
        )
        loaded.append((svg, fiber, contour))

    layers = []
    all_fiber, all_contour = [], []
    preview_paths = []

    def _add_polymer_only(svg, contour, base_angle, count):
        """Append ``count`` polymer-only print layers (no fibre) over a region."""
        for k in range(count):
            ang = base_angle + (90.0 * k if cross_hatch else 0.0)
            infill, _ = _infill_at(contour, svg, ang)
            layers.append({"fiber": [], "contour": infill, "type": "P"})
            all_contour.extend(infill)

    # ── Bottom polymer-only cap (uses the first design layer's region) ──
    if polymer_base_layers > 0:
        _add_polymer_only(loaded[0][0], loaded[0][2], angles[0], polymer_base_layers)
        print(f"  Base: {polymer_base_layers} polymer-only layer(s)")

    # ── Fibre stack ──
    for i, (svg, fiber, contour) in enumerate(loaded):
        print(f"\n=== SVG {i + 1}: {svg} ===")
        base_angle = angles[i % len(angles)]
        n_i = max(1, counts[i % len(counts)])

        # Stack n_i identical laminae for this design layer.  Each lamina is
        # one polymer print-layer (+ one fibre print-layer where the design
        # layer has fibres); cross_hatch rotates the infill +90° per lamina.
        first_infill = None
        region = "footprint"
        for r in range(n_i):
            # Polymer in this lamina: always when the layer has no fibre, and
            # when ``polymer_in_fiber_layers`` for fibre layers (else the fibre
            # layers are fibre-only — polymer lives only in the cap layers).
            add_polymer = polymer_in_fiber_layers or not fiber
            if add_polymer:
                ang = base_angle + (90.0 * r if cross_hatch else 0.0)
                infill, region = _infill_at(contour, svg, ang)
                layers.append({"fiber": [], "contour": infill, "type": "P"})
                all_contour.extend(infill)
                if first_infill is None:
                    first_infill = infill
            if fiber:
                # When paired with a coplanar polymer layer, the fibre shares
                # its Z (layer_height 0); a fibre-only layer advances Z itself.
                fib = _reverse_fibre(fiber) if (continuous and r % 2 == 1) else fiber
                f_layer = {"fiber": fib, "contour": [], "type": "F"}
                if add_polymer and coplanar:
                    f_layer["layer_height"] = 0.0
                layers.append(f_layer)
                all_fiber.extend(fib)
        print(f"  Layer {i}: {n_i} printed lamina(e) × "
              f"[polymer{' + fibre' if fiber else ''}] "
              f"({region}, base {base_angle}°, {len(fiber)} fibre paths/lamina)")

        # Per-layer preview: this layer's fibres + (first) polymer infill.
        pv = _layer_preview_path(i, svg)
        fig, _ = _plot_paths(
            fiber, first_infill or [],
            title=f"Layer {i}: {len(fiber)} fibre + polymer "
                  f"(x{n_i} print laminae)")
        fig.savefig(pv, dpi=200, bbox_inches="tight")
        plt.close(fig)
        preview_paths.append(pv)
        print(f"  Preview saved to {pv}")

    # ── Top polymer-only cap (uses the last design layer's region) ──
    if polymer_top_layers > 0:
        _add_polymer_only(loaded[-1][0], loaded[-1][2],
                          angles[(len(loaded) - 1) % len(angles)], polymer_top_layers)
        print(f"  Top: {polymer_top_layers} polymer-only layer(s)")

    gen = FibrifierGcodeGenerator(params=params)
    gen.generate_multilayer(layers, filename=output_gcode)

    return {
        "gcode_path": output_gcode,
        "preview_paths": preview_paths,
        "preview_path": preview_paths[0] if preview_paths else None,
        "layer_print_layers": counts,
        "n_layers": len(layers),
        "n_fiber_paths": len(all_fiber),
        "n_polymer_paths": len(all_contour),
        "total_fiber_mm": sum(p.length for p in all_fiber),
        "total_stretches": gen._stretch_counter,
        # The assembled per-print-layer structure (in print order) and the
        # layer height — consumed by feax4d.viz for 3D print-path plots.
        "layers": layers,
        "layer_height": params.layer_height,
    }


def fibre_paths_to_gcode(
    source: Union[str, _Path, Sequence[Union[str, _Path]]],
    output_gcode: Optional[str] = None,
    params: Optional[FibrifierParams] = None,
    layers: Optional[Sequence[int]] = None,
    polymer_fill: bool = True,
    infill_angle: Union[float, Sequence[float]] = 45.0,
    infill_pitch: float = 0.8,
    infill_inset: float = 0.3,
    layer_print_layers: Union[int, Sequence[int]] = 1,
    polymer_base_layers: int = 0,
    polymer_top_layers: int = 0,
    polymer_in_fiber_layers: bool = True,
    cross_hatch: bool = True,
    coplanar: bool = True,
    fiber_cut: Optional[bool] = None,
    skip_empty: Optional[bool] = None,
    **kwargs,
):
    """Generate Fibrifier G-code from feax4d fibre-path output.

    Parameters
    ----------
    source : path-like or list of path-like
        Either a directory containing ``fibre_paths_layer{i}.svg`` files, or
        an explicit list of SVG file paths (used in the given order).
    output_gcode : str, optional
        Output ``.gcode`` path.  Defaults to ``<dir>/fibrifier.gcode`` when
        ``source`` is a directory, else ``output_fibrifier.gcode``.
    params : FibrifierParams, optional
        Print parameters (temperatures, speeds, offsets, …).  Defaults to the
        9T Labs reference configuration.
    layers : sequence of int, optional
        Restrict to these layer indices when ``source`` is a directory.
    polymer_fill : bool
        If True (default), fill every layer's footprint with polymer infill
        (P) and lay the fibres on the adjacent fibre layer (F) — polymer
        everywhere the fibre isn't.  If False, only fibre layers are emitted.
    infill_angle, infill_pitch, infill_inset
        Polymer infill scan-line angle (scalar or one per layer), spacing and
        boundary inset [mm].  Used only when ``polymer_fill`` is True.
    layer_print_layers : int or sequence of int
        Number of physical print laminae per design layer (int for all, or one
        per layer e.g. ``[n0, n1]``) — this is how you build part thickness.
        Design layer ``i`` is stacked ``n_i`` times, keeping the part a true
        bilayer; mechanically consistent with re-optimising at
        ``T_LAYER_i = n_i * layer_height``.
    polymer_base_layers, polymer_top_layers : int
        Number of pure-polymer (no fibre) cap layers below / above the fibre
        stack — e.g. ``2`` + ``layer_print_layers=10`` + ``2`` gives a
        2-polymer / 10-fibre / 2-polymer sandwich.
    polymer_in_fiber_layers : bool
        If True (default), each fibre layer also carries coplanar polymer
        infill in its non-fibre region.  Set False for **fibre-only** fibre
        layers (polymer only in the cap layers) — fewer polymer layers.
    cross_hatch : bool
        Rotate the polymer infill +90° between successive laminae of a layer.
    fiber_cut : bool, optional
        Insert a fibre cut at the end of each fibre stretch (default, ``True``).
        ``False`` prints the fibre continuously — no severing, the whole path is
        extruded (no nozzle-dead-length pull-through or ironing-out move).
        Overrides ``params.fiber.fiber_cut`` when given.
    coplanar : bool
        If True (default), the fibre of each lamina shares the **same Z** as
        its polymer matrix (they tile complementary in-plane regions, so the
        lamina is one ``layer_height`` thick).  If False, the fibre is printed
        one layer above its polymer (Fibrifier "polymer bed + fibre on top").
    skip_empty : bool, optional
        Skip path-less SVGs.  Defaults to False when ``polymer_fill`` (an
        all-polymer layer still needs filling) and True otherwise.
    **kwargs
        Forwarded to the backend (``connection_threshold``,
        ``min_fiber_length``, ``smooth_sigma``, ``decimate_epsilon``,
        ``flip_y``, ``offset_x``, ``offset_y``, ``layer_height``, …).

    Returns
    -------
    dict
        Backend summary (gcode_path, preview_path, n_layers, n_fiber_paths,
        n_polymer_paths/n_contour_paths, total_fiber_mm, total_stretches).
    """
    if skip_empty is None:
        skip_empty = not polymer_fill

    if isinstance(source, (str, _Path)) and _Path(source).is_dir():
        svgs = collect_layer_svgs(source, layers=layers, skip_empty=skip_empty)
        if not svgs:
            raise FileNotFoundError(
                f"No fibre_paths_layer*.svg found in {source}"
            )
        if output_gcode is None:
            output_gcode = str(_Path(source) / "fibrifier.gcode")
    else:
        svgs = [str(s) for s in ([source] if isinstance(source, (str, _Path)) else source)]
        if skip_empty:
            svgs = [s for s in svgs if _svg_has_paths(s)]
        if output_gcode is None:
            output_gcode = "output_fibrifier.gcode"

    # The backend writes the g-code and a preview PNG next to it; make sure
    # the target directory exists.
    out_parent = _Path(output_gcode).parent
    if str(out_parent):
        out_parent.mkdir(parents=True, exist_ok=True)

    if polymer_fill:
        return svg_to_gcode_polymer_fill(
            svgs, output_gcode=output_gcode, params=params,
            infill_angle=infill_angle, infill_pitch=infill_pitch,
            infill_inset=infill_inset, layer_print_layers=layer_print_layers,
            polymer_base_layers=polymer_base_layers, polymer_top_layers=polymer_top_layers,
            polymer_in_fiber_layers=polymer_in_fiber_layers,
            cross_hatch=cross_hatch, coplanar=coplanar, fiber_cut=fiber_cut, **kwargs,
        )
    return svg_to_gcode(svgs, output_gcode=output_gcode, params=params, **kwargs)


__all__ = [
    "svg_to_gcode",
    "svg_to_gcode_polymer_fill",
    "fibre_paths_to_gcode",
    "collect_layer_svgs",
    "FibrifierGcodeGenerator",
    "FibrifierParams",
    "FibrifierTemperatureParams",
    "FibrifierSpeedParams",
    "FibrifierExtrusionParams",
    "FibrifierFiberParams",
    "FibrifierRetractionParams",
    "FibrifierLayer",
    "FibrifierModel",
]
