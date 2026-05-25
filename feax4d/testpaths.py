"""Synthetic test fibre-path SVGs (built with svgpathtools) for verifying the
g-code / visualisation pipeline without running an optimisation + SH stage.

The generated SVG follows the feax4d convention consumed by
:func:`feax4d.fibre_paths_to_gcode`: a ``<g id="fibre_paths">`` group of
``<path>`` elements (discretised by the svgpathtools-based parser) plus an
optional ``<g id="contour">`` rectangle marking the polymer region.
Coordinates are in millimetres (the SVG / machine units of the fibrifier
stage).
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import Optional


def make_test_fibre_svg(
    out_path,
    width: float = 100.0,
    height: float = 50.0,
    pitch: float = 4.0,
    kind: str = "lines",
    amplitude: float = 6.0,
    with_contour: bool = True,
    stroke_mm: float = 0.3,
):
    """Write a simple analytic fibre-path SVG using svgpathtools.

    Fibre paths run **along the long axis** of a ``width × height`` rectangle,
    evenly spaced by ``pitch`` across the short axis.

    Parameters
    ----------
    out_path : path-like
        Where to write the ``.svg``.
    width, height : float
        Rectangle size [mm].  The fibres run along whichever is longer.
    pitch : float
        Spacing between adjacent fibre paths across the short axis [mm].
    kind : {"lines", "wave", "snake"}
        ``"lines"`` — straight fibres (one ``Line`` each, all same direction).
        ``"wave"`` — gently curved fibres (a ``CubicBezier`` each, amplitude
        ``amplitude``), to also exercise curve discretisation.
        ``"snake"`` — one continuous serpentine/boustrophedon ``<path>`` that
        snakes back and forth along the long axis (a single fibre).
    with_contour : bool
        Also emit a rectangular ``id="contour"`` polymer region (the full
        footprint).
    stroke_mm : float
        SVG stroke width.

    Returns
    -------
    pathlib.Path
        The written SVG path.
    """
    from svgpathtools import Path as SvgPath, Line, CubicBezier

    W, H = float(width), float(height)
    long_is_x = W >= H
    L, S = max(W, H), min(W, H)              # long, short extents
    n = max(1, int(round(S / pitch)))

    def _offset(i):
        return (i + 0.5) * S / n            # position across the short axis

    def _ends(s, forward):
        """(start, end) of the long-axis traverse at offset ``s``."""
        if long_is_x:
            lo, hi = complex(0.0, s), complex(L, s)
        else:
            lo, hi = complex(s, 0.0), complex(s, L)
        return (lo, hi) if forward else (hi, lo)

    if kind == "snake":
        # One continuous serpentine (boustrophedon) path: each long-axis pass
        # reverses direction and is joined to the next by a short step across
        # the short axis — a single <path>.
        segs = []
        for i in range(n):
            a, b = _ends(_offset(i), forward=(i % 2 == 0))
            segs.append(Line(a, b))
            if i < n - 1:                    # connector to the next pass
                nxt, _ = _ends(_offset(i + 1), forward=((i + 1) % 2 == 0))
                segs.append(Line(b, nxt))
        fibre_d = [SvgPath(*segs).d()]
    else:
        def _fibre(s):
            a, b = _ends(s, forward=True)
            if kind == "wave":
                if long_is_x:
                    return SvgPath(CubicBezier(a, complex(L / 3, s + amplitude),
                                               complex(2 * L / 3, s - amplitude), b))
                return SvgPath(CubicBezier(a, complex(s + amplitude, L / 3),
                                           complex(s - amplitude, 2 * L / 3), b))
            return SvgPath(Line(a, b))

        fibre_d = [_fibre(_offset(i)).d() for i in range(n)]

    contour_d = []
    if with_contour:
        rect = SvgPath(
            Line(complex(0, 0), complex(W, 0)),
            Line(complex(W, 0), complex(W, H)),
            Line(complex(W, H), complex(0, H)),
            Line(complex(0, H), complex(0, 0)),
        )
        contour_d.append(rect.d())

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" '
        f'viewBox="0 0 {W} {H}">',
        f'  <desc>test fibre paths ({kind}, {n} along long axis) — feax4d.make_test_fibre_svg</desc>',
        f'  <g id="fibre_paths" fill="none" stroke="#000000" stroke-width="{stroke_mm}">',
    ]
    lines += [f'    <path d="{d}"/>' for d in fibre_d]
    lines.append('  </g>')
    if contour_d:
        lines.append(f'  <g id="contour" fill="none" stroke="#888888" stroke-width="{stroke_mm}">')
        lines += [f'    <path d="{d}"/>' for d in contour_d]
        lines.append('  </g>')
    lines.append('</svg>')

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
