"""Vendored Fibrifier (9T Labs) G-code backend.

This subpackage is a minimal, self-contained vendoring of the fibrifier
G-code generator from **libertas**
(https://github.com/Naruki-Ichihara/libertas), GPL-3.0-or-later — the same
license as feax4d.  Only the four files needed to turn fibre-path SVGs into
Fibrifier-format G-code are included (``path``, ``layer``, ``svg_parser``,
``fibrifier_gcode``); they depend on numpy + matplotlib only, so feax4d does
not pull in libertas's full dependency stack (fullcontrol, plotly, pydantic,
triangle, svgpathtools).

The internal ``libertas.*`` imports have been rewritten to
``feax4d.fibrifier.*``; the generator logic is otherwise unchanged.
"""
from feax4d.fibrifier.path import Path
from feax4d.fibrifier.layer import Layer
from feax4d.fibrifier.svg_parser import parse_svg_to_paths, save_paths_to_svg
from feax4d.fibrifier.fibrifier_gcode import (
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

__all__ = [
    "Path",
    "Layer",
    "parse_svg_to_paths",
    "save_paths_to_svg",
    "svg_to_fibrifier_gcode",
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
