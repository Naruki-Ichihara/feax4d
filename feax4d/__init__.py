"""feax4d: a 4D-printing design toolkit built on feax.

Two stages, exposed both as composable building blocks and as high-level
config-driven runners:

1. **Bilayer thermal shell topology optimisation** — design per-layer density
   and fibre orientation of a cooled CFRP/polymer laminate so its thermal
   warping holds a target shape under load (e.g. a cantilever that stays flat).
   See :class:`OptimizeConfig` / :func:`optimize`.

2. **Aligned-SH fibre-path generation** — turn the optimised per-layer fibre
   director into manufacturable fibre-path polylines (SVG / NPZ / VTU / PNG).
   See :func:`generate_fibre_paths`.
"""
from feax4d.materials import Lamina, Polymer, make_layer_constitutive
from feax4d.shell import (
    BilayerThermalShell,
    N_FIELDS,
    make_bilayer_shell,
    uniform_transverse_load,
    edge_predicate,
    cantilever_edges,
    clamp_bc,
)
from feax4d.objectives import (
    make_shape_match_fn,
    ud_penalty_total,
    rho_contrast_penalty_total,
    mag_consistency_total,
)
from feax4d.optimize import (
    OptimizeConfig,
    OptimizeResult,
    optimize,
    pack,
    unpack,
    initial_design,
    bounds,
    make_process_fn,
)
from feax4d.fibre import (
    AlignedSHStep,
    setup_aligned_sh,
    run_aligned_sh,
    run_aligned_sh_steps,
    extract_fibre_paths,
    extract_region_contours,
    write_paths_svg,
    director_from_a2,
    generate_fibre_paths,
)
from feax4d.utils import (
    save_result,
    load_design,
    load_summary,
    split_design,
)
from feax4d.viz import (
    plot_print_layers,
    plot_print_paths,
    plot_print_layers_plotly,
    plot_print_paths_plotly,
)
from feax4d.gcode import (
    svg_to_gcode,
    svg_to_gcode_polymer_fill,
    fibre_paths_to_gcode,
    collect_layer_svgs,
    FibrifierParams,
    FibrifierTemperatureParams,
    FibrifierSpeedParams,
    FibrifierExtrusionParams,
    FibrifierFiberParams,
    FibrifierRetractionParams,
    FibrifierGcodeGenerator,
    FibrifierLayer,
    FibrifierModel,
)

__version__ = "0.1.0"

__all__ = [
    # materials
    "Lamina", "Polymer", "make_layer_constitutive",
    # shell / problem
    "BilayerThermalShell", "N_FIELDS", "make_bilayer_shell",
    "uniform_transverse_load", "edge_predicate", "cantilever_edges", "clamp_bc",
    # objectives
    "make_shape_match_fn", "ud_penalty_total",
    "rho_contrast_penalty_total", "mag_consistency_total",
    # optimisation
    "OptimizeConfig", "OptimizeResult", "optimize",
    "pack", "unpack", "initial_design", "bounds", "make_process_fn",
    # fibre paths
    "AlignedSHStep", "setup_aligned_sh", "run_aligned_sh", "run_aligned_sh_steps",
    "extract_fibre_paths", "extract_region_contours", "write_paths_svg",
    "director_from_a2", "generate_fibre_paths",
    # persistence
    "save_result", "load_design", "load_summary", "split_design",
    # visualisation
    "plot_print_layers", "plot_print_paths",
    "plot_print_layers_plotly", "plot_print_paths_plotly",
    # fibrifier g-code
    "svg_to_gcode", "svg_to_gcode_polymer_fill", "fibre_paths_to_gcode",
    "collect_layer_svgs",
    "FibrifierParams", "FibrifierTemperatureParams", "FibrifierSpeedParams",
    "FibrifierExtrusionParams", "FibrifierFiberParams", "FibrifierRetractionParams",
    "FibrifierGcodeGenerator", "FibrifierLayer", "FibrifierModel",
]
