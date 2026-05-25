"""Verification case: a 100 x 50 mm plate with straight fibre paths along the
long axis — built analytically with svgpathtools, no optimisation needed.

It exercises the downstream pipeline (SVG parse → polymer fill → Fibrifier
g-code → 3D view) on a known, simple input.

    python examples/verify_paths.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

import feax4d


def main():
    out_dir = Path(__file__).with_name("output_verify")
    svg = out_dir / "fibre_paths_layer0.svg"

    # 100 x 50 mm rectangle; straight fibres along the long (100 mm) axis,
    # spaced 4 mm across the 50 mm width.
    feax4d.make_test_fibre_svg(svg, width=100.0, height=50.0, pitch=2.0, kind="snake")
    print(f"wrote {svg}")

    # ── Print parameters (edit these here) ──
    params = feax4d.FibrifierParams()
    params.layer_height = 0.10                       # mm
    params.offset_x = 175.0                          # machine X offset (mm)
    params.offset_y = 135.0                          # machine Y offset (mm)
    # temperatures [°C]
    params.temperature.cf_print_temp = 230           # fibre extruder
    params.temperature.pl_print_temp = 230           # polymer nozzle
    params.temperature.bed_temperature = 50
    params.temperature.build_chamber_temperature = 20
    # speeds
    params.speed.cf_print_feedrate = 600             # fibre feedrate [mm/min]
    params.speed.perimeter_speed = 40
    params.speed.rapid_move_feedrate = 5500
    # extrusion multipliers
    params.extrusion.cf_extrusion_multiplier = 0.99  # fibre
    params.extrusion.pl_extrusion_multiplier = 0.68  # polymer
    # fibre settings
    params.fiber.fiber_width = 1.5                   # mm
    params.fiber.minimal_printable_length = 23.73    # mm
    params.fiber.nozzle_dead_length = 21.5           # mm
    # (params.fiber.fiber_cut is overridden by the fiber_cut= kwarg below)

    # SVG -> polymer-filled Fibrifier g-code.
    result = feax4d.fibre_paths_to_gcode(
        [str(svg)],
        output_gcode=str(out_dir / "verify.gcode"),
        params=params,            # ← in-file print parameters
        polymer_fill=True,
        infill_angle=90.0,        # polymer infill across the fibres
        infill_pitch=0.8,
        polymer_base_layers=2,    # 2 polymer-only base layers
        layer_print_layers=10,    # 10 fibre layers
        polymer_top_layers=2,     # 2 polymer-only top layers
        polymer_in_fiber_layers=False,  # fibre-only middle layers (polymer only in caps)
        connection_threshold=4.0,
        fiber_cut=False,          # continuous fibre — no cutting; climb Z between
                                  # fibre layers at the same point (reversed laminae)
    )
    print(f"layers={result['n_layers']}  fibre paths={result['n_fiber_paths']}  "
          f"polymer paths={result['n_polymer_paths']}  "
          f"total fibre={result['total_fiber_mm']:.0f} mm")

    # 3D print-path view.
    fig = feax4d.plot_print_paths(result, elev=22, azim=-60, layer_gap=6.0)
    png = out_dir / "verify_print_paths_3d.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"saved {png}")
    plt.show()


if __name__ == "__main__":
    main()
