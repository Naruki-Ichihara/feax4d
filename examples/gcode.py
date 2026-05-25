"""Fibrifier G-code generation via the feax4d high-level API.

Takes the per-layer fibre-path SVGs written by ``fibre_paths.py`` and emits a
single multi-layer Fibrifier (9T Labs) G-code file plus a preview PNG.

    python examples/gcode.py
"""
from pathlib import Path

import feax4d


def main():
    # generate_fibre_paths writes to <history dir>/fibre_paths by default.
    fibre_dir = Path(__file__).with_name("output_stable_plate") / "fibre_paths"
    if not fibre_dir.exists():
        raise FileNotFoundError(
            f"{fibre_dir} not found — run examples/fibre_paths.py first."
        )

    params = feax4d.FibrifierParams()
    params.temperature.bed_temperature = 90      # °C
    params.layer_height = 0.15                    # mm

    result = feax4d.fibre_paths_to_gcode(
        params=params,
        polymer_fill=True,                        # fill every layer's footprint with polymer
        infill_angle=[0.0, 90.0],                 # per-layer polymer scan direction
        infill_pitch=1.0,                         # polymer line spacing [mm]
        connection_threshold=3,                # merge fibre path ends within 10 mm
    )
    print("\nG-code:", result["gcode_path"])
    print(f"layers={result['n_layers']}  fibre paths={result['n_fiber_paths']}  "
          f"polymer paths={result.get('n_polymer_paths')}  "
          f"total fibre={result['total_fiber_mm']:.0f} mm")


if __name__ == "__main__":
    main()
