"""Interactive 3D visualisation of the printed toolpaths.

Builds the polymer-filled, multi-lamina print stack from the fibre paths of a
stable-plate run and renders it in 3D: fibre toolpaths (colour-coded per path)
and polymer-infill toolpaths, stacked at their print Z heights.  Opens an
interactive window — **drag with the mouse to rotate / zoom**.

    python examples/visualize.py

Run ``stable_plate.py`` and ``fibre_paths.py`` first to produce the fibre paths.
"""
from pathlib import Path

import matplotlib.pyplot as plt   # default (interactive) backend — mouse-rotatable

import feax4d


def main():
    fibre_dir = Path(__file__).with_name("output_stable_plate") / "fibre_paths"
    if not fibre_dir.exists():
        raise FileNotFoundError(
            f"{fibre_dir} not found — run examples/stable_plate.py then "
            f"examples/fibre_paths.py first."
        )

    # Assemble the print layers (also writes the g-code); returns the layer stack.
    params = feax4d.FibrifierParams()
    params.layer_height = 0.15
    result = feax4d.fibre_paths_to_gcode(
        fibre_dir,
        params=params,
        polymer_fill=True,
        infill_angle=[0.0, 90.0],
        infill_pitch=1.0,
        layer_print_layers=[3, 3],     # 3 print laminae per design layer
        connection_threshold=10.0,
    )

    fig = feax4d.plot_print_paths(result, elev=24, azim=-58)
    out = fibre_dir / "print_paths_3d.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"layers={result['n_layers']}  fibre toolpaths={result['n_fiber_paths']}  "
          f"polymer toolpaths={result['n_polymer_paths']}")
    print(f"Saved 3D print-path view to {out}")

    # Interactive window: drag to rotate, scroll to zoom (needs a GUI backend;
    # no-op under a headless/Agg backend).
    plt.show()


if __name__ == "__main__":
    main()
