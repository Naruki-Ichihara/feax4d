"""Fibre-path generation (+ polymer-filled G-code) via the feax4d API.

Reads the optimisation history written by ``stable_plate.py``, writes the
per-layer fibre paths (SVG / NPZ / VTU / PNG), then converts them into a
Fibrifier G-code where each layer's footprint is filled with polymer infill
(P) and the fibres are laid on the adjacent fibre layer (F) — polymer
everywhere the fibre isn't.

    python examples/fibre_paths.py
"""
from pathlib import Path

import feax4d


def main():
    history = Path(__file__).with_name("output_stable_plate") / "history.xdmf"
    if not history.exists():
        raise FileNotFoundError(
            f"{history} not found — run examples/stable_plate.py first."
        )

    # 1) Aligned-SH fibre paths (writes <history dir>/fibre_paths/*.svg).
    feax4d.generate_fibre_paths(
        xdmf_path=history,
        stripe_period=3.0e-3,     # fibre / tow spacing [m]
        refine_factor=4,
        n_steps=600,
        rho_cutoff=0.5,
    )
    fibre_dir = history.parent / "fibre_paths"


if __name__ == "__main__":
    main()
