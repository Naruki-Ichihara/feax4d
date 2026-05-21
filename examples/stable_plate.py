"""Stable-plate optimisation via the feax4d high-level API.

A cantilever clamped on its right edge (span : width = 2 : 1) carries a
downward line load on the free edge; the optimiser tailors the bilayer
density + fibre orientation so the cooling-induced thermal warping cancels
the deflection and the plate stays flat.

    python examples/stable_plate.py
"""
from pathlib import Path

import feax4d


def main():
    cfg = feax4d.OptimizeConfig(
        Lx=200.0e-3, Ly=100.0e-3, Nx=80, Ny=40,
        clamp="right",            # free edge (loaded) is the left edge
        load_mag=5.0,             # N/m, downward on the free edge
        delta_t=-150.0,           # K, one-shot cooling
        target_fn=None,           # None ⇒ flat target (stay-flat objective)
        max_iter=100,
        output_dir=Path(__file__).with_name("output_stable_plate"),
    )
    result = feax4d.optimize(cfg)
    print(f"\nbest objective = {result.best_obj:.6e} @ iter {result.best_iter} "
          f"({result.stop_reason})")
    print(f"history: {result.xdmf_path}")


if __name__ == "__main__":
    main()
