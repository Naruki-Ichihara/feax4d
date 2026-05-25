"""Shape matching to a sinusoidal *dimple* target (thermal only, no load).

Same plate as ``stable_plate.py`` (200 x 100 mm, 80 x 40 QUAD4).  The bilayer is
asked to morph — driven purely by the one-shot cooling, **no mechanical load** —
into a grid of dimples whose amplitude is **zero at the clamped left edge and
grows linearly toward the right**:

    w*(x, y) = A · (x / Lx) · sin(Nx·π·x/Lx) · sin(Ny·π·y/Ly)

Setup:
* mesh: rectangle 200 x 100 mm, 80 x 40 (same as stable_plate.py)
* BC: the **left** edge is fully clamped
* load: none (thermal eigenstrain from ΔT is the only driver)
* target: linearly-ramped dimple field (0 amplitude at x=0)

The optimiser tailors the per-layer density + fibre orientation so the cooling-
induced warping reproduces the dimple field in least-squares (displacement)
sense.

    python examples/dimple_target.py
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as onp

import feax as fe
import feax4d


# Geometry (same as stable_plate.py).
LX, LY, NX, NY = 200.0e-3, 100.0e-3, 80, 40

# Dimple target: product of sine waves, amplitude ramped linearly in x.
AMP = 3.0e-3          # peak amplitude at the right edge [m]
N_WAVES_X = 4         # half-waves along x
N_WAVES_Y = 2         # half-waves along y


def dimple_target(x, y):
    """Target transverse displacement w*(x, y) [m].

    A sin·sin dimple grid whose amplitude is 0 at the clamped left edge
    (x = 0) and increases linearly to ``AMP`` at the right edge (x = Lx).
    """
    envelope = x / LX                       # 0 at left edge → 1 at right edge
    return AMP * envelope \
        * onp.sin(N_WAVES_X * onp.pi * x / LX) \
        * onp.sin(N_WAVES_Y * onp.pi * y / LY)


def main():
    mesh = fe.mesh.rectangle_mesh(
        Nx=NX, Ny=NY, domain_x=LX, domain_y=LY, ele_type="QUAD4",
    )

    tol = 1e-6
    left = lambda p: jnp.isclose(p[0], 0.0, atol=tol)    # clamped edge

    # Boundary conditions: fully clamp the left edge (u,v,w and θx,θy).
    bc_specs = [
        fe.DirichletBCSpec(location=left, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=left, component="all", value=0.0, variable_index=1),
    ]

    cfg = feax4d.OptimizeConfig(
        mesh=mesh,
        bc_specs=bc_specs,
        load_location_fns=(),               # ← no mechanical load (thermal only)
        filter_rho_radius=0.05 * LX,
        filter_theta_radius=0.05 * LX,
        delta_t=-150.0,
        target_fn=dimple_target,            # ramped dimple shape-matching objective
        max_iter=100,
        output_dir=Path(__file__).with_name("output_dimple_target"),
    )
    result = feax4d.optimize(cfg)
    print(f"\nbest objective = {result.best_obj:.6e} @ iter {result.best_iter} "
          f"({result.stop_reason})")
    print(f"history: {result.xdmf_path}")


if __name__ == "__main__":
    main()
