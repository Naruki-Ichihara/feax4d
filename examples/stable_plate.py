"""Stable-plate optimisation, built explicitly with the general feax4d API.

A rectangular cantilever (span : width = 2 : 1) is fully clamped on its right
edge and carries a uniform downward line load on the free (left) edge.  The
optimiser tailors the bilayer density + fibre orientation so the cooling-
induced thermal warping cancels the deflection and the plate stays flat.

This is the same physical problem as the convenience defaults, but the mesh,
boundary conditions and load are supplied explicitly — swap them to solve a
different problem (see ``general_problem.py``).

    python examples/stable_plate.py
"""
from pathlib import Path

import jax.numpy as jnp

import feax as fe
import feax4d


def main():
    # ── Rectangular cantilever geometry ──
    Lx, Ly, Nx, Ny = 200.0e-3, 100.0e-3, 80, 40
    mesh = fe.mesh.rectangle_mesh(
        Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly, ele_type="QUAD4",
    )

    tol = 1e-6
    right = lambda p: jnp.isclose(p[0], Lx, atol=tol)    # clamped edge
    left = lambda p: jnp.isclose(p[0], 0.0, atol=tol)    # free (loaded) edge

    # ── Boundary conditions: fully clamp the right edge (u,v,w and θx,θy) ──
    bc_specs = [
        fe.DirichletBCSpec(location=right, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=right, component="all", value=0.0, variable_index=1),
    ]

    # ── Load: uniform transverse line load on the free (left) edge (5 N/m, −z) ──
    load_fn = feax4d.uniform_transverse_load(5.0)

    cfg = feax4d.OptimizeConfig(
        mesh=mesh,
        bc_specs=bc_specs,
        load_location_fns=(left,),
        surface_load_fns=[load_fn],
        filter_rho_radius=0.05 * Lx,        # absolute filter radii (m)
        filter_theta_radius=0.05 * Lx,
        delta_t=-150.0,                     # K, one-shot cooling
        target_fn=None,                     # None ⇒ flat target (stay-flat objective)
        max_iter=100,
        output_dir=Path(__file__).with_name("output_stable_plate"),
    )
    result = feax4d.optimize(cfg)
    print(f"\nbest objective = {result.best_obj:.6e} @ iter {result.best_iter} "
          f"({result.stop_reason})")
    print(f"history: {result.xdmf_path}")


if __name__ == "__main__":
    main()
