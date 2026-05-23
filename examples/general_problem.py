"""General mesh / BC / load with the feax4d optimisation API.

The high-level `optimize` is not limited to the rectangular cantilever: pass
your own ``mesh`` (any feax mesh), Dirichlet BCs (``bc_specs`` or ``bc_fn``)
and Neumann loads (``load_location_fns`` + ``surface_load_fns``).

This example optimises a doubly-clamped plate (both short edges fully fixed)
with a custom transverse load applied on a central strip, asking it to stay
flat (``target_fn=None``).

    python examples/general_problem.py
"""
from pathlib import Path

import jax.numpy as jnp

import feax as fe
import feax4d


def main():
    Lx, Ly = 0.24, 0.12
    mesh = fe.mesh.rectangle_mesh(Nx=48, Ny=24, domain_x=Lx, domain_y=Ly, ele_type="QUAD4")

    tol = 1e-6
    left = lambda p: jnp.isclose(p[0], 0.0, atol=tol)
    right = lambda p: jnp.isclose(p[0], Lx, atol=tol)

    # Boundary conditions: fully clamp both short edges (5 DOFs each, var 0 & 1).
    bc_specs = [
        fe.DirichletBCSpec(location=e, component="all", value=0.0, variable_index=v)
        for e in (left, right) for v in (0, 1)
    ]

    # Load: transverse line load on a central strip, magnitude varying with y.
    centre_strip = lambda p: jnp.isclose(p[0], 0.5 * Lx, atol=0.5 * Lx / 48)

    def strip_load(vals, x, *iv):
        tz = 10.0 * (x[1] - 0.5 * Ly) / (0.5 * Ly)   # anti-symmetric in y
        return [jnp.array([0.0, 0.0, tz]), jnp.zeros(2)]

    cfg = feax4d.OptimizeConfig(
        mesh=mesh,
        bc_specs=bc_specs,
        load_location_fns=(centre_strip,),
        surface_load_fns=[strip_load],
        filter_rho_radius=0.012,        # absolute filter radii (m) for a general mesh
        filter_theta_radius=0.012,
        target_fn=None,                 # stay flat
        delta_t=-150.0,
        max_iter=60,
        output_dir=Path(__file__).with_name("output_general_problem"),
    )
    result = feax4d.optimize(cfg)
    print(f"\nbest obj = {result.best_obj:.4e} @ iter {result.best_iter} "
          f"({result.stop_reason})")
    print(f"history: {result.xdmf_path}")


if __name__ == "__main__":
    main()
