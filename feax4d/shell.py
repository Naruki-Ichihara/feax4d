"""Two-layer thermomechanical Mindlin/FSDT shell problem.

:class:`BilayerThermalShell` is a :class:`feax.Problem` whose material at
each quadrature point is a per-layer density + fibre-orientation design
(8 node-based scalar fields).  Cooling ``ΔT`` is applied in one shot as a
thermal eigenstrain (linear theory), and an optional uniform transverse
line load is applied on a free edge via the surface weak form.

The constitutive blend, thicknesses, ``ΔT`` and load magnitude are passed
through ``additional_info`` (hashable scalars / frozen dataclasses) so the
problem stays a valid JAX pytree.
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as np

import feax as fe
from feax.mechanics.shell import (
    laminate_stiffness,
    laminate_thermal_loads,
    mindlin_strains,
    mindlin_resultants,
)

from feax4d.materials import Lamina, Polymer, make_layer_constitutive


N_FIELDS = 8   # (rho, x1, x2, x3) × 2 layers


class BilayerThermalShell(fe.Problem):
    """Per-quad-point bilayer laminate with density + orientation per layer.

    ``volume_vars`` ordering (8 node-based scalar fields)::

        (rho0, x1_0, x2_0, x3_0,  rho1, x1_1, x2_1, x3_1)

    Linear strains (no von Kármán); the cooling ``ΔT`` is applied in one
    shot via :func:`feax.mechanics.shell.laminate_thermal_loads`.  A uniform
    transverse line load ``load_mag`` (physical −z) is applied on whatever
    edge is registered through ``location_fns``.

    Built with ``additional_info=(lamina, polymer, thicks, delta_t,
    load_mag, sgn_beta)`` — see :func:`make_bilayer_shell`.
    """

    def custom_init(self, lamina, polymer, thicks, delta_t, load_mag, sgn_beta):
        self.lamina = lamina
        self.polymer = polymer
        self.thicks = np.asarray(thicks)
        self.zero_thetas = np.zeros(len(thicks))
        self.delta_t = delta_t
        self.load_mag = load_mag
        self.sgn_beta = sgn_beta
        self.layer_fn = make_layer_constitutive(lamina, polymer, sgn_beta)

    def get_weak_form(self):
        layer_fn = self.layer_fn
        thicks = self.thicks
        zero_thetas = self.zero_thetas
        delta_t = self.delta_t

        def weak_form(vals, grads, x,
                      rho0, x1_0, x2_0, x3_0,
                      rho1, x1_1, x2_1, x3_1):
            C0, a0, Gs0 = layer_fn(rho0, x1_0, x2_0, x3_0)
            C1, a1, Gs1 = layer_fn(rho1, x1_1, x2_1, x3_1)
            C_layers = np.stack([C0, C1])
            G_layers = np.stack([Gs0, Gs1])
            alpha_layers = np.stack([a0, a1])

            A, B, D, G_s = laminate_stiffness(
                C_layers, G_layers, zero_thetas, thicks,
            )
            N_T, M_T = laminate_thermal_loads(
                C_layers, alpha_layers, zero_thetas, thicks,
                dT_avg=delta_t, dT_grad=0.0,
            )

            theta = vals[1]
            eps, kappa, gamma = mindlin_strains(
                grads[0], grads[1], theta, nonlinear="linear",
            )
            N, M, Q = mindlin_resultants(
                eps, kappa, gamma, A, D, G_s, B=B, N_T=N_T, M_T=M_T,
            )

            # Body: thermal eigenstrain only.  The transverse line load is
            # applied in get_surface_weak_forms.
            grad0 = np.concatenate([N, Q[None, :]], axis=0)
            mass0 = np.zeros(3)
            grad1 = M
            mass1 = Q
            return ([mass0, mass1], [grad0, grad1])

        return weak_form

    def get_surface_weak_forms(self):
        # Uniform transverse line load on the registered (free) edge,
        # physical direction −z.  In feax the residual integrand is
        # ``+= t = −t_phys``, so a downward load ``t_phys = −load_mag`` is
        # written ``tz = +load_mag`` on the w-component of variable 0.
        load_mag = self.load_mag

        def edge_load(vals, x, *iv):
            return [np.array([0.0, 0.0, load_mag]), np.zeros(2)]

        return [edge_load]


def make_bilayer_shell(
    mesh,
    lamina: Lamina,
    polymer: Polymer,
    delta_t: float,
    load_mag: float,
    sgn_beta: float = 10.0,
    location_fns: Tuple[Callable, ...] = (),
) -> BilayerThermalShell:
    """Construct a two-variable (vec=[3,2]) bilayer thermal shell on ``mesh``.

    ``location_fns`` selects the edge(s) where the transverse line load is
    applied (typically the single free edge of a cantilever).
    """
    thicks = (float(lamina.thickness), float(lamina.thickness))
    return BilayerThermalShell(
        mesh=[mesh, mesh], vec=[3, 2], dim=2,
        ele_type=[mesh.ele_type, mesh.ele_type],
        location_fns=tuple(location_fns),
        additional_info=(lamina, polymer, thicks, delta_t, load_mag, sgn_beta),
    )


# ── Geometry / boundary helpers ──────────────────────────────────────────────

_EDGE_OPPOSITE = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}


def edge_predicate(edge: str, Lx: float, Ly: float, tol: float):
    """Return a location predicate ``pt -> bool`` for one rectangle edge."""
    if edge == "left":
        return lambda pt: np.isclose(pt[0], 0.0, atol=tol)
    if edge == "right":
        return lambda pt: np.isclose(pt[0], Lx, atol=tol)
    if edge == "bottom":
        return lambda pt: np.isclose(pt[1], 0.0, atol=tol)
    if edge == "top":
        return lambda pt: np.isclose(pt[1], Ly, atol=tol)
    raise ValueError(f"unknown edge {edge!r} (use right/left/top/bottom)")


def cantilever_edges(clamp: str, Lx: float, Ly: float, tol: float):
    """Return ``(clamped_edge_fn, free_edge_fn)`` for a cantilever.

    ``clamp`` names the fully-clamped edge; the load is applied on the
    opposite edge.
    """
    free = _EDGE_OPPOSITE[clamp]
    return (
        edge_predicate(clamp, Lx, Ly, tol),
        edge_predicate(free, Lx, Ly, tol),
    )


def clamp_bc(problem, clamped_edge) -> fe.DirichletBC:
    """Fully clamp one edge: all 5 DOFs (u,v,w and θx,θy) zero."""
    return fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=clamped_edge, component="all", value=0.0,
                           variable_index=0),
        fe.DirichletBCSpec(location=clamped_edge, component="all", value=0.0,
                           variable_index=1),
    ]).create_bc(problem)
