"""Material models for the bilayer thermal laminate.

Two ingredients build every design layer:

* :class:`Lamina`  — an orthotropic fibre-reinforced lamina (fibre = local
  1-axis), the high-stiffness / low-CTE phase.
* :class:`Polymer` — an isotropic polymer matrix, the soft / high-CTE phase
  that fills the low-density regions (no SIMP void).

:func:`make_layer_constitutive` blends the two by a Heaviside-projected
density via a linear rule of mixtures, returning the per-quad-point effective
in-plane stiffness ``C_eff``, thermal-expansion tensor ``α_eff`` and
transverse-shear tensor ``G_s_eff`` from the libertas orientation parameters
``(ρ̃, x1, x2, x3)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as np

from feax.mechanics.shell import isotropic_in_plane_stiffness
from feax.mechanics.orientation import (
    orientation_tensor_2d,
    quadratic_closure,
    orientation_averaged_stiffness,
)


@dataclass(frozen=True)
class Lamina:
    """Orthotropic fibre-reinforced lamina (fibre direction = local 1-axis).

    Defaults are a typical CFRP lamina.  ``alpha_fibre`` is the (small,
    possibly negative) longitudinal CTE; ``alpha_trans`` the large
    transverse CTE — their asymmetry, plus the bilayer stacking, is what
    drives the thermal warping.
    """

    E1: float = 140.0e9
    E2: float = 10.0e9
    G12: float = 5.0e9
    nu12: float = 0.30
    G13: float = 5.0e9
    G23: float = 3.0e9
    alpha_fibre: float = -0.5e-6
    alpha_trans: float = 30.0e-6
    thickness: float = 0.5e-3


@dataclass(frozen=True)
class Polymer:
    """Isotropic polymer matrix filling the low-density regions.

    Defaults are a typical thermoplastic (soft, large isotropic CTE).
    """

    E: float = 2.0e9
    nu: float = 0.40
    alpha: float = 70.0e-6

    @property
    def G(self) -> float:
        """Shear modulus ``E / (2(1+ν))``."""
        return self.E / (2.0 * (1.0 + self.nu))


def make_layer_constitutive(lamina: Lamina, polymer: Polymer, sgn_beta: float = 10.0):
    """Build the per-layer constitutive blend function.

    Returns a pure function ``layer(rho_p, x1, x2, x3) -> (C_eff, α_eff,
    Gs_eff)`` for one layer at one quadrature point, where ``rho_p`` is the
    SIMP/Heaviside-projected density and ``(x1, x2, x3)`` the libertas
    orientation parameters.  Linear rule-of-mixtures blend between the
    isotropic polymer matrix and the orientation-averaged fibre stiffness /
    CTE (both expressed in laminate axes)::

        C_eff = (1 − ρ̃)·C_poly + ρ̃·C_fibre(a₂)
        α_eff = (1 − ρ̃)·α_poly·I + ρ̃·α_fibre(a₂)
        Gs_eff = (1 − ρ̃)·G_poly·I + ρ̃·diag(G13, G23)
    """
    C_poly = isotropic_in_plane_stiffness(polymer.E, polymer.nu)        # (2,2,2,2)
    G_poly_2x2 = polymer.G * np.eye(2)
    G_fibre_2x2 = np.diag(np.array([lamina.G13, lamina.G23]))
    alpha_poly_tensor = polymer.alpha * np.eye(2)

    def layer(rho_p, x1, x2, x3):
        a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
        a4 = quadratic_closure(a2)
        C_fibre = orientation_averaged_stiffness(
            a2, a4, E1=lamina.E1, E2=lamina.E2, G12=lamina.G12, nu12=lamina.nu12,
        )
        # α(a₂) = α_t·I + (α_f − α_t)·a₂  in laminate axes.
        alpha_fibre = (
            lamina.alpha_trans * np.eye(2)
            + (lamina.alpha_fibre - lamina.alpha_trans) * a2
        )
        one_minus = 1.0 - rho_p
        C_eff = one_minus * C_poly + rho_p * C_fibre
        alpha_eff = one_minus * alpha_poly_tensor + rho_p * alpha_fibre
        Gs_eff = one_minus * G_poly_2x2 + rho_p * G_fibre_2x2
        return C_eff, alpha_eff, Gs_eff

    return layer
