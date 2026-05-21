"""Objective and regularisation terms for the bilayer shell optimisation.

* :func:`make_shape_match_fn` — the primary objective: squared L2 distance
  of the transverse displacement ``w`` from a target field, normalised by a
  caller-supplied denominator.  A flat (``w_target ≡ 0``) target gives the
  "stay-flat" load-compensation objective; a non-zero target gives shape
  matching / form finding.

* Three regularisers that push the design toward a clean manufacturable
  state, all returning O(1) scalars (see the libertas formulation):
    - :func:`ud_penalty_total`        — density-weighted rank-1 (UD) penalty
    - :func:`rho_contrast_penalty_total` — grey-scale → {0,1} penalty
    - :func:`mag_consistency_total`   — couples orientation magnitude to ρ
"""
from __future__ import annotations

import jax
import jax.numpy as np

from feax.mechanics.orientation import orientation_tensor_2d


def make_shape_match_fn(problem, w_target_flat, denom):
    """Return ``J(sol) = ‖w − w_target‖² / denom`` (transverse only).

    ``denom`` is supplied by the caller (e.g. ‖w_init‖² of the initial
    design) so the objective is O(1); with ``w_target ≡ 0`` this is the
    stay-flat objective ‖w‖² / denom.
    """
    denom = np.maximum(denom, 1e-30)

    @jax.jit
    def shape_match_fn(sol_flat):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        w = sol_list[0][:, 2]
        diff = w - w_target_flat
        return np.sum(diff * diff) / denom

    return shape_match_fn


# ── UD / contrast / magnitude-consistency penalties ─────────────────────────

def _ud_penalty_one(x1, x2, x3, sgn_beta):
    """4·det(a₂) / trace(a₂)² for one node — 0 = UD (rank-1), 1 = isotropic."""
    a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
    tr = a2[0, 0] + a2[1, 1]
    det = a2[0, 0] * a2[1, 1] - a2[0, 1] * a2[0, 1]
    return 4.0 * det / (tr * tr + 1e-12)


def ud_penalty_total(design8, rho0_f, rho1_f, sgn_beta):
    """Density-weighted ``P_UD`` averaged across both layers (0 = UD where ρ>0).

    Weighted by raw filtered density so the UD enforcement acts only where
    there is material — void regions (ρ ≈ 0) do not contribute.
    """
    vmap = jax.vmap(lambda a, b, c: _ud_penalty_one(a, b, c, sgn_beta))
    p0 = vmap(design8[1], design8[2], design8[3])
    p1 = vmap(design8[5], design8[6], design8[7])
    w0 = np.sum(rho0_f) + 1e-12
    w1 = np.sum(rho1_f) + 1e-12
    return 0.5 * (np.sum(rho0_f * p0) / w0 + np.sum(rho1_f * p1) / w1)


def rho_contrast_penalty_total(rho0_f, rho1_f):
    """Mean ``4·ρ·(1−ρ)`` across both layers — pushes ρ → {0, 1}."""
    p0 = 4.0 * rho0_f * (1.0 - rho0_f)
    p1 = 4.0 * rho1_f * (1.0 - rho1_f)
    return 0.5 * (np.mean(p0) + np.mean(p1))


def _mag_consistency_one(x1, x2, x3, rho, sgn_beta):
    """``(|T| − ρ)²`` for one node, ``|T| = √((a₁₁−a₂₂)² + 4a₁₂²) ∈ [0,1]``."""
    a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
    Tx = a2[0, 0] - a2[1, 1]
    Ty = 2.0 * a2[0, 1]
    T_mag = np.sqrt(Tx * Tx + Ty * Ty + 1e-30)
    return (T_mag - rho) ** 2


def mag_consistency_total(design8, rho0_f, rho1_f, sgn_beta):
    """Mean ``(|T| − ρ)²`` across both layers — kills density/magnitude mismatch."""
    vmap = jax.vmap(lambda a, b, c, r: _mag_consistency_one(a, b, c, r, sgn_beta))
    p0 = vmap(design8[1], design8[2], design8[3], rho0_f)
    p1 = vmap(design8[5], design8[6], design8[7], rho1_f)
    return 0.5 * (np.mean(p0) + np.mean(p1))
