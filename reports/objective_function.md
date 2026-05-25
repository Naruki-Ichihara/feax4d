# feax4d: Bilayer Thermal Shell — Physics and Objective Function

This report documents the physics solved by `feax4d` (the forward bilayer
thermomechanical shell problem, in `feax4d/shell.py` and `feax4d/materials.py`)
and the objective (loss) function minimised by `feax4d.optimize` (in
`feax4d/objectives.py`, assembled in `feax4d/optimize.py`). The optimiser
(NLopt MMA) drives the per-layer density and fibre-orientation design fields so
that a cooled bilayer shell deforms into a prescribed target shape, while three
regularisers keep the design manufacturable. Each design evaluation solves one
linear shell problem and differentiates it by the adjoint method.

## Design variables

Each of the two laminate layers $k \in \{0, 1\}$ carries four node-based scalar
fields, packed as eight fields total:

$$ \big(\rho_k,\; x_{1,k},\; x_{2,k},\; x_{3,k}\big), \qquad k = 0, 1 . $$

$\rho_k$ is a density and $(x_{1,k}, x_{2,k}, x_{3,k})$ are the libertas
orientation parameters that build the 2-D Advani–Tucker orientation tensor
$\mathbf{a}_2$. Before they enter the physics, the raw fields are passed
through a Helmholtz filter (giving $\rho_{f}$ and the filtered orientation
parameters) and the density is SIMP-projected, $\tilde\rho = \rho_{f}^{\,p}$,
inside the rule-of-mixtures stiffness/CTE blend.

## The forward problem: a bilayer thermomechanical shell

Each design evaluation solves one linear first-order shear-deformation (FSDT /
Mindlin–Reissner) shell problem with two field variables on the mesh:

- variable 0: midplane displacement $\mathbf{u}_0 = (u, v, w)$ (in-plane $u, v$ and transverse deflection $w$);
- variable 1: section rotations $\boldsymbol{\theta} = (\theta_x, \theta_y)$.

### Kinematics (small strain)

The membrane strain, curvature and transverse-shear strain are

$$ \boldsymbol{\varepsilon} = \tfrac12\big(\nabla\mathbf{u}_{\text{in}} + \nabla\mathbf{u}_{\text{in}}^\top\big), \qquad \boldsymbol{\kappa} = \tfrac12\big(\nabla\boldsymbol\theta + \nabla\boldsymbol\theta^\top\big), \qquad \boldsymbol{\gamma} = \nabla w + \boldsymbol\theta, $$

with $\mathbf{u}_{\text{in}} = (u, v)$. Linear theory is used (no von Kármán),
so there is one linear factorisation per design evaluation.

### Per-layer constitutive law (rule of mixtures)

Each layer $k$ blends an isotropic polymer matrix with the orientation-averaged
fibre stiffness, weighted by the SIMP-projected density
$\tilde\rho_k = \rho_{f,k}^{\,p}$:

$$ \mathbf{C}_k = (1 - \tilde\rho_k)\,\mathbf{C}_{\text{poly}} + \tilde\rho_k\,\mathbf{C}_{\text{fibre}}(\mathbf{a}_{2,k}), $$

$$ \boldsymbol{\alpha}_k = (1 - \tilde\rho_k)\,\alpha_{\text{poly}}\mathbf{I} + \tilde\rho_k\,\boldsymbol{\alpha}_{\text{fibre}}(\mathbf{a}_{2,k}), \qquad \boldsymbol{\alpha}_{\text{fibre}} = \alpha_t\,\mathbf{I} + (\alpha_f - \alpha_t)\,\mathbf{a}_2, $$

with an analogous blend for the transverse-shear modulus $\mathbf{G}_{s,k}$. The
fibre stiffness $\mathbf{C}_{\text{fibre}}(\mathbf{a}_2)$ is the Advani–Tucker
orientation-averaged orthotropic lamina stiffness (quadratic closure
$\mathbf{a}_4 = \mathbf{a}_2 \otimes \mathbf{a}_2$). Where $\rho \to 0$ the layer
becomes the polymer matrix (a "polymer floor", not SIMP void), so both layers
are always physically present and the laminate is well-posed everywhere.

### Laminate stiffness (CLT integrals)

With two equal-thickness layers ($t$ each, midplane at $z = 0$, interfaces
$z \in \{-t, 0, t\}$), the extensional, coupling, bending and transverse-shear
stiffnesses are the through-thickness moments of $\mathbf{C}_k$:

$$ \mathbf{A} = \sum_k \mathbf{C}_k (z_{k+1} - z_k), \quad \mathbf{B} = \tfrac12\sum_k \mathbf{C}_k (z_{k+1}^2 - z_k^2), \quad \mathbf{D} = \tfrac13\sum_k \mathbf{C}_k (z_{k+1}^3 - z_k^3), $$

$$ \mathbf{G}_s = \kappa_s \sum_k \mathbf{G}_{s,k} (z_{k+1} - z_k), \qquad \kappa_s = \tfrac56 . $$

A difference between the two layers (density and/or orientation) makes the
coupling stiffness $\mathbf{B} \neq 0$ — the mechanism behind thermal warping.

### Thermal eigenload

The plate is cooled by a uniform $\Delta T$ in one shot. The thermal force and
moment resultants are

$$ \mathbf{N}_T = \sum_k (\mathbf{C}_k : \boldsymbol{\alpha}_k) \int_{z_k}^{z_{k+1}} \Delta T \, dz, \qquad \mathbf{M}_T = \sum_k (\mathbf{C}_k : \boldsymbol{\alpha}_k) \int_{z_k}^{z_{k+1}} z\,\Delta T \, dz . $$

A midplane-symmetric laminate has $\mathbf{M}_T = 0$ and stays flat; the
optimiser deliberately breaks that symmetry to generate a tailored
$\mathbf{M}_T$ (hence curvature) at each point.

### Resultants and weak form

The linear constitutive resultants (thermal terms subtracted) are

$$ \mathbf{N} = \mathbf{A}:\boldsymbol\varepsilon + \mathbf{B}:\boldsymbol\kappa - \mathbf{N}_T, \quad \mathbf{M} = \mathbf{B}:\boldsymbol\varepsilon + \mathbf{D}:\boldsymbol\kappa - \mathbf{M}_T, \quad \mathbf{Q} = \mathbf{G}_s\,\boldsymbol\gamma . $$

The equilibrium weak form solved for $(\mathbf{u}_0, \boldsymbol\theta)$ — for
all admissible test fields $(\delta\mathbf{u}_0, \delta\boldsymbol\theta)$ — is

$$ \int_\Omega \Big[\, \mathbf{N} : \nabla_{\!s}\delta\mathbf{u}_{\text{in}} + \mathbf{Q}\cdot\nabla\delta w + \mathbf{M} : \nabla_{\!s}\delta\boldsymbol\theta + \mathbf{Q}\cdot\delta\boldsymbol\theta \,\Big]\,d\Omega \;=\; \int_{\Gamma_N} \mathbf{t}\cdot\delta\mathbf{u}_0 \, d\Gamma . $$

There is no body load; the only external load is the surface traction
$\mathbf{t}$. Per quadrature point the weak form is:

```python
A, B, D, G_s = laminate_stiffness(C_layers, G_layers, ZERO_THETAS, thicks)
N_T, M_T = laminate_thermal_loads(C_layers, alpha_layers, ZERO_THETAS, thicks,
                                  dT_avg=delta_t, dT_grad=0.0)
eps, kappa, gamma = mindlin_strains(grads[0], grads[1], vals[1], nonlinear="linear")
N, M, Q = mindlin_resultants(eps, kappa, gamma, A, D, G_s, B=B, N_T=N_T, M_T=M_T)
grad0 = np.concatenate([N, Q[None, :]], axis=0)   # flux for (u, v, w)
return ([np.zeros(3), Q], [grad0, M])             # mass: 0 for u; Q couples to theta
```

### Boundary conditions and loads

- **Dirichlet** (caller-supplied via `bc_specs` or `bc_fn`): e.g. a clamped edge fixes all five DOFs, $u = v = w = 0$ and $\theta_x = \theta_y = 0$.
- **Neumann** (caller-supplied via `load_location_fns` + `surface_load_fns`): transverse surface tractions $\mathbf{t}(\mathbf{x}) = (0, 0, t_z)$, which may be uniform, spatially varying (e.g. the line load that ramps across the width in `stable_plate.py`), or absent (thermal-only morphing, as in `dimple_target.py`). feax's sign convention puts $+\mathbf{t} = -\mathbf{t}_{\text{phys}}$ in the residual.

## Primary term: displacement error (shape matching)

Let $\mathbf{u}_i = (u, v, w)_i$ be the displacement of variable 0 at node $i$
(in-plane $u, v$ and transverse $w$), and let $\mathbf{u}^*_i$ be the target
displacement. The target is built from a transverse target field $w^*(x, y)$
with zero in-plane components:

$$ \mathbf{u}^*_i = (0,\; 0,\; w^*_i) . $$

The primary objective is the squared $L^2$ **displacement** error (not just the
transverse deflection), normalised by the same quantity evaluated on the
initial design $\mathbf{u}^{0}$:

$$ J_{\text{shape}} \;=\; \frac{\displaystyle\sum_i \lVert \mathbf{u}_i - \mathbf{u}^*_i \rVert^2}{\displaystyle\sum_i \lVert \mathbf{u}^{0}_i - \mathbf{u}^*_i \rVert^2} . $$

Two special cases:

- $w^* \equiv 0$ (no `target_fn`): the *stay-flat* objective $J_{\text{shape}} = \lVert \mathbf{u} \rVert^2 / \lVert \mathbf{u}^0 \rVert^2$ — the cooling-induced warping must cancel the load deflection and any in-plane motion, i.e. keep the original shape.
- $w^* \neq 0$: shape matching / form finding — the bilayer is tailored to morph into $w^*$ (e.g. a sinusoidal dimple field).

The normalisation makes $J_{\text{shape}} \approx 1$ at iteration 0, so the
objective is dimensionless and $O(1)$ regardless of amplitude or load. In code:

```python
def make_shape_match_fn(problem, u_target, denom):
    denom = np.maximum(denom, 1e-30)

    @jax.jit
    def shape_match_fn(sol_flat):
        uvw = problem.unflatten_fn_sol_list(sol_flat)[0]   # (n_nodes, 3)
        diff = uvw - u_target
        return np.sum(diff * diff) / denom

    return shape_match_fn
```

## Regularisation penalties

Three additive penalties push the design toward a clean, manufacturable state.
Each is an $O(1)$ scalar averaged over the two layers. They are evaluated on the
raw filtered density $\rho_{f}$ and the filtered orientation tensor
$\mathbf{a}_2$.

### Unidirectional (rank-1) penalty

For one node, with orientation tensor $\mathbf{a}_2$,

$$ p^{\text{UD}} \;=\; \frac{4\,\det \mathbf{a}_2}{(\operatorname{tr}\mathbf{a}_2)^2 + \varepsilon} \;\in\; [0, 1], $$

which is $0$ for a rank-1 (unidirectional) tensor and $1$ for an isotropic one.
The total is **density-weighted** so the UD enforcement only acts where there is
material (void regions do not contribute spurious UD states):

$$ P_{\text{UD}} \;=\; \frac{1}{2}\sum_{k \in \{0,1\}} \frac{\sum_i \rho_{f,k,i}\; p^{\text{UD}}_{k,i}}{\sum_i \rho_{f,k,i}} . $$

### Density grey-scale (contrast) penalty

Pushes the densities toward the binary set $\{0, 1\}$:

$$ P_{\rho} \;=\; \frac{1}{2}\sum_{k \in \{0,1\}} \operatorname{mean}_i\big[\, 4\,\rho_{f,k,i}\,(1 - \rho_{f,k,i}) \,\big] \;\in\; [0, 1]. $$

It is $0$ when every density is $0$ or $1$, and $1$ at the fully grey state
$\rho = \tfrac12$.

### Magnitude–density consistency penalty

Couples the orientation-tensor magnitude $|T|$ to the density, where

$$ |T| \;=\; \sqrt{(a_{11} - a_{22})^2 + 4\,a_{12}^2} \;\in\; [0, 1]. $$

The penalty removes "$\rho \approx 0$ but $|T| \approx 1$" and the reverse
mismatch:

$$ P_{\text{mag}} \;=\; \frac{1}{2}\sum_{k \in \{0,1\}} \operatorname{mean}_i\big[\, (|T|_{k,i} - \rho_{k,i})^2 \,\big]. $$

## Total objective

The quantity actually minimised is the displacement error plus the weighted
penalties:

$$ J \;=\; J_{\text{shape}} \;+\; \gamma_{\text{UD}}\, P_{\text{UD}} \;+\; \gamma_{\rho}\, P_{\rho} \;+\; \gamma_{\text{mag}}\, P_{\text{mag}} . $$

In `feax4d/optimize.py` this is assembled per design evaluation as:

```python
def objective_fn(x_flat):
    design8, rho0_f, rho1_f = process(x_flat)
    iv = fe.InternalVars(volume_vars=design8, surface_vars=())
    sol = solver(iv, init_guess)
    ud_val  = ud_penalty_total(design8, rho0_f, rho1_f, sb)
    rho_val = rho_contrast_penalty_total(rho0_f, rho1_f)
    mag_val = mag_consistency_total(design8, rho0_f, rho1_f, sb)
    loss = (
        shape_match_fn(sol)
        + cfg.ud_penalty * ud_val
        + cfg.rho_contrast_penalty * rho_val
        + cfg.mag_consistency_penalty * mag_val
    )
    return loss, (sol, ud_val, rho_val, mag_val)
```

The whole function is JIT-compiled and differentiated end-to-end with
`jax.value_and_grad`, so MMA receives exact gradients of $J$ with respect to all
$8 \times n_\text{nodes}$ design variables (the linear FE solve is
differentiated via the adjoint method inside the solver).

### Default penalty weights

| Symbol | `OptimizeConfig` field | Default | Role |
|--------|------------------------|---------|------|
| $\gamma_{\text{UD}}$ | `ud_penalty` | $1.0$ | density-weighted rank-1 (UD) enforcement |
| $\gamma_{\rho}$ | `rho_contrast_penalty` | $0.5$ | drive $\rho \to \{0, 1\}$ |
| $\gamma_{\text{mag}}$ | `mag_consistency_penalty` | $1.0$ | couple $|T|$ to $\rho$ |

## Monitoring during optimisation

The displacement error is also reported in physical units. Each iteration logs
the RMS and maximum nodal error magnitude $\lVert \mathbf{u} - \mathbf{u}^* \rVert$
(in mm), stored in the history (`rms_err_mm`, `max_err_mm`), the CSV, the
convergence plot, and `result.json`. The corresponding spatial fields
(`displacement_error`, `displacement_error_mag`, plus the transverse `w_error`)
are written to the XDMF history for every iteration, so the error can be
inspected in ParaView as it evolves.
