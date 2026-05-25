"""High-level driver for the bilayer thermal shell topology optimisation.

:class:`OptimizeConfig` collects every knob; :func:`optimize` runs the full
NLopt/MMA loop and writes a ParaView XDMF history (with the ``density_{bot,
top}`` / ``a2_{bot,top}`` per-node fields that :mod:`feax4d.fibre` consumes),
returning an :class:`OptimizeResult`.

Lower-level building blocks (:func:`pack`, :func:`unpack`,
:func:`make_process_fn`, :func:`initial_design`) are exposed for users who
want to assemble their own loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as np
import numpy as onp
import nlopt

import feax as fe
import feax.gene as gene
from feax.mechanics.orientation import orientation_tensor_2d, principal_direction

from feax4d.materials import Lamina, Polymer
from feax4d.shell import N_FIELDS, make_bilayer_shell, cantilever_edges, clamp_bc
from feax4d.objectives import (
    make_shape_match_fn,
    ud_penalty_total,
    rho_contrast_penalty_total,
    mag_consistency_total,
)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class OptimizeConfig:
    """All parameters of one bilayer-shell optimisation run.

    The problem domain, boundary conditions and loads can be supplied
    explicitly (general case) or left to the built-in rectangular-cantilever
    convenience:

    * **General**: pass ``mesh`` (any feax mesh), ``bc_specs`` or ``bc_fn``
      (Dirichlet BCs), and ``load_location_fns`` + ``surface_load_fns``
      (Neumann loads).
    * **Convenience**: leave those ``None`` and a rectangle ``Lx×Ly`` (``Nx×Ny``
      QUAD4) clamped on the ``clamp`` edge with a uniform transverse load
      ``load_mag`` on the opposite edge is built automatically.
    """

    # ── Domain ──
    # Explicit mesh (any feax mesh).  If None, a rectangle is built below.
    mesh: Optional[object] = None
    # Rectangle fallback (used only when ``mesh is None``).
    Lx: float = 200.0e-3
    Ly: float = 100.0e-3
    Nx: int = 80
    Ny: int = 40

    # ── Boundary conditions ──
    # Provide ONE of: a list of fe.DirichletBCSpec, or a builder bc_fn(problem)
    # -> fe.DirichletBC.  If both None, the ``clamp`` edge is fully clamped.
    bc_specs: Optional[Sequence] = None
    bc_fn: Optional[Callable] = None
    clamp: str = "right"             # cantilever convenience clamp edge

    # ── Loads ──
    # General Neumann loads: where they apply + the surface weak form per
    # region (signature (vals, x, *iv) -> [t_uvw(3,), t_theta(2,)]).
    load_location_fns: Optional[Sequence[Callable]] = None
    surface_load_fns: Optional[Sequence[Callable]] = None
    # Cantilever convenience: uniform transverse load on the free edge.
    load_mag: float = 5.0            # N/m

    # Materials / thermal.
    lamina: Lamina = field(default_factory=Lamina)
    polymer: Polymer = field(default_factory=Polymer)
    delta_t: float = -150.0          # K, one-shot cooling

    # Objective: target transverse field w_target(x, y).  None ⇒ flat (w≡0),
    # i.e. the stay-flat / load-compensation objective.
    target_fn: Optional[Callable[[onp.ndarray, onp.ndarray], onp.ndarray]] = None

    # Design parameterisation.
    rho_init: float = 0.5
    simp_penalty: float = 3.0
    sgn_beta: float = 10.0
    # Helmholtz filter radii.  Absolute (m) if given, else ``frac`` × domain size.
    filter_rho_radius: Optional[float] = None
    filter_theta_radius: Optional[float] = None
    filter_rho_frac: float = 0.05
    filter_theta_frac: float = 0.05
    x3_lb: float = -1.0
    x3_ub: float = 1.0
    ori_tol: float = 1e-2

    # Penalty weights.
    ud_penalty: float = 1.0
    rho_contrast_penalty: float = 0.5
    mag_consistency_penalty: float = 1.0

    # Convergence (stop when ANY tolerance is met; MAX_ITER is a safety cap).
    ftol_rel: float = 1e-5
    xtol_rel: float = 1e-5
    xtol_abs: float = 1e-5
    max_iter: Optional[int] = 200

    # Output.
    output_dir: Path = Path("output_feax4d")
    save_vtu: bool = False
    save_result: bool = True     # write design.npz + result.json on completion
    snapshot_every: int = 1
    verbose: bool = True


@dataclass
class OptimizeResult:
    """Outcome of :func:`optimize`."""

    x_opt: onp.ndarray
    history: dict
    denom: float
    best_obj: float
    best_iter: int
    n_iters: int
    stop_reason: str
    n_nodes: int
    mesh: object
    problem: object
    xdmf_path: Path


# ── Pack / unpack ────────────────────────────────────────────────────────────

def unpack(x_flat, n_nodes: int):
    """Split a flat design vector into 8 node fields (rho/x1/x2/x3 × 2 layers)."""
    return tuple(
        x_flat[k * n_nodes:(k + 1) * n_nodes] for k in range(N_FIELDS)
    )


def pack(fields, n_nodes: int):
    """Concatenate 8 node fields into one flat design vector."""
    return np.concatenate([np.asarray(f).reshape(n_nodes) for f in fields])


def initial_design(cfg: OptimizeConfig, n_nodes: int) -> onp.ndarray:
    """Uniform-density, unaligned-orientation start (libertas convention)."""
    init_layer = (cfg.rho_init, -1.0 + cfg.ori_tol, -1.0 + cfg.ori_tol, 0.0)
    x0 = onp.empty(N_FIELDS * n_nodes)
    for k in range(N_FIELDS):
        s = slice(k * n_nodes, (k + 1) * n_nodes)
        _, fld = divmod(k, 4)
        x0[s] = init_layer[fld]
    return x0


def bounds(cfg: OptimizeConfig, n_nodes: int):
    """(lower, upper) box bounds: ρ∈[0,1], x1,x2∈[-1,1], x3∈[x3_lb,x3_ub]."""
    n_total = N_FIELDS * n_nodes
    lower = onp.empty(n_total)
    upper = onp.empty(n_total)
    for k in range(N_FIELDS):
        s = slice(k * n_nodes, (k + 1) * n_nodes)
        if k % 4 == 0:        # rho
            lower[s], upper[s] = 0.0, 1.0
        elif k % 4 == 3:      # x3
            lower[s], upper[s] = cfg.x3_lb, cfg.x3_ub
        else:                 # x1, x2
            lower[s], upper[s] = -1.0, 1.0
    return lower, upper


def make_process_fn(cfg, filter_rho, filter_theta, n_nodes):
    """Return ``process(x_flat) -> (design8, rho0_f, rho1_f)``.

    ``design8`` is the InternalVars tuple fed to the shell (SIMP-projected
    density + filtered orientation params); the raw filtered densities are
    returned for the grey-scale penalties.
    """
    p = cfg.simp_penalty

    def process(x_flat):
        rho0, x1_0, x2_0, x3_0, rho1, x1_1, x2_1, x3_1 = unpack(x_flat, n_nodes)
        rho0_f = filter_rho(rho0)
        rho1_f = filter_rho(rho1)
        design8 = (
            rho0_f ** p,
            filter_theta(x1_0), filter_theta(x2_0), filter_theta(x3_0),
            rho1_f ** p,
            filter_theta(x1_1), filter_theta(x2_1), filter_theta(x3_1),
        )
        return design8, rho0_f, rho1_f

    return process


# ── Driver ───────────────────────────────────────────────────────────────────

def optimize(cfg: OptimizeConfig) -> OptimizeResult:
    """Run the bilayer thermal shell optimisation defined by ``cfg``."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if cfg.verbose else (lambda *a, **k: None)

    # ── Domain: caller-supplied mesh, else a rectangle. ──
    if cfg.mesh is not None:
        mesh = cfg.mesh
    else:
        mesh = fe.mesh.rectangle_mesh(
            Nx=cfg.Nx, Ny=cfg.Ny, domain_x=cfg.Lx, domain_y=cfg.Ly, ele_type="QUAD4",
        )
    n_nodes = mesh.points.shape[0]
    pts = onp.asarray(mesh.points)
    log(f"Mesh: {n_nodes} nodes, {mesh.cells.shape[0]} cells")

    # Characteristic size for the default (fractional) filter radius / tol.
    span = float(max(pts[:, 0].max() - pts[:, 0].min(),
                     pts[:, 1].max() - pts[:, 1].min()))
    tol = 1e-5 * span

    # ── Loads: explicit (location_fns + surface_load_fns) or cantilever. ──
    if cfg.load_location_fns is not None:
        load_loc_fns = tuple(cfg.load_location_fns)
        load_fns = cfg.surface_load_fns      # None ⇒ uniform load_mag per region
    else:
        # Cantilever convenience: uniform transverse load on the free edge.
        _, free_edge = cantilever_edges(cfg.clamp, cfg.Lx, cfg.Ly, tol)
        load_loc_fns = (free_edge,)
        load_fns = None

    problem = make_bilayer_shell(
        mesh, cfg.lamina, cfg.polymer, cfg.delta_t,
        sgn_beta=cfg.sgn_beta, location_fns=load_loc_fns,
        surface_load_fns=load_fns, load_mag=cfg.load_mag,
    )

    # ── Boundary conditions: explicit specs / builder, else clamp one edge. ──
    if cfg.bc_specs is not None:
        bc = fe.DirichletBCConfig(list(cfg.bc_specs)).create_bc(problem)
    elif cfg.bc_fn is not None:
        bc = cfg.bc_fn(problem)
    else:
        clamped_edge, _ = cantilever_edges(cfg.clamp, cfg.Lx, cfg.Ly, tol)
        bc = clamp_bc(problem, clamped_edge)

    r_rho = cfg.filter_rho_radius if cfg.filter_rho_radius is not None else cfg.filter_rho_frac * span
    r_theta = cfg.filter_theta_radius if cfg.filter_theta_radius is not None else cfg.filter_theta_frac * span
    filter_rho = gene.create_helmholtz_filter(mesh, radius=r_rho)
    filter_theta = gene.create_helmholtz_filter(mesh, radius=r_theta)
    volume_fn = gene.create_volume_fn(problem)

    # Target transverse field at the nodes (flat if no target_fn).
    pts = onp.asarray(mesh.points)
    if cfg.target_fn is None:
        w_target_arr = np.zeros(n_nodes)
    else:
        w_target_arr = np.asarray(cfg.target_fn(pts[:, 0], pts[:, 1]))
    # Target *displacement* (n_nodes, 3): in-plane zero, transverse = w*.
    u_target_arr = np.stack([np.zeros(n_nodes), np.zeros(n_nodes), w_target_arr], axis=1)

    # Pre-warm the linear solver with a shape-correct sample InternalVars.
    sample_iv = fe.InternalVars(
        volume_vars=tuple(
            fe.InternalVars.create_node_var(problem, v)
            for v in (cfg.rho_init, 1.0 - 1e-2, 1.0 - 1e-2, +1.0,
                      cfg.rho_init, 1.0 - 1e-2, 1.0 - 1e-2, -1.0)
        ),
        surface_vars=(),
    )
    solver_opts = fe.DirectSolverOptions(solver="auto", verbose=cfg.verbose)
    solver = fe.create_solver(
        problem, bc=bc, solver_options=solver_opts,
        adjoint_solver_options=solver_opts, iter_num=1, internal_vars=sample_iv,
    )
    init_guess = fe.zero_like_initial_guess(problem, bc)

    process = make_process_fn(cfg, filter_rho, filter_theta, n_nodes)

    lower, upper = bounds(cfg, n_nodes)
    n_total = N_FIELDS * n_nodes
    x0 = initial_design(cfg, n_nodes)

    # Normalisation: ‖u − u_target‖² (full displacement) of the initial design.
    design8_0, _, _ = process(np.array(x0))
    sol0 = solver(fe.InternalVars(volume_vars=design8_0, surface_vars=()), init_guess)
    uvw0 = problem.unflatten_fn_sol_list(sol0)[0]
    denom = float(np.maximum(np.sum((uvw0 - u_target_arr) ** 2), 1e-30))
    shape_match_fn = make_shape_match_fn(problem, u_target_arr, denom)
    log(f"Init ‖u−u*‖_rms : "
        f"{float(np.sqrt(np.mean(np.sum((uvw0 - u_target_arr) ** 2, axis=1)))) * 1e3:.4f} mm")

    sb = cfg.sgn_beta

    def objective_fn(x_flat):
        design8, rho0_f, rho1_f = process(x_flat)
        iv = fe.InternalVars(volume_vars=design8, surface_vars=())
        sol = solver(iv, init_guess)
        ud_val = ud_penalty_total(design8, rho0_f, rho1_f, sb)
        rho_val = rho_contrast_penalty_total(rho0_f, rho1_f)
        mag_val = mag_consistency_total(design8, rho0_f, rho1_f, sb)
        loss = (
            shape_match_fn(sol)
            + cfg.ud_penalty * ud_val
            + cfg.rho_contrast_penalty * rho_val
            + cfg.mag_consistency_penalty * mag_val
        )
        return loss, (sol, ud_val, rho_val, mag_val)

    def mean_density(x_flat):
        rho0, _, _, _, rho1, _, _, _ = unpack(x_flat, n_nodes)
        return 0.5 * (volume_fn(filter_rho(rho0)) + volume_fn(filter_rho(rho1)))

    obj_and_grad = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))
    mean_density_jit = jax.jit(mean_density)

    a2_vmap = jax.jit(jax.vmap(
        lambda a, b, c: orientation_tensor_2d(a, b, c, sgn_beta=sb)[0]
    ))

    def snapshot_fields(x_arr, sol=None):
        x_jax = np.array(x_arr)
        r0, x1_0, x2_0, x3_0, r1, x1_1, x2_1, x3_1 = unpack(x_jax, n_nodes)
        rho0_f = onp.array(filter_rho(r0)); rho1_f = onp.array(filter_rho(r1))
        a2_0 = onp.array(a2_vmap(filter_theta(x1_0), filter_theta(x2_0), filter_theta(x3_0)))
        a2_1 = onp.array(a2_vmap(filter_theta(x1_1), filter_theta(x2_1), filter_theta(x3_1)))
        d2_0 = onp.asarray(principal_direction(a2_0))
        d2_1 = onp.asarray(principal_direction(a2_1))
        d3_0 = onp.column_stack([d2_0, onp.zeros(d2_0.shape[0])])
        d3_1 = onp.column_stack([d2_1, onp.zeros(d2_1.shape[0])])
        if sol is None:
            design8, _, _ = process(x_jax)
            sol = solver(fe.InternalVars(volume_vars=design8, surface_vars=()), init_guess)
        sol_list = problem.unflatten_fn_sol_list(sol)
        uvw = onp.asarray(sol_list[0])
        rotation = onp.asarray(sol_list[1])
        w = uvw[:, 2]
        w_target = onp.asarray(w_target_arr)
        w_error = w - w_target
        a2_bot_vec = onp.column_stack([a2_0[:, 0, 0], a2_0[:, 1, 1], a2_0[:, 0, 1]])
        a2_top_vec = onp.column_stack([a2_1[:, 0, 0], a2_1[:, 1, 1], a2_1[:, 0, 1]])
        zeros = onp.zeros_like(w)
        # Target displacement (0,0,w*) and full displacement-error vector.
        disp_target = onp.column_stack([zeros, zeros, w_target])
        disp_error = uvw - disp_target                       # (u, v, w − w*)
        disp_error_mag = onp.linalg.norm(disp_error, axis=1)  # ‖u − u*‖ (objective)
        return [
            ("density_bot", rho0_f), ("density_top", rho1_f),
            ("director_bot", d3_0), ("director_top", d3_1),
            ("a2_bot", a2_bot_vec), ("a2_top", a2_top_vec),
            ("displacement", uvw), ("rotation", rotation),
            ("w", w),                          # achieved transverse displacement
            ("w_target", w_target),            # target transverse field
            ("w_error", w_error),              # transverse error  w − w*
            ("w_error_abs", onp.abs(w_error)),  # |transverse error|
            ("displacement_target", disp_target),    # (0,0,w*) for warp-by-vector
            ("displacement_error", disp_error),      # (u, v, w−w*) full disp error
            ("displacement_error_mag", disp_error_mag),  # ‖u − u*‖ (objective field)
        ]

    history = {"iter": [], "obj": [], "vol": [], "ud": [], "rho_pen": [], "mag": [],
               "rms_err_mm": [], "max_err_mm": []}

    @jax.jit
    def err_metrics(sol_flat):
        """(rms, max) nodal displacement-error magnitude ‖u − u*‖ in metres."""
        uvw = problem.unflatten_fn_sol_list(sol_flat)[0]
        mag = np.sqrt(np.sum((uvw - u_target_arr) ** 2, axis=1))
        return np.sqrt(np.mean(mag * mag)), np.max(mag)
    iter_count = [0]
    best = {"obj": float("inf"), "iter": 0}

    vtu_dir = out / "vtu"
    if cfg.save_vtu:
        vtu_dir.mkdir(parents=True, exist_ok=True)

    def save_vtu(step, fields):
        if cfg.save_vtu:
            fe.utils.save_sol(mesh, str(vtu_dir / f"iter_{step:04d}.vtu"), point_infos=fields)

    bc_src = ("bc_specs" if cfg.bc_specs is not None
              else "bc_fn" if cfg.bc_fn is not None else f"clamp {cfg.clamp!r}")
    log(f"Design vars  : {n_total}  (8 fields × {n_nodes} nodes)")
    log(f"BC / load    : {bc_src};  {len(load_loc_fns)} surface load region(s);  "
        f"ΔT={cfg.delta_t} K")
    log(f"Objective    : {'flat (stay-flat)' if cfg.target_fn is None else 'shape match'}"
        f" (displacement error)")
    log(f"Convergence  : ftol_rel={cfg.ftol_rel:g}  xtol_rel={cfg.xtol_rel:g}  "
        f"xtol_abs={cfg.xtol_abs:g}  (max-iter cap = {cfg.max_iter})")
    log("-" * 60)

    xdmf_path = out / "history.xdmf"
    stop_reason = ["", ]

    with fe.XDMFWriter(mesh, xdmf_path) as xw:
        init_fields = snapshot_fields(x0)
        xw.write_iteration(0, point_infos=init_fields)
        save_vtu(0, init_fields)

        def _objective(xx, grad):
            x_jax = np.array(xx)
            (val, (sol, ud_val, rho_val, mag_val)), g = obj_and_grad(x_jax)
            if grad.size > 0:
                grad[:] = onp.array(g)
            v = float(mean_density_jit(x_jax))
            val_f = float(val)
            rms_e, max_e = err_metrics(sol)
            rms_mm = float(rms_e) * 1e3
            max_mm = float(max_e) * 1e3
            iter_count[0] += 1
            if val_f < best["obj"]:
                best["obj"], best["iter"] = val_f, iter_count[0]
            history["iter"].append(iter_count[0])
            history["obj"].append(val_f)
            history["vol"].append(v)
            history["ud"].append(float(ud_val))
            history["rho_pen"].append(float(rho_val))
            history["mag"].append(float(mag_val))
            history["rms_err_mm"].append(rms_mm)
            history["max_err_mm"].append(max_mm)
            log(f"Iter {iter_count[0]:4d}: obj={val_f:.4e}  ρ̄={v:.4f}  "
                f"P_UD={float(ud_val):.4f}  P_ρ={float(rho_val):.4f}  "
                f"P_mag={float(mag_val):.4f}  rms_err={rms_mm:.3f}mm  "
                f"best={best['obj']:.4e}")
            if iter_count[0] % cfg.snapshot_every == 0:
                fields = snapshot_fields(xx, sol=sol)
                xw.write_iteration(iter_count[0], point_infos=fields)
                save_vtu(iter_count[0], fields)
            return val_f

        opt = nlopt.opt(nlopt.LD_MMA, n_total)
        opt.set_lower_bounds(lower); opt.set_upper_bounds(upper)
        opt.set_min_objective(_objective)
        opt.set_ftol_rel(cfg.ftol_rel)
        opt.set_xtol_rel(cfg.xtol_rel)
        opt.set_xtol_abs(cfg.xtol_abs)
        if cfg.max_iter is not None:
            opt.set_maxeval(cfg.max_iter)

        x_opt = opt.optimize(x0)
        stop_reason[0] = {
            nlopt.SUCCESS: "SUCCESS",
            nlopt.STOPVAL_REACHED: "STOPVAL_REACHED",
            nlopt.FTOL_REACHED: "FTOL_REACHED (objective settled)",
            nlopt.XTOL_REACHED: "XTOL_REACHED (design settled)",
            nlopt.MAXEVAL_REACHED: "MAXEVAL_REACHED (cap hit — not converged)",
            nlopt.MAXTIME_REACHED: "MAXTIME_REACHED",
        }.get(opt.last_optimize_result(), str(opt.last_optimize_result()))
        log(f"\nOptimization done ({stop_reason[0]}; {iter_count[0]} iters). "
            f"final obj = {opt.last_optimum_value():.6e}  "
            f"best obj = {best['obj']:.6e} @ iter {best['iter']}")
        final_fields = snapshot_fields(x_opt)
        xw.write_iteration(iter_count[0] + 1, point_infos=final_fields)
        save_vtu(iter_count[0] + 1, final_fields)

    _write_history_csv(out / "history.csv", history)
    _save_history_plot(out / "history.png", history)

    result = OptimizeResult(
        x_opt=onp.asarray(x_opt), history=history, denom=denom,
        best_obj=best["obj"], best_iter=best["iter"], n_iters=iter_count[0],
        stop_reason=stop_reason[0], n_nodes=n_nodes, mesh=mesh, problem=problem,
        xdmf_path=xdmf_path,
    )

    if cfg.save_result:
        from feax4d.utils import save_result
        save_result(result, cfg, out_dir=out)
        log(f"Saved design.npz + result.json to {out}")

    return result


def _write_history_csv(path, history):
    import csv
    cols = ("iter", "obj", "vol", "ud", "rho_pen", "mag", "rms_err_mm", "max_err_mm")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(history["iter"])):
            w.writerow([history[k][i] for k in cols])


def _save_history_plot(path, history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(history["iter"], history["obj"], "b-")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("‖u − u*‖² / ‖·‖²_init")
    axes[0].set_title("Objective (displacement error)"); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(history["iter"], history.get("rms_err_mm", []), "r-", label="rms")
    if history.get("max_err_mm"):
        axes[1].semilogy(history["iter"], history["max_err_mm"], "r--", alpha=0.5, label="max")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("error [mm]")
    axes[1].set_title("Displacement error ‖u − u*‖"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["iter"], history["vol"], "g-")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Mean fibre fraction")
    axes[2].set_title("Mean ρ (diagnostic)"); axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
