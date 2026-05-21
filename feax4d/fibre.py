"""Aligned Swift–Hohenberg fibre-path generation.

Turns the per-layer fibre director of an optimisation result into
manufacturable fibre-path polylines by relaxing an aligned Swift–Hohenberg
(SH) stripe field to steady state, then extracting the zero-level contours::

    E[u] = ∫  ½ ((k₀² + ∇²) u)²  +  ½ γ ((d·∇) u)²  +  ¼ u⁴  −  ½ ε u²  dx

The directional "pencil" term ``γ ((d·∇)u)²`` penalises variation of ``u``
along the director ``d``, so stripes run *parallel* to the fibre.  The 4th-
order operator is split into two coupled 2nd-order equations (``v = (1+∇²)u``)
so CG1 elements suffice; time-stepped with semi-implicit backward Euler.

High-level entry point: :func:`generate_fibre_paths` (reads an optimisation
XDMF history and writes VTU / PNG / SVG / NPZ per layer).  Building blocks
(:func:`setup_aligned_sh`, :func:`run_aligned_sh_steps`,
:func:`extract_fibre_paths`, :func:`write_paths_svg`) are exposed for custom
pipelines.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe


# ── Aligned-SH problem (one backward-Euler step) ────────────────────────────

class AlignedSHStep(fe.Problem):
    """One backward-Euler step of aligned SH (mixed formulation).

    Dimensionless coordinates (mesh scaled by ``k₀`` ⇒ ``k₀ = 1``).
    Variables: ``var 0 = u`` (SH field), ``var 1 = v`` (auxiliary
    ``v = (1+∇²)u``).  Internal vars per node: director ``(d_x, d_y)``,
    previous step ``u_old`` (masked), and ``rho_mask`` ∈ {0,1}.
    """

    def custom_init(self, dt, epsilon, gamma):
        self.DT = dt
        self.EPS = epsilon
        self.GAMMA = gamma

    def get_weak_form(self):
        DT, EPS, GAMMA = self.DT, self.EPS, self.GAMMA

        def weak_form(vals, grads, x, d_x, d_y, u_old, rho_mask):
            u = vals[0][0]
            v = vals[1][0]
            grad_u = grads[0][0]
            grad_v = grads[1][0]

            d = jnp.array([d_x, d_y])
            d_grad_u = d[0] * grad_u[0] + d[1] * grad_u[1]

            eps_eff = EPS * rho_mask
            gamma_eff = GAMMA * rho_mask

            # Eq u: backward Euler with Eyre convex/concave split of the cubic.
            u_old_m = rho_mask * u_old
            u_old_sq = u_old_m * u_old_m
            mass_u = (
                u - u_old_m + DT * v
                - DT * eps_eff * u
                + 3.0 * DT * u_old_sq * u
                - 2.0 * DT * u_old_sq * u_old_m
            )
            grad_u_term = -DT * grad_v + DT * gamma_eff * d_grad_u * d

            # Eq v: v − u − ∇²u = 0.
            mass_v = v - u
            grad_v_term = grad_u

            return (
                [jnp.array([mass_u]), jnp.array([mass_v])],
                [grad_u_term[None, :], grad_v_term[None, :]],
            )

        return weak_form


# ── Time-stepping driver ───────────────────────────────────────────────────

def setup_aligned_sh(mesh_phys, stripe_period, epsilon=1.0, gamma=4.0, dt=0.5):
    """Build the aligned-SH problem + solver once (reuse across layers).

    Returns ``(problem, solver, init_guess, mesh_dimless, n_nodes)``.
    """
    k0 = 2.0 * onp.pi / stripe_period
    pts_scaled = onp.asarray(mesh_phys.points) * k0
    mesh = fe.Mesh(pts_scaled, onp.asarray(mesh_phys.cells), ele_type=mesh_phys.ele_type)
    n_nodes = mesh.points.shape[0]

    problem = AlignedSHStep(
        mesh=[mesh, mesh], vec=[1, 1], dim=2,
        ele_type=[mesh_phys.ele_type, mesh_phys.ele_type],
        additional_info=(dt, epsilon, gamma),
    )
    bc = fe.DirichletBCConfig([]).create_bc(problem)

    sample_iv = fe.InternalVars(
        volume_vars=(jnp.zeros(n_nodes), jnp.zeros(n_nodes),
                     jnp.zeros(n_nodes), jnp.ones(n_nodes)),
        surface_vars=(),
    )
    solver_opts = fe.DirectSolverOptions()
    solver = fe.create_solver(
        problem, bc=bc, solver_options=solver_opts,
        adjoint_solver_options=solver_opts, iter_num=1, internal_vars=sample_iv,
    )
    init_guess = fe.zero_like_initial_guess(problem, bc)
    return problem, solver, init_guess, mesh, n_nodes


def run_aligned_sh_steps(problem, solver, init_guess, n_nodes, director, rho_mask,
                         n_steps=600, seed=42, verbose=True):
    """Relax aligned SH from a random IC using a pre-built solver."""
    rng = onp.random.default_rng(seed)
    u_old = 0.05 * rng.standard_normal(n_nodes) * onp.asarray(rho_mask)

    d_x = jnp.asarray(director[:, 0])
    d_y = jnp.asarray(director[:, 1])
    rho_mask_j = jnp.asarray(rho_mask)

    @jax.jit
    def step_fn(sol, u_old_j):
        iv = fe.InternalVars(volume_vars=(d_x, d_y, u_old_j, rho_mask_j), surface_vars=())
        return solver(iv, sol)

    @jax.jit
    def extract_u(sol):
        return problem.unflatten_fn_sol_list(sol)[0][:, 0]

    sol = init_guess
    u_old_j = jnp.asarray(u_old)
    for step in range(n_steps):
        sol = step_fn(sol, u_old_j)
        u_new = extract_u(sol)
        u_old_j = u_new * rho_mask_j   # re-mask so void noise doesn't accumulate
        if verbose and (step % 50 == 0 or step == n_steps - 1):
            u_np = onp.asarray(u_old_j)
            print(f"  Step {step:4d}: u range [{u_np.min():+.3f}, {u_np.max():+.3f}],  "
                  f"‖u‖₂ = {onp.linalg.norm(u_np):.2f}")
    return onp.asarray(u_old_j)


def run_aligned_sh(mesh_phys, director, rho_mask, stripe_period,
                   epsilon=1.0, gamma=4.0, dt=0.5, n_steps=600, seed=42, verbose=True):
    """Convenience: setup + run in one call (single-layer use)."""
    problem, solver, init_guess, _, n_nodes = setup_aligned_sh(
        mesh_phys, stripe_period, epsilon, gamma, dt,
    )
    return run_aligned_sh_steps(
        problem, solver, init_guess, n_nodes, director, rho_mask,
        n_steps=n_steps, seed=seed, verbose=verbose,
    )


# ── Structured-grid helpers ─────────────────────────────────────────────────

def director_from_a2(a2_vec):
    """Sign-coherent unit director from ``a₂`` columns ``(a₁₁, a₂₂, a₁₂)``."""
    Tx = a2_vec[:, 0] - a2_vec[:, 1]
    Ty = 2.0 * a2_vec[:, 2]
    theta = 0.5 * onp.arctan2(Ty, Tx)
    return onp.column_stack([onp.cos(theta), onp.sin(theta)])


def _refine_rectangle(pts_orig, L_X, L_Y, refine_factor):
    """Refined QUAD4 rectangle over the same domain; returns (mesh, (Nx, Ny))."""
    x_min, x_max = float(pts_orig[:, 0].min()), float(pts_orig[:, 0].max())
    y_min, y_max = float(pts_orig[:, 1].min()), float(pts_orig[:, 1].max())
    n_total = pts_orig.shape[0]
    Nx_p1 = int(round(onp.sqrt(n_total * L_X / L_Y)))
    Ny_p1 = n_total // Nx_p1
    if Nx_p1 * Ny_p1 != n_total:
        Nx_p1 = len(onp.unique(pts_orig[:, 0]))
        Ny_p1 = len(onp.unique(pts_orig[:, 1]))
    Nx_orig, Ny_orig = Nx_p1 - 1, Ny_p1 - 1

    refined = fe.mesh.rectangle_mesh(
        Nx=Nx_orig * refine_factor, Ny=Ny_orig * refine_factor,
        domain_x=L_X, domain_y=L_Y, ele_type="QUAD4",
    )
    pts_shifted = onp.asarray(refined.points) + onp.array([x_min, y_min])
    refined = fe.Mesh(pts_shifted, onp.asarray(refined.cells), ele_type="QUAD4")
    return refined, (Nx_orig, Ny_orig)


def _bilinear_interp_struct(field, pts_new, x_min, x_max, y_min, y_max, Nx_orig, Ny_orig):
    """Bilinear interp of a feax structured-grid nodal field at arbitrary points.

    feax ``rectangle_mesh`` orders nodes **y-fastest** (flat = i_x·(Ny+1)+j_y).
    """
    dx = (x_max - x_min) / Nx_orig
    dy = (y_max - y_min) / Ny_orig
    # Reshape to (Nx+1, Ny+1) [x, y] then transpose to (Ny+1, Nx+1) [y, x].
    grid = onp.asarray(field).reshape(Nx_orig + 1, Ny_orig + 1).T

    u = (pts_new[:, 0] - x_min) / dx
    v = (pts_new[:, 1] - y_min) / dy
    i0 = onp.clip(onp.floor(u).astype(int), 0, Nx_orig - 1)
    j0 = onp.clip(onp.floor(v).astype(int), 0, Ny_orig - 1)
    fx = u - i0
    fy = v - j0
    i1 = i0 + 1
    j1 = j0 + 1
    f00 = grid[j0, i0]; f10 = grid[j0, i1]
    f01 = grid[j1, i0]; f11 = grid[j1, i1]
    return ((1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10
            + (1 - fx) * fy * f01 + fx * fy * f11)


def extract_fibre_paths(u_grid, mask_grid, x_min, x_max, y_min, y_max,
                        contour_level=0.0, min_path_length=2.0e-3):
    """Extract fibre-path polylines from the zero-level contours of ``u``.

    ``u_grid`` / ``mask_grid`` are (Ny+1, Nx+1) [y, x] grids.  Returns a list
    of ``(n_pts, 2)`` polylines in physical (x, y) [m].
    """
    from skimage import measure

    u_clip = onp.asarray(u_grid).copy()
    u_clip[mask_grid <= 0.5] = 1e10   # sentinel outside the mask → no isoline
    contours = measure.find_contours(u_clip, level=contour_level)

    Ny_p1, Nx_p1 = u_clip.shape
    Lx, Ly = x_max - x_min, y_max - y_min
    paths = []
    for c in contours:
        if c.shape[0] < 2:
            continue
        # find_contours returns (row, col) = (y_index, x_index).
        x = x_min + (c[:, 1] / (Nx_p1 - 1)) * Lx
        y = y_min + (c[:, 0] / (Ny_p1 - 1)) * Ly
        seg = onp.column_stack([x, y])
        if min_path_length > 0:
            if onp.linalg.norm(seg[1:] - seg[:-1], axis=1).sum() < min_path_length:
                continue
        paths.append(seg)
    return paths


def extract_region_contours(region_grid, x_min, x_max, y_min, y_max, min_pts=4):
    """Closed boundary polylines of a binary region on a structured [y, x] grid.

    The grid is zero-padded so regions touching the domain edge still yield
    *closed* loops; interior holes (e.g. the fibre region inside the polymer
    region) come back as their own closed contours.  Returns polylines in
    physical (x, y) [m] — suitable as ``id="contour"`` polygons that bound the
    polymer infill.
    """
    from skimage import measure

    g = onp.asarray(region_grid).astype(float)
    Ny_p1, Nx_p1 = g.shape          # [y, x]
    padded = onp.pad(g, 1, mode="constant", constant_values=0.0)
    Lx, Ly = x_max - x_min, y_max - y_min

    polys = []
    for c in measure.find_contours(padded, level=0.5):
        if c.shape[0] < min_pts:
            continue
        # Undo the 1-px pad, then map (row=y, col=x) → physical.
        col = onp.clip(c[:, 1] - 1.0, 0.0, Nx_p1 - 1)
        row = onp.clip(c[:, 0] - 1.0, 0.0, Ny_p1 - 1)
        x = x_min + (col / (Nx_p1 - 1)) * Lx
        y = y_min + (row / (Ny_p1 - 1)) * Ly
        poly = onp.column_stack([x, y])
        if not onp.allclose(poly[0], poly[-1]):
            poly = onp.vstack([poly, poly[0]])   # ensure closed
        polys.append(poly)
    return polys


def write_paths_svg(paths, svg_path, x_min, x_max, y_min, y_max,
                    stroke_width_mm=0.1, stroke_color="#000000", units="mm",
                    scale=1000.0, contours=None):
    """Save fibre polylines as an SVG (m → mm).

    ``paths`` are written as ``<g id="fibre_paths">`` (parsed as fibre/stripe
    downstream).  If ``contours`` (a list of closed polygons in physical
    coords) is given, they are written as ``<g id="contour">`` — the
    density-derived polymer region that bounds the polymer infill.
    """
    Lx_phys = (x_max - x_min) * scale
    Ly_phys = (y_max - y_min) * scale
    x0_phys, y0_phys = x_min * scale, y_min * scale

    def _poly_pts(poly):
        xs = poly[:, 0] * scale
        ys = (y_max + y_min - poly[:, 1]) * scale   # flip y so SVG matches physical up
        return " ".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys))

    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>']
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{Lx_phys:.4f}{units}" height="{Ly_phys:.4f}{units}" '
        f'viewBox="{x0_phys:.4f} {y0_phys:.4f} {Lx_phys:.4f} {Ly_phys:.4f}">'
    )
    lines.append(f'  <desc>{len(paths)} fibre-path polylines (aligned-SH zero contours).</desc>')
    lines.append(f'  <g id="fibre_paths" fill="none" stroke="{stroke_color}" '
                 f'stroke-width="{stroke_width_mm}">')
    for poly in paths:
        lines.append(f'    <polyline points="{_poly_pts(poly)}"/>')
    lines.append('  </g>')
    if contours:
        lines.append(f'  <g id="contour" fill="none" stroke="#888888" '
                     f'stroke-width="{stroke_width_mm}">')
        for poly in contours:
            lines.append(f'    <polyline points="{_poly_pts(poly)}"/>')
        lines.append('  </g>')
    lines.append('</svg>')
    with open(svg_path, "w") as f:
        f.write("\n".join(lines))


# ── High-level: read XDMF history → per-layer fibre paths ───────────────────

def generate_fibre_paths(
    xdmf_path,
    out_dir=None,
    stripe_period=3.0e-3,
    epsilon=1.0,
    gamma=4.0,
    dt=0.5,
    n_steps=600,
    rho_cutoff=0.5,
    refine_factor=4,
    min_path_length=2.0e-3,
    svg_stroke_mm=0.3,
    layers=(0, 1),
    verbose=True,
):
    """Generate fibre paths for each laminate layer from an optimisation history.

    Reads the *final* frame of ``xdmf_path`` (expects per-node fields
    ``density_{bot,top}`` and ``a2_{bot,top}``), relaxes the aligned-SH stripe
    field along the per-layer director, and writes per layer into ``out_dir``::

        sh_stripe_layer{i}.vtu / .png
        fibre_paths_layer{i}.svg / .npz

    Returns a dict ``{layer_index: list_of_polylines}``.
    """
    import meshio

    xdmf_path = Path(xdmf_path)
    if out_dir is None:
        out_dir = xdmf_path.parent / "fibre_paths"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    log(f"Reading {xdmf_path} ...")
    with meshio.xdmf.TimeSeriesReader(str(xdmf_path)) as reader:
        pts, cells = reader.read_points_cells()
        _, point_data, _ = reader.read_data(reader.num_steps - 1)
    if pts.shape[1] > 2:
        pts = pts[:, :2]

    L_X = float(pts[:, 0].max() - pts[:, 0].min())
    L_Y = float(pts[:, 1].max() - pts[:, 1].min())

    fine_mesh, (Nx_o, Ny_o) = _refine_rectangle(pts, L_X, L_Y, refine_factor)
    fine_pts = onp.asarray(fine_mesh.points)
    Nx_f, Ny_f = Nx_o * refine_factor, Ny_o * refine_factor
    log(f"  Original mesh: {pts.shape[0]} nodes ({Nx_o}×{Ny_o} elements)")
    log(f"  Refined mesh:  {fine_pts.shape[0]} nodes ({Nx_f}×{Ny_f}, ×{refine_factor})")
    log(f"  Stripe period: {stripe_period*1e3:g} mm  "
        f"(elements/period: {stripe_period/(L_X/Nx_f):.1f})")

    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())

    def interp(field):
        return _bilinear_interp_struct(field, fine_pts, x_min, x_max, y_min, y_max, Nx_o, Ny_o)

    log("\nBuilding SH solver (shared across layers)...")
    sh_problem, sh_solver, sh_init, _, sh_n_nodes = setup_aligned_sh(
        fine_mesh, stripe_period, epsilon, gamma, dt,
    )

    results = {}
    for layer in layers:
        suffix = "bot" if layer == 0 else "top"
        rho = interp(onp.asarray(point_data[f"density_{suffix}"]))
        a2_vec = onp.asarray(point_data[f"a2_{suffix}"])
        a2_xx = interp(a2_vec[:, 0]); a2_yy = interp(a2_vec[:, 1]); a2_xy = interp(a2_vec[:, 2])
        theta = 0.5 * onp.arctan2(2.0 * a2_xy, a2_xx - a2_yy)
        d = onp.column_stack([onp.cos(theta), onp.sin(theta)])

        mask = (rho > rho_cutoff).astype(onp.float64)
        log(f"\nLayer {layer} ({suffix}): {int(mask.sum())} solid / {rho.size} total")

        u = run_aligned_sh_steps(
            sh_problem, sh_solver, sh_init, sh_n_nodes, d, mask,
            n_steps=n_steps, seed=42 + layer, verbose=verbose,
        )
        S = mask * (0.5 + 0.5 * onp.tanh(2.0 * u))

        path = out_dir / f"sh_stripe_layer{layer}.vtu"
        fe.utils.save_sol(fine_mesh, str(path), point_infos=[
            ("density", rho),
            ("director", onp.column_stack([d, onp.zeros(d.shape[0])])),
            ("u_raw", u), ("stripe", S), ("void_mask", mask.astype(onp.float32)),
        ])
        log(f"  Saved {path}")

        # feax y-fastest order ⇒ reshape (Nx+1, Ny+1) then transpose to [y, x].
        u_grid = u.reshape(Nx_f + 1, Ny_f + 1).T
        S_grid = S.reshape(Nx_f + 1, Ny_f + 1).T
        mask_grid = mask.reshape(Nx_f + 1, Ny_f + 1).T

        _save_stripe_png(out_dir / f"sh_stripe_layer{layer}.png", S_grid,
                         layer, suffix, x_min, x_max, y_min, y_max, L_X, L_Y, log)

        paths = extract_fibre_paths(u_grid, mask_grid, x_min, x_max, y_min, y_max,
                                    contour_level=0.0, min_path_length=min_path_length)
        total_len = sum(onp.linalg.norm(p[1:] - p[:-1], axis=1).sum() for p in paths)
        log(f"  Extracted {len(paths)} fibre paths "
            f"({sum(p.shape[0] for p in paths)} pts, total length {total_len*1e3:.0f} mm)")

        # Polymer region = the non-fibre (low-density) complement of this layer,
        # written as closed ``contour`` polygons so the g-code stage fills
        # polymer there only (and excludes the fibre region as holes).
        polymer_grid = (mask_grid <= 0.5).astype(onp.float64)
        polymer_contours = extract_region_contours(
            polymer_grid, x_min, x_max, y_min, y_max)
        log(f"  Polymer region: {len(polymer_contours)} boundary contour(s) "
            f"(fibre area excluded)")

        write_paths_svg(paths, str(out_dir / f"fibre_paths_layer{layer}.svg"),
                        x_min, x_max, y_min, y_max, stroke_width_mm=svg_stroke_mm,
                        contours=polymer_contours)
        onp.savez(out_dir / f"fibre_paths_layer{layer}.npz",
                  **{f"path_{i:04d}": p for i, p in enumerate(paths)},
                  count=len(paths), bounds=onp.array([x_min, x_max, y_min, y_max]))
        log(f"  Saved fibre_paths_layer{layer}.svg / .npz")
        results[layer] = paths

    log(f"\nAll outputs in {out_dir}")
    return results


def _save_stripe_png(png_path, S_grid, layer, suffix, x_min, x_max, y_min, y_max, L_X, L_Y, log):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, max(2.0, 8 * L_Y / L_X)))
    ax.imshow(S_grid, origin="lower", cmap="gray", vmin=0.0, vmax=1.0,
              extent=(x_min, x_max, y_min, y_max), aspect="equal")
    ax.set_title(f"Layer {layer} ({suffix}) — Aligned SH stripes")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(png_path, dpi=200); plt.close(fig)
    log(f"  Saved {png_path}")
