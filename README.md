# feax4d

A 4D-printing design toolkit built on [feax](../feax). It packages the two
stages of the "stable plate" workflow as a reusable API:

1. **Bilayer thermal shell topology optimisation** — design the per-layer
   density and continuous fibre orientation of a cooled CFRP / polymer
   laminate so that its thermal warping holds a target shape under load
   (e.g. a cantilever that *stays flat* under a free-edge load by cancelling
   the deflection with the bilayer's CTE-asymmetry warping).

2. **Aligned-SH fibre-path generation** — relax an aligned Swift–Hohenberg
   stripe field along the optimised per-layer director and extract
   manufacturable fibre-path polylines (SVG / NPZ / VTU / PNG).

3. **Fibrifier G-code** — convert the fibre-path SVGs into a multi-layer
   Fibrifier (9T Labs) G-code file (`fibrify`). Each design layer becomes a
   polymer-infill layer (P, filling the footprint — polymer everywhere the
   fibre isn't) followed by a fibre layer (F). The backend is a minimal
   self-contained vendoring of [libertas](https://github.com/Naruki-Ichihara/libertas)
   (GPL-3.0, same license), needing only numpy + matplotlib + shapely.

`feax` is a dependency (the FE backend); `feax4d` adds the 4D-printing
problem formulation, objectives, optimisation driver, fibre-path stage and
the G-code stage.

## Install

```bash
pip install -e .        # feax must already be importable in the environment
```

## High-level usage

```python
import feax4d

# 1) Optimise a stay-flat cantilever (flat target ⇒ load compensation).
cfg = feax4d.OptimizeConfig(
    Lx=200e-3, Ly=100e-3, Nx=80, Ny=40,
    clamp="right", load_mag=5.0, delta_t=-150.0,
    target_fn=None,                 # None ⇒ flat; or (x, y) -> w_target for shape matching
    output_dir="output_stable_plate",
)
result = feax4d.optimize(cfg)       # writes output_stable_plate/history.xdmf

# 2) Generate fibre paths from the result (writes <dir>/fibre_paths/*.svg).
feax4d.generate_fibre_paths(result.xdmf_path, stripe_period=3e-3)

# 3) Convert the fibre-path SVGs into Fibrifier G-code.
#    polymer_fill=True (default) fills every layer's footprint with polymer
#    infill (P layer) and lays the fibres on the adjacent fibre layer (F),
#    i.e. polymer everywhere the fibre isn't.
params = feax4d.FibrifierParams()
params.temperature.bed_temperature = 90
feax4d.fibre_paths_to_gcode(
    result.xdmf_path.parent / "fibre_paths", params=params,
    polymer_fill=True, infill_angle=[0.0, 90.0], infill_pitch=1.0,
)
```

See [`examples/stable_plate.py`](examples/stable_plate.py),
[`examples/fibre_paths.py`](examples/fibre_paths.py) and
[`examples/gcode.py`](examples/gcode.py).

### General mesh / BC / load

`optimize` is not tied to the rectangular cantilever — supply your own domain,
Dirichlet BCs and Neumann loads:

```python
import jax.numpy as jnp
import feax as fe
import feax4d

mesh = fe.mesh.rectangle_mesh(Nx=48, Ny=24, domain_x=0.24, domain_y=0.12, ele_type="QUAD4")
left  = lambda p: jnp.isclose(p[0], 0.0,  atol=1e-6)
right = lambda p: jnp.isclose(p[0], 0.24, atol=1e-6)

bc_specs = [fe.DirichletBCSpec(location=e, component="all", value=0.0, variable_index=v)
            for e in (left, right) for v in (0, 1)]            # doubly clamped

strip = lambda p: jnp.isclose(p[0], 0.12, atol=2.5e-3)
def strip_load(vals, x, *iv):                                  # (vals, x, *iv) -> [t_uvw, t_theta]
    return [jnp.array([0.0, 0.0, 10.0 * (x[1] - 0.06) / 0.06]), jnp.zeros(2)]

cfg = feax4d.OptimizeConfig(
    mesh=mesh, bc_specs=bc_specs,
    load_location_fns=(strip,), surface_load_fns=[strip_load],
    filter_rho_radius=0.012, filter_theta_radius=0.012,        # absolute radii for a general mesh
    target_fn=None,
)
feax4d.optimize(cfg)
```

- `mesh` — any feax mesh (omit ⇒ a rectangle from `Lx,Ly,Nx,Ny`).
- `bc_specs` (list of `fe.DirichletBCSpec`) **or** `bc_fn(problem) -> fe.DirichletBC`; omit both ⇒ the `clamp` edge is fully clamped.
- `load_location_fns` + `surface_load_fns` (one weak form per region); omit ⇒ uniform transverse `load_mag` on the free edge.
- `filter_{rho,theta}_radius` set absolute filter radii (else `frac` × domain span).

See [`examples/general_problem.py`](examples/general_problem.py).

## Low-level building blocks

The package also exposes composable pieces so you can build your own loop:

| Stage | Building blocks |
|-------|-----------------|
| Material | `Lamina`, `Polymer`, `make_layer_constitutive` |
| Problem  | `BilayerThermalShell`, `make_bilayer_shell`, `cantilever_edges`, `clamp_bc` |
| Objective | `make_shape_match_fn`, `ud_penalty_total`, `rho_contrast_penalty_total`, `mag_consistency_total` |
| Optimiser | `pack`, `unpack`, `initial_design`, `bounds`, `make_process_fn` |
| Fibre paths | `setup_aligned_sh`, `run_aligned_sh_steps`, `extract_fibre_paths`, `write_paths_svg`, `director_from_a2` |
| G-code | `svg_to_gcode`, `fibre_paths_to_gcode`, `collect_layer_svgs`, `FibrifierParams` |
| Visualisation | `plot_print_paths`, `plot_print_layers` (3D toolpath view) |
| Persistence | `save_result`, `load_design`, `load_summary`, `split_design` |

## Notes

- Two-variable FSDT/Mindlin shell (`vec=[3, 2]`): variable 0 = `(u, v, w)`,
  variable 1 = `(θx, θy)`. Linear strains; cooling applied in one shot.
- The optimisation writes per-node `density_{bot,top}` and `a2_{bot,top}`
  fields each iteration, which the fibre stage consumes from the final frame.
- feax `rectangle_mesh` orders nodes **y-fastest** (`flat = i_x·(Ny+1) + j_y`);
  the fibre stage's structured-grid reshape accounts for this.
