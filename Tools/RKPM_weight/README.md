# RKPM Weight Tool

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the maintained English code
architecture diagrams, module responsibilities and runtime data contracts. The
architecture document must be reviewed and updated in the same change whenever
the code structure, runtime flow, interfaces or MPMD protocol changes.

`main.py` separates the weight solver backend from the output transport through
two independent runtime arguments:

| Argument | Values | Purpose |
|---|---|---|
| `--solver` | `rkpm` / `ml` | Traditional RKPM solve or Transolver inference |
| `--transport` | `file` / `mpmd` | Write `.id/.lag` files or return weights directly to IAMReX |
| `--batch-size` | positive integer | Markers processed per vectorized batch (default: `4096`) |
| `--device` | `cpu` / `cuda` / `auto` | PyTorch device used by the ML backend (default: `cpu`) |
| `--volume-mode` | `prescribed` / `solve-epsilon` | Keep the existing volume rule or solve coupled marker volumes (default: `prescribed`; RKPM file mode only) |

Both solver backends use batched array operations. Traditional RKPM assembles
all polynomial bases and moment matrices in a batch and solves the independent
10-by-10 systems with batched NumPy linear algebra. The ML backend performs
coordinate preprocessing and Transolver inference in batches. In MPMD mode,
the fixed stencil metadata is packed and validated once, then reused on every
exchange; weights remain arrays throughout the solve and send path.

`batch_size` limits temporary memory without changing marker or stencil order.
The default is intended as a balanced starting point. Reduce it when the Python
rank has limited memory; increase it only after benchmarking the target CPU or
accelerator. Numerical results are independent of how markers are split into
batches, up to normal floating-point roundoff.

The default ML device remains `cpu` for portability and reproducibility. Use
`device = cuda` on a GPU allocation to require CUDA, or `device = auto` to use
CUDA when PyTorch can see a GPU and otherwise fall back to CPU. `--device` has
no effect on the traditional NumPy RKPM backend.

## Python Environment

Run the tool with a Python 3 interpreter in which the required dependencies are
installed. In the commands below, `python3` means that interpreter; activate a
virtual environment first or replace `python3` with its absolute path when
needed.

NumPy is required for all modes. Point-cloud generation and visualization also
require Matplotlib, ML inference requires PyTorch, and MPMD transport requires
`mpi4py`. The MPI implementation used by `mpi4py` must be compatible with the
one used to build IAMReX.
Matrix-solved volumes additionally require SciPy with the `cg(..., rtol=...)`
interface (SciPy 1.12 or newer); prescribed-volume runs do not import SciPy.

## Runtime Configuration File

All Python-side runtime arguments can be stored in an AMReX-style `key = value`
configuration file and loaded with one option. The bundled
`unit_test/inputs.rkpm` is a test-case example; production runs should use a
case-specific configuration file:

```bash
python3 main.py --config /path/to/case/inputs.rkpm
```

Paths in the configuration are resolved relative to the configuration file,
not the shell's working directory. Explicit command-line arguments override
the corresponding configuration values, for example:

```bash
python3 main.py --config /path/to/case/inputs.rkpm \
  --solver ml --device cuda --batch-size 2048 --check-action abort
```

The configuration controls the Python weight service. The IAMReX executable,
MPI rank counts and `IAMReX_MPMD=1` remain part of the `mpiexec` command.

## File Mode

Build new stencils and mappings from a point cloud:

```bash
python3 main.py \
  --solver rkpm --transport file \
  --inputs /path/to/case/inputs.3d \
  --geometry /path/to/case/lagrangian_points.txt \
  --body-frame \
  --output-prefix /path/to/case/rkpm_mappings
```

Point-cloud generation requires an explicit coordinate frame. Choose exactly
one of the following options:

- Use `--body-frame` when the point-cloud coordinates are relative to the body.
  The tool reads `particle_inputs.x`, `particle_inputs.y` and
  `particle_inputs.z` from the AMReX inputs file and translates the points to
  that world-coordinate center before constructing their grid stencils.
- Use `--world-frame` when the point-cloud coordinates are already expressed in
  the AMReX world coordinates bounded by `prob_lo` and `prob_hi`. No translation
  is applied.

Omitting both options is an error; the tool does not guess the coordinate
frame. In a runtime configuration file, `body_frame = true` selects body-frame
coordinates and `body_frame = false` explicitly selects world-frame
coordinates.

Read marker positions and stencils from existing `.id/.lag` files and recompute
only the ML weights:

```bash
python3 main.py \
  --solver ml --transport file \
  --inputs /path/to/case/inputs.3d \
  --id /path/to/case/rkpm_mappings.id \
  --lag /path/to/case/rkpm_mappings.lag \
  --model-dir /path/to/model_directory \
  --model-code /path/to/transolver_slim.py \
  --output-prefix /path/to/case/rkpm_mappings_ml
```

The traditional RKPM backend can use the same mapping-recompute path by changing
`--solver ml` to `--solver rkpm`.

When building a new mapping from point-cloud geometry, all 3x3x3 support cells
are constructed by broadcasting integer grid offsets. The generator therefore
scales with the number of markers and their 27-point supports instead of
repeatedly scanning the full Eulerian bounding grid for every marker.

## Matrix-Solved Marker Volumes

Use `--volume-mode solve-epsilon` to restore the legacy RKPM-3d
spread/interpolate solve, with the units converted to the current mapping
contract. Both point-cloud generation and existing-mapping recomputation are
supported. For example, recompute the bundled fixture into a separate directory:

```bash
python3 main.py --solver rkpm --transport file \
  --volume-mode solve-epsilon --epsilon-spectrum \
  --output-prefix /path/to/experiment/rkpm_mappings
```

For your own case, also set `--inputs`, `--id` and `--lag`, or use
`--geometry` with an explicit `--body-frame` / `--world-frame`. Configuration
files accept `volume_mode`, `epsilon_rtol`, `epsilon_maxiter` and
`epsilon_spectrum` (optionally prefixed with `rkpm.`). CLI flags override them.
`--epsilon-rtol` defaults to `1e-8`; `--epsilon-maxiter` defaults to `1000`.

Let J be the sparse interpolation matrix whose entries are `.lag` values
`w * Vcell`; `Vcell` normally equals 1 and is **not** the physical grid-cell
volume. On the uniform finest grid the tool solves

```text
A = J J^T
A eps = 1
V_lag[I] = eps[I] * Delta_V,    Delta_V = dx * dy * dz
```

The old RKPM-3d `implicit_A()` uses physical windows. Writing those as
`w_old = J_old / Delta_V`, its matrix is `A_old = J_old J_old^T / Delta_V`
and its unknown `epsilon` has volume units. The current `.lag` field `eps` is
dimensionless. Copying the old solution directly into `.lag` would introduce a
factor-of-cell-volume error. This restores the operator using the current J;
it retains the current polynomial correction rather than copying the legacy
moment assembly. In particular, the old `window.py` accumulates two different
terms into `[6,7]` while omitting `[6,8]`; its unused `compute_A_matrix()` also
calls the moment routine with an incompatible argument list.

The new path first computes corrected windows using a neutral unit-`eps`
seed. This scalar cancels out of the local moment correction, so the resulting
J does not depend on the old prescribed marker volume. It then applies A using
sparse J and J^T, with Jacobi-preconditioned conjugate gradients, without
constructing a dense marker-by-marker matrix or searching the whole Euler grid.
Each marker's solved `eps` is stored on every row of its stencil; `Vcell` and
the corrected interpolation weights retain their existing meanings. Geometry
diagnostics use the resulting nonuniform physical marker volumes.

Successful runs write `.id`, `.lag` and `.epsilon.json`. The JSON records
residuals, iteration count, volume statistics and each marker's `eps` and
`V_lag`. Nonconvergence and nonpositive volumes are errors; no new mapping is
written and no clipping or prescribed-volume fallback is performed. A converged
but nonpositive solution still writes `.epsilon.json` with `accepted=false`,
the failure reason and the raw signed weights for inspection. In the
bundled 2830-marker fixture the matrix solve converges at `rtol=1e-8`, but
produces 60 nonpositive weights (approximately `eps_min=-2.84`). Accordingly,
the example above reports an error for that fixture: convergence does not
guarantee a usable positive quadrature rule.

`--epsilon-spectrum` additionally computes the extremal eigenvalues of
`sqrt(D) A sqrt(D)`, where `D=diag(eps)`. This symmetric matrix has the same
eigenvalues as the actual fixed-position marker correction `A D`. For a fixed
body and the direct-forcing subiteration alone, the marker error satisfies
`e_next = (I - A D) e`; strict contraction requires all eigenvalues in `(0,2)`.
The report gives `lambda_min`, `lambda_max`, `forcing_spectral_radius` and a
conservative `forcing_contracts` check. Small systems use dense eigenvalues;
larger systems use iterative sparse extremal eigenvalue solves, which can be
expensive or fail to converge near a null mode. Such failure is reported before
writing mappings. A noncontractive spectrum is reported as a warning and is
recorded in the JSON, so it can be studied explicitly.

Solving `A eps=1` enforces `A D 1=1`, which constrains only the constant mode;
it neither guarantees positive volumes nor bounds the other eigenvalues.
Redundant markers can make A singular and the solution nonunique. A large eps
can work for separated supports, while a small residual can coexist with an
unstable mode when corrected windows have negative entries. These diagnostics
exclude pressure projection, fluid time integration and rigid-body coupling,
and do not prove stability of a complete CFD run. Fewer markers also require
separate checks of surface coverage and force/torque accuracy.

Pinelli et al. (2010), Section 2.3 after Eq. (47) and Table 1, also discuss a
separate issue: the conditioning of the volume solve itself. Their numerical
tests favor Lagrangian spacing comparable to the local Eulerian grid spacing.
Overly dense markers drive the smallest eigenvalue toward zero and produce
oscillatory volume solutions; the paper mentions small-singular-value filtering
or regularization as possible remedies. Comparable marker and Eulerian volumes
are a scaling consequence when local surface spacing and effective layer
thickness are both of order h, not a substitute for checking the actual marker
distribution. A unit-eps seed does not enforce that geometric requirement.
The paper's epsilon is a strip width, multiplied by marker arc length in 2D
(surface area in 3D); this tool's eps is the resulting volume divided by the
Eulerian cell volume. Negative solved weights do not by themselves imply an
indefinite A: the current `J J^T` is always positive semidefinite.

Volume solving is restricted to traditional RKPM **file mode**. Generate the
mapping offline before C++ initialization. MPMD transfers weights only and
cannot update per-marker `eps`; selecting volume solving with MPMD or ML is an
explicit error. The default `prescribed` mode keeps the existing behavior.

## MPMD Mode

C++ still reads `rkpm_mappings.id/.lag` during initialization. The `.lag` file
provides the fixed `i/j/k/Vcell/eps` stencil. The Python server receives marker
positions from C++, computes 27 weights per marker and returns only the weights:

```bash
mpiexec -n 2 env IAMReX_MPMD=1 \
    /path/to/amr3d.gnu.MPI.ex /path/to/case/inputs.3d \
  : -n 1 python3 \
      /path/to/IAMReX/Tools/RKPM_weight/main.py \
      --config /path/to/case/inputs.rkpm
```

The example uses two CFD ranks and one Python-server rank; adjust only the CFD
rank count for the target case because the Python server currently requires
exactly one rank. Set `solver` in the case configuration, or override it with
`--solver rkpm` or `--solver ml`, to select the backend without changing the
MPMD protocol.

Marker positions use double precision (`MPI_DOUBLE`) so cell-index validation
does not lose coordinate precision in transit. Returned weights remain single
precision (`MPI_FLOAT`) to match the ML model output. Coordinates within
floating-point roundoff of an internal grid line are snapped to that line
before applying `floor`, so the exact line consistently belongs to the cell on
its high-index side.

Before sending MPMD weights, the server can check the zeroth moment and the
first moment in cell units. `check_action` selects `off`, `warn`, or `abort`,
and `check_interval` controls how often the check runs. The check uses the
actual float32 array sent to C++. With the provided model and mapping, the
traditional RKPM backend satisfies the default `1e-6` tolerances, while the ML
backend currently exceeds the first-moment tolerance and therefore reports a
warning (or aborts when `check_action = abort`).

## Current MPMD Protocol Limitation

The protocol returns weights but not stencil `i/j/k` indices. A marker may move
inside the center cell of its original stencil, but it cannot cross a cell
boundary. The server aborts the MPI job if it detects such a crossing, preventing
C++ from applying new weights to stale cell indices. Supporting continuously
moving particles requires extending both the C++ and Python sides to transfer
updated stencil indices.

## Tests

Run the complete unit-test suite from any working directory with:

```bash
/path/to/IAMReX/Tools/RKPM_weight/unit_test/run_all_tests.sh
```

See `unit_test/README.md` for dependency handling, interpreter selection and
details of the individual tests.

The tool root intentionally contains only `src/`, `unit_test/`, `main.py`,
`README.md` and `ARCHITECTURE.md`. Bundled mapping/input fixtures, the standalone
alignment verifier and generated reference images live under `unit_test/`.
