<!--
SPDX-FileCopyrightText: 2026 IAMReX contributors
SPDX-License-Identifier: BSD-3-Clause
-->

# RKPM Weight Tests

Run the complete suite from any working directory:

```bash
/path/to/IAMReX/Tools/RKPM_weight/unit_test/run_all_tests.sh
```

The script uses `python3` by default. To use a particular virtual environment
without relying on its activation state, provide the path to its interpreter:

```bash
RKPM_TEST_PYTHON=/path/to/python \
  /path/to/IAMReX/Tools/RKPM_weight/unit_test/run_all_tests.sh
```

NumPy is required for the core suite. The ML test is skipped if PyTorch or the
model files are unavailable. The MPMD test is skipped if either `mpiexec` or
`mpi4py` is unavailable. `mpi4py` and `mpiexec` must use compatible MPI
implementations for that test.

Most tests use a three-marker synthetic fixture with a nonzero domain origin,
anisotropic cell spacing and complete 3x3x3 stencils. The full alignment
regression additionally uses the bundled 2830-marker files under
`fixtures/real_case/`.

## Test Details

Seven `test_*.py` files currently define nine `unittest` test methods;
`test_validation.py` and `test_fixed_stencil.py` contain two methods each, and
each other file contains one.

### `test_rkpm_reproduction.py`

#### `RKPMReproductionTests.test_rkpm_reproduces_quadratic_basis`

- **What it tests:** The traditional RKPM solver reproduces the constant,
  linear and quadratic polynomial bases on a nonzero-origin, anisotropic grid.
- **How it tests it:** It solves three synthetic 3x3x3 marker stencils, writes
  and reloads temporary `.id/.lag` files, and evaluates them with
  `compute_reproduction_residuals()`. Zeroth-, first- and second-order
  residuals must all be within `1e-10`.

### `test_real_mapping_alignment.py`

#### `RealMappingAlignmentTests.test_real_mapping_reproduces_quadratic_basis`

- **What it tests:** The complete bundled 2830-marker mapping remains aligned
  with its AMReX grid and satisfies constant, linear and quadratic reproduction.
- **How it tests it:** It reads the real inputs, `.id` and `.lag` files,
  reconstructs cell centers using the actual domain origin and separate
  directional grid spacings, and computes the reproduction residuals. Their
  maximum errors must be below `1e-9`, `1e-6` cell and `1e-6` cell squared,
  respectively.

### `test_validation.py`

#### `MomentValidationTests.test_moment_check_accepts_valid_weights`

- **What it tests:** A valid normalized weight set passes both the zeroth- and
  first-moment checks.
- **How it tests it:** It places unit weight at a cell center coincident with
  the marker, computes the two errors, and requires both to pass `1e-12`
  tolerances.

#### `MomentValidationTests.test_moment_check_distinguishes_sum_and_first_moment_errors`

- **What it tests:** The validator distinguishes a normalization error from a
  pure first-moment error.
- **How it tests it:** It first adds `0.01` to one weight and expects a `0.01`
  sum error. It then transfers `0.01` from the center to the positive-x neighbor;
  the sum must remain correct while the first-moment error becomes `0.01` cell.

### `test_mapping.py`

#### `MappingTests.test_mapping_round_trip_preserves_order_and_metadata`

- **What it tests:** Weight replacement and `.id/.lag` serialization preserve
  marker ordering, stencil metadata and values without mutating the input map.
- **How it tests it:** It inserts known marker-dependent weights, confirms the
  original weights remain zero, saves to a temporary directory, reloads both
  files, and compares marker IDs, row order, `i/j/k`, `Vcell`, `eps` and every
  weight value.

### `test_fixed_stencil.py`

#### `FixedStencilTests.test_fixed_stencil_accepts_valid_markers_and_rejects_invalid_state`

- **What it tests:** The fixed-stencil guard accepts a valid stencil and rejects
  states that the current weight-only MPMD protocol cannot represent safely.
- **How it tests it:** A valid 3x3x3 stencil must pass. Moving one marker by one
  x-direction cell width must raise the center-cell-crossing error, and removing
  one of the 27 rows must raise the incomplete-stencil error.

#### `FixedStencilTests.test_grid_line_roundoff_snaps_to_high_index_cell`

- **What it tests:** A marker on an internal grid line follows the shared
  half-open-cell convention and remains in the high-index cell when floating-
  point roundoff places it infinitesimally below that line.
- **How it tests it:** It checks the exact grid-line coordinate and the next
  representable double below it, then verifies that a marker genuinely farther
  into the low-index cell is still rejected.

### `test_mpmd_transport.py`

#### `MPMDTransportTests.test_mpmd_preserves_double_positions_and_float32_weight_order`

- **What it tests:** The Python MPMD loop receives double-precision marker
  positions, preserves the float32 weight values and marker/stencil ordering
  across repeated exchanges, and shuts down cleanly.
- **How it tests it:** It starts two real MPI application contexts.
  `mpmd_client.py` emulates the C++ root rank, while `mpmd_server.py` runs the
  production `serve_mpmd()` loop with deterministic weights that encode the
  exchange, marker, stencil position and received x-coordinate. One marker is
  placed exactly on a grid line to exercise the double-plus-snap path. The
  client performs two exact array comparisons, sends a zero marker count, and
  the parent process enforces a 30-second deadlock timeout.

This test verifies the Python server and the agreed wire protocol; it does not
compile or execute the IAMReX C++ implementation itself.

### `test_ml_solver.py`

#### `MLWeightSolverTests.test_ml_output_is_finite_normalized_and_close_to_rkpm`

- **What it tests:** The Transolver backend returns structurally valid,
  normalized weights whose values and first moment remain reasonably close to
  the traditional RKPM reference.
- **How it tests it:** It runs both solvers on the same three-marker fixture,
  checks shape `(3, 27)`, finiteness and normalization within `1e-6`, and then
  requires RMSE below `2e-3`, maximum weight error below `1e-2`, and maximum
  first-moment error below `1e-2` cell. It skips when PyTorch or model files are
  unavailable.

## Supporting Files

The following Python files support the tests but are not discovered as
independent tests because their names do not start with `test_`:

- `common.py` constructs the shared three-marker fixture, including a nonzero
  domain origin, anisotropic grid spacing and complete 3x3x3 stencils.
- `mpmd_client.py` acts as the IAMReX-side protocol emulator used only by the
  MPMD transport test.
- `mpmd_server.py` starts the production transport loop with the deterministic
  solver used only by that test.
- `verify_rkpm_alignment.py` provides the reproduction-residual calculations
  reused by two tests and can also be run as a standalone full-mapping checker.

The full mapping verifier remains directly executable:

```bash
cd /path/to/IAMReX/Tools/RKPM_weight
python3 unit_test/verify_rkpm_alignment.py
```
