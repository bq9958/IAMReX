# RKPM Weight Tool

`main.py` separates the weight solver backend from the output transport through
two independent runtime arguments:

| Argument | Values | Purpose |
|---|---|---|
| `--solver` | `rkpm` / `ml` | Traditional RKPM solve or Transolver inference |
| `--transport` | `file` / `mpmd` | Write `.id/.lag` files or return weights directly to IAMReX |

## Python Environment

Run the tool with a Python 3 interpreter in which the required dependencies are
installed. In the commands below, `python3` means that interpreter; activate a
virtual environment first or replace `python3` with its absolute path when
needed.

NumPy is required for all modes. Point-cloud generation and visualization also
require Matplotlib, ML inference requires PyTorch, and MPMD transport requires
`mpi4py`. The MPI implementation used by `mpi4py` must be compatible with the
one used to build IAMReX.

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
  --solver ml --check-action abort
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
  --output-prefix /path/to/case/rkpm_mappings
```

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

The tool root intentionally contains only `src/`, `unit_test/`, `main.py` and
this README. Bundled mapping/input fixtures, the standalone alignment verifier
and generated reference images live under `unit_test/`.
