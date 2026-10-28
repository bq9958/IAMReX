<!--
SPDX-FileCopyrightText: 2026 IAMReX contributors
SPDX-License-Identifier: BSD-3-Clause
-->

# RKPM Weight Architecture

This document is the maintained architecture map for the RKPM Weight tool. It
covers the production modules, runtime modes, solver backends and external data
contracts. Update it in the same change whenever a module responsibility,
dependency, runtime path, command-line option, data shape or MPMD contract
changes.

## System Architecture

```mermaid
flowchart TB
    subgraph ExternalInputs[External inputs and peers]
        RunConfig["Runtime configuration<br/>solver, transport and explicit geometry frame"]
        AMReXInputs["AMReX inputs file<br/>domain, cells, AMR level"]
        Geometry["Point-cloud geometry<br/>N markers x 3, body or world frame"]
        MappingFiles["Mapping files<br/>.id positions and .lag stencils"]
        ModelFiles["Transolver assets<br/>model code, checkpoint, statistics"]
        IAMReX["IAMReX DiffusedIB_Parallel<br/>C++ MPI application"]
    end

    subgraph Orchestration[Entry and orchestration]
        Main["main.py<br/>parse configuration and dispatch mode"]
        GridSpec["GridSpec<br/>finest-level spacing and domain"]
        SolverFactory["build_solver()<br/>select RKPM or ML backend"]
    end

    subgraph FileMode[File transport]
        GeometryMode["Geometry generation mode"]
        RecomputeMode["Existing mapping mode"]
        SupportGeneration["SI_generated.py<br/>vectorized 3x3x3 support generation"]
        Mapping["mapping.py<br/>load, build, replace and serialize mappings"]
        Diagnostics["error.py and visual.py<br/>conservation checks and optional plots"]
        Epsilon["epsilon.py (optional, SciPy)<br/>matrix-free A eps = 1 and forcing spectrum"]
        MappingOutput["Generated .id and .lag files"]
    end

    subgraph SharedCore[Shared solver core]
        GridIndex["grid_index.py<br/>snap roundoff and compute containing cells"]
        Packed["PackedStencils<br/>cached NumPy stencil arrays"]
        RKPM["RKPMSolver<br/>batched moment-system solve"]
        Window["window.py<br/>kernel, polynomial basis and batched linear algebra"]
        ML["MLWeightSolver<br/>batched Transolver inference"]
        WeightArray["ID-ordered weight array<br/>N markers x stencil size"]
    end

    subgraph MPMDMode[MPMD transport]
        Transport["mpmd_transport.py<br/>receive, validate, solve and send"]
        MomentValidation["validation.py<br/>zeroth- and first-moment residuals"]
    end

    subgraph Verification[Regression protection]
        UnitTests["unit_test/<br/>unit and integration test suite"]
        RealFixture["2830-marker real-case fixture"]
        MPIEmulator["Python MPI client<br/>emulates the C++ root rank"]
    end

    RunConfig --> Main
    AMReXInputs --> GridSpec
    Main --> GridSpec
    Main --> SolverFactory
    Main --> GeometryMode
    Main --> RecomputeMode
    Main --> Transport

    Geometry --> GeometryMode --> SupportGeneration --> Mapping
    RecomputeMode --> Mapping
    MappingFiles --> RecomputeMode
    SolverFactory --> RKPM
    SolverFactory --> ML
    MappingFiles --> Packed
    Mapping --> Packed
    Packed --> RKPM
    Packed --> ML
    RKPM --> Window --> WeightArray
    ModelFiles --> ML --> WeightArray
    GridSpec --> SupportGeneration
    GridSpec --> RKPM
    GridSpec --> ML
    GridIndex -.-> SupportGeneration
    GridIndex -.-> Mapping
    GridIndex -.-> Transport

    WeightArray --> Mapping --> Diagnostics --> MappingOutput
    Mapping -- "file mode, solve-epsilon" --> Epsilon
    Epsilon -- "per-marker eps and JSON diagnostics" --> MappingOutput

    IAMReX -- "marker count and float64 positions" --> Transport
    MappingFiles -- "fixed 27-cell stencil" --> Transport
    Transport --> Packed
    Transport -- "selected solve_array()" --> RKPM
    Transport -- "selected solve_array()" --> ML
    WeightArray --> Transport
    Transport --> MomentValidation
    Transport -- "float32 weights" --> IAMReX

    UnitTests -.-> Main
    UnitTests -.-> SupportGeneration
    UnitTests -.-> Mapping
    UnitTests -.-> RKPM
    UnitTests -.-> ML
    UnitTests -.-> Transport
    UnitTests -.-> Diagnostics
    RealFixture --> UnitTests
    MPIEmulator --> UnitTests
```

## Solver Backend Pipelines

```mermaid
flowchart LR
    Positions["Marker positions<br/>N x 3"]
    Stencils["Cached stencil indices<br/>N x 27 x 3"]
    Centers["Eulerian cell centers<br/>N x 27 x 3"]

    Positions --> Relative["Relative coordinates"]
    Stencils --> Centers --> Relative

    subgraph TraditionalRKPM[Traditional RKPM backend]
        Kernel["Evaluate separable<br/>three-point kernel"]
        Basis["Build 10-term<br/>quadratic basis"]
        Moment["Assemble batched<br/>10 x 10 moment matrices"]
        LinearSolve["Batched np.linalg.solve<br/>explicit column RHS for NumPy 1.x/2.x"]
        Corrected["Corrected RKPM weights"]
        Kernel --> Moment
        Basis --> Moment --> LinearSolve --> Corrected
        Kernel --> Corrected
        Basis --> Corrected
    end

    subgraph MachineLearning[Machine-learning backend]
        Dimensionless["Divide by dx, dy, dz"]
        NormalizeInput["Normalize model features"]
        Transolver["Batched Transolver inference<br/>CPU or CUDA"]
        NormalizeOutput["Denormalize and enforce<br/>unit weight sum"]
        Dimensionless --> NormalizeInput --> Transolver --> NormalizeOutput
    end

    Relative --> Kernel
    Relative --> Basis
    Relative --> Dimensionless
    Corrected --> Weights["Weight array<br/>N x 27"]
    NormalizeOutput --> Weights
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Parse configuration, construct the grid and solver, and dispatch file or MPMD execution. |
| `src/SI_generated.py` | Read and transform geometry, estimate marker volume, and construct vectorized support domains. |
| `src/grid_index.py` | Apply the shared grid-line snapping convention and compute containing-cell indices. |
| `src/mapping.py` | Read, validate, build, update and write `.id/.lag` mappings. |
| `src/weight_solver.py` | Pack static stencils, cache reusable arrays, batch markers and expose RKPM/ML solver backends. |
| `src/window.py` | Evaluate the RKPM kernel, polynomial basis, moment matrices and corrected weights with NumPy. |
| `src/epsilon.py` | Pack sparse interpolation J, solve `(J J^T) eps = 1` with preconditioned CG, validate positive volumes and optionally diagnose the fixed-position forcing spectrum. |
| `src/mpmd_transport.py` | Implement the single-rank Python MPI server and its wire protocol with IAMReX. |
| `src/validation.py` | Compute zeroth- and first-moment residuals on the exact weight array sent to C++. |
| `src/error.py` | Compute vectorized volume, force and torque conservation diagnostics for file mode. |
| `src/visual.py` | Produce optional point-cloud and support-domain diagnostic plots. |
| `unit_test/` | Protect numerical reproduction, batching, mapping, support generation, validation and MPI transport. |

## Runtime Data Contracts

| Data | Shape / type | Owner and lifetime |
|---|---|---|
| Marker positions | `(N, 3)`, float64 | Loaded from `.id` in file mode or received from IAMReX every MPMD exchange. |
| Point-cloud frame | Explicit body or world frame | Body-frame coordinates are translated by `particle_inputs.{x,y,z}`; world-frame coordinates are used unchanged. |
| Stencil indices | `(N, 27, 3)`, integer | Loaded from `.lag`, packed once and cached for the solver lifetime. |
| RKPM metadata | `eps` per stencil row | Loaded from `.lag`; traditional RKPM recovers each marker's Lagrangian volume from it. |
| Volume mode | `prescribed` (default) or `solve-epsilon` | File-mode RKPM can replace prescribed volumes with a coupled solve after computing local windows from unit-eps seeds. ML and MPMD reject volume updates. |
| Solved volumes | `(N,)` float64 `eps` and `V_lag` | `V_lag = eps * prod(dx)`; positive solved eps is copied to all stencil rows of its marker. Conservation diagnostics accept scalar or per-marker physical volumes. |
| Epsilon report | `<output-prefix>.epsilon.json` | File mode records CG residuals, iterations and marker volumes; converged nonpositive results are retained with `accepted=false` and no mapping. Optional `--epsilon-spectrum` adds eigenvalue bounds and the forcing spectral radius for positive solutions. |
| Solver weights | `(N, 27)`, floating point | Produced in marker-ID and `.lag` row order by either backend. |
| MPMD positions | flat `N*3`, `MPI_DOUBLE` | Sent from the C++ root rank to the Python server. |
| MPMD weights | flat `N*27`, `MPI_FLOAT` | Sent from the Python server to the C++ root rank. |

The current MPMD protocol transfers new weights but not new stencil indices.
Consequently, marker motion is valid only while every marker remains inside the
center cell of its cached 3x3x3 stencil. `validate_fixed_stencils()` aborts or
reports the mismatch before stale indices can be used.

`--volume-mode solve-epsilon` uses the actual interpolation multiplier
`J[I,j] = w[I,j] * Vcell[I,j]`. Its dimensionless A is `J J^T`; the legacy
RKPM-3d physical operator is `A / prod(dx)`. Only sparse J is assembled, with
one column per unique global integer Euler cell. Matrix-vector products apply
spreading then interpolation. The solver rejects failed residual checks and
nonpositive volumes before writing a mapping. With positive eps, the optional
spectrum of `sqrt(D) A sqrt(D)` diagnoses only the fixed-position forcing
iteration `I-A D`, not full CFD stability. See README for singular-system,
positivity, coverage and spectral limitations.

## Architecture Maintenance Checklist

When production code changes, review this document in the same pull request:

1. Update the system diagram if a module, dependency or runtime path changes.
2. Update the backend diagram if RKPM or ML preprocessing and inference changes.
3. Update the data-contract table if a shape, dtype, file field or MPI message changes.
4. Update the module table when files are added, removed or assigned new responsibilities.
5. Update the test count and verification nodes after changing the test suite.
