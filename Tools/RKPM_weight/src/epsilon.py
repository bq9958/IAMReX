# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Solve marker volumes using the actual interpolation/spreading operators.

On the uniform finest grid, J stores dimensionless interpolation weights
``w * Vcell`` from the .lag contract (Vcell is normally 1). The legacy physical
operator is A_phys = J J.T / Delta_V. Thus solving
``(J J.T) eps = 1`` gives exactly eps = V_lag / Delta_V for IAMReX.
Only J is assembled; applications of A use sparse interpolation and spreading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, eigsh

from .weight_solver import validate_stencils


@dataclass(frozen=True)
class EpsilonResult:
    marker_ids: tuple[int, ...]
    eps: np.ndarray
    volumes: np.ndarray
    diagnostics: dict


class EpsilonSolveError(ValueError):
    """An inadmissible converged solution, retained for research diagnostics."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def build_interpolation_matrix(lag_map):
    """Pack global integer cell indices into a sparse marker-by-cell matrix."""
    marker_ids = tuple(validate_stencils(lag_map))
    if not marker_ids:
        raise ValueError("Cannot solve epsilon for an empty marker set")
    counts = np.asarray([len(lag_map[mid]) for mid in marker_ids])
    cells = []
    weights = []
    for mid in marker_ids:
        for row in lag_map[mid]:
            cell = np.asarray([row[axis] for axis in ("i", "j", "k")], dtype=float)
            if not np.all(np.isfinite(cell)) or np.any(cell != np.floor(cell)):
                raise ValueError(f"Marker {mid} has noninteger cell indices")
            cells.append(cell.astype(np.int64))
            weights.append(float(row["w"]) * float(row.get("Vcell", 1.0)))
    weights = np.asarray(weights)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Interpolation weights contain NaN or Inf")
    _, columns = np.unique(np.asarray(cells), axis=0, return_inverse=True)
    rows = np.repeat(np.arange(len(marker_ids)), counts)
    interpolation = coo_matrix(
        (weights, (rows, columns)),
        shape=(len(marker_ids), int(columns.max()) + 1),
    ).tocsr()
    interpolation.eliminate_zeros()
    return marker_ids, interpolation


def forcing_spectrum(interpolation, eps, *, tolerance=1.0e-7, maxiter=1000):
    """Extremal eigenvalues of sqrt(D) J J.T sqrt(D), D = diag(eps).

    This is similar to the marker correction J J.T D for positive eps.
    The spectral radius below describes only the fixed-position direct-forcing
    subiteration, excluding projection, time integration and rigid-body motion.
    """
    eps = np.asarray(eps, dtype=float)
    if (eps.shape != (interpolation.shape[0],)
            or np.any(eps <= 0.0) or not np.all(np.isfinite(eps))):
        raise ValueError("Spectral analysis requires finite positive marker eps")
    root = np.sqrt(eps)
    scaled = interpolation.multiply(root[:, None]).tocsr()
    count = len(eps)
    if count <= 256:
        eigenvalues = np.linalg.eigvalsh((scaled @ scaled.T).toarray())
        minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
    else:
        transpose = scaled.T.tocsr()
        operator = LinearOperator(
            (count, count), matvec=lambda x: scaled @ (transpose @ x), dtype=float
        )
        # A deterministic starting vector makes diagnostics reproducible.
        start = np.random.default_rng(0).normal(size=count)
        minimum = float(eigsh(
            operator, k=1, which="SA", v0=start, tol=tolerance,
            maxiter=maxiter, return_eigenvectors=False,
        )[0])
        maximum = float(eigsh(
            operator, k=1, which="LA", v0=start, tol=tolerance,
            maxiter=maxiter, return_eigenvectors=False,
        )[0])
    radius = max(abs(1.0 - minimum), abs(1.0 - maximum))
    return {
        "lambda_min": minimum,
        "lambda_max": maximum,
        "forcing_spectral_radius": float(radius),
        # A near-null mode cannot certify strict contraction numerically.
        "forcing_contracts": bool(minimum > 1.0e-10 and maximum < 2.0 - 1.0e-10),
    }


def solve_epsilon(lag_map, cell_volume, *, rtol=1.0e-8, maxiter=1000, spectrum=False):
    """Solve A eps = 1 with Jacobi-preconditioned CG and verify the result.

    A is symmetric positive semidefinite; redundant markers can make it singular.
    Nonconvergence and nonpositive volumes are errors, never silently clipped or
    replaced by prescribed values. A compatible singular system can converge;
    convergence alone does not establish uniqueness or forcing stability.
    """
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError("Eulerian cell volume must be finite and positive")
    if not np.isfinite(rtol) or rtol <= 0.0 or maxiter <= 0:
        raise ValueError("epsilon rtol and maxiter must be positive")
    marker_ids, interpolation = build_interpolation_matrix(lag_map)
    count = len(marker_ids)
    transpose = interpolation.T.tocsr()
    operator = LinearOperator(
        (count, count), matvec=lambda x: interpolation @ (transpose @ x), dtype=float
    )
    diagonal = np.asarray(interpolation.multiply(interpolation).sum(axis=1)).ravel()
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("A has a zero or invalid diagonal; check marker supports")
    preconditioner = LinearOperator(
        (count, count), matvec=lambda x: x / diagonal, dtype=float
    )
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    rhs = np.ones(count)
    eps, info = cg(
        operator, rhs, M=preconditioner, rtol=rtol, atol=0.0,
        maxiter=maxiter, callback=callback,
    )
    residual = operator @ eps - rhs
    relative = float(np.linalg.norm(residual) / np.sqrt(count))
    if (info != 0 or not np.all(np.isfinite(eps))
            or not np.isfinite(relative)
            or relative > max(1.1 * rtol, 1.0e-14)):
        raise ValueError(
            f"A eps = 1 did not converge: info={info}, iterations={iterations}, "
            f"relative residual={relative:.6e}. Check marker redundancy/supports "
            "or increase --epsilon-maxiter."
        )
    nonpositive = int(np.count_nonzero(eps <= 0.0))
    volumes = eps * cell_volume
    diagnostics = {
        "accepted": nonpositive == 0,
        "nonpositive_count": nonpositive,
        "marker_count": count,
        "euler_cell_count": interpolation.shape[1],
        "cell_volume": float(cell_volume),
        "iterations": iterations,
        "relative_residual": relative,
        "max_residual": float(np.max(np.abs(residual))),
        "eps_min": float(eps.min()),
        "eps_mean": float(eps.mean()),
        "eps_max": float(eps.max()),
        "volume_min": float(volumes.min()),
        "volume_mean": float(volumes.mean()),
        "volume_max": float(volumes.max()),
    }
    result = EpsilonResult(marker_ids, eps, volumes, diagnostics)
    if nonpositive:
        message = (
            f"A eps = 1 produced {nonpositive} nonpositive marker volumes "
            f"(eps min={eps.min():.6e}, max={eps.max():.6e}, "
            f"relative residual={relative:.6e}). A small residual does not "
            "guarantee positive quadrature weights; no mapping was written."
        )
        diagnostics["failure_reason"] = message
        raise EpsilonSolveError(message, result)
    if spectrum:
        diagnostics.update(forcing_spectrum(interpolation, eps, maxiter=maxiter))
    return result
