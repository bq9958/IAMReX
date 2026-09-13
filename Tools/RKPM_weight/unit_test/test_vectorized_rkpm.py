# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Equivalence tests for vectorized and batched traditional RKPM solves."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from unit_test.common import build_small_fixture

from src import window
from src.weight_solver import RKPMSolver, stencil_centers


def _scalar_kernel(value: float) -> float:
    absolute = abs(value)
    if 0.5 <= absolute <= 1.5:
        return (1.0 / 6.0) * (
            5.0
            - 3.0 * absolute
            - np.sqrt(1.0 - 3.0 * (1.0 - absolute) ** 2)
        )
    if absolute <= 0.5:
        return (1.0 / 3.0) * (1.0 + np.sqrt(1.0 - 3.0 * value**2))
    return 0.0


def _scalar_reference(support, marker, scales, lagrangian_volume):
    """Evaluate the pre-vectorization equations with explicit scalar loops."""
    moment = np.zeros((10, 10))
    basis_rows = []
    base_weights = []
    for x, y, z, cell_volume in support:
        displacement = np.asarray([x, y, z]) - marker
        dx, dy, dz = displacement
        basis = np.asarray(
            [1.0, dx, dy, dz, dx * dy, dy * dz, dz * dx, dx**2, dy**2, dz**2]
        )
        normalized = displacement / scales
        base = (
            _scalar_kernel(float(normalized[0]))
            * _scalar_kernel(float(normalized[1]))
            * _scalar_kernel(float(normalized[2]))
            * cell_volume
            / lagrangian_volume
        )
        moment += np.outer(basis, basis) * base
        basis_rows.append(basis)
        base_weights.append(base)

    right_hand_side = np.zeros(10)
    right_hand_side[0] = 1.0
    correction = np.linalg.solve(moment, right_hand_side)
    return np.asarray(base_weights) * (np.asarray(basis_rows) @ correction)


class VectorizedRKPMTests(unittest.TestCase):
    def test_batched_solve_uses_explicit_rhs_column_for_numpy_2(self):
        matrices = np.broadcast_to(np.eye(10), (3, 10, 10)).copy()
        original_solve = np.linalg.solve

        def require_explicit_columns(coefficients, right_hand_side):
            self.assertEqual(right_hand_side.shape, (3, 10, 1))
            return original_solve(coefficients, right_hand_side)

        with mock.patch.object(
            window.np.linalg,
            "solve",
            side_effect=require_explicit_columns,
        ):
            corrections = window.compute_b_I(matrices)

        expected = np.zeros((3, 10))
        expected[:, 0] = 1.0
        self.assertEqual(corrections.shape, (3, 10))
        np.testing.assert_array_equal(corrections, expected)

    def test_vectorized_batches_match_independent_scalar_equations(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids = tuple(sorted(lag_map))
        centers = stencil_centers(lag_map, grid, marker_ids)
        cell_volume = float(np.prod(grid.dx))
        supports = np.empty((len(marker_ids), 27, 4))
        supports[..., :3] = centers
        supports[..., 3] = cell_volume
        lagrangian_volumes = np.asarray(
            [lag_map[marker_id][0]["eps"] * cell_volume for marker_id in marker_ids]
        )
        scales = grid.dx * 1.001

        vectorized = window.compute_modified_window_functions_batch(
            supports,
            positions,
            scales[0],
            scales[1],
            scales[2],
            lagrangian_volumes,
        )
        scalar = np.asarray(
            [
                _scalar_reference(
                    supports[index],
                    positions[index],
                    scales,
                    lagrangian_volumes[index],
                )
                for index in range(len(marker_ids))
            ]
        )
        np.testing.assert_allclose(vectorized, scalar, rtol=1.0e-11, atol=1.0e-12)

    def test_solver_chunking_preserves_marker_and_stencil_order(self):
        grid, positions, _, lag_map = build_small_fixture()
        single_ids, single_batch = RKPMSolver(
            grid, batch_size=len(positions)
        ).solve_array(positions, lag_map)
        chunked_ids, chunked = RKPMSolver(
            grid, batch_size=2
        ).solve_array(positions, lag_map)

        self.assertEqual(single_ids, chunked_ids)
        np.testing.assert_allclose(chunked, single_batch, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
