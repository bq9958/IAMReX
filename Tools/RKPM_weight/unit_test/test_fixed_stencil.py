# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for fixed 3x3x3 MPMD stencil validation."""

from __future__ import annotations

import copy
import unittest

from unit_test.common import build_small_fixture

from src.weight_solver import validate_fixed_stencils, validate_stencils


class FixedStencilTests(unittest.TestCase):
    def test_fixed_stencil_accepts_valid_markers_and_rejects_invalid_state(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids = validate_stencils(lag_map, expected_size=27)
        validate_fixed_stencils(positions, lag_map, grid, marker_ids)

        crossed = positions.copy()
        crossed[1, 0] += grid.dx[0]
        with self.assertRaisesRegex(ValueError, "left the center cell"):
            validate_fixed_stencils(crossed, lag_map, grid, marker_ids)

        incomplete = copy.deepcopy(lag_map)
        incomplete[2].pop()
        with self.assertRaisesRegex(ValueError, "not a complete 3x3x3 stencil"):
            validate_fixed_stencils(positions, incomplete, grid, marker_ids)


if __name__ == "__main__":
    unittest.main()
