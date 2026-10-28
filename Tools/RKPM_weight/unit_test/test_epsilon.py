# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical and file-contract tests for matrix-solved marker volumes."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unit_test.common import TEST_DIR, build_small_fixture

import main
from src import error, mapping
from src.epsilon import EpsilonSolveError, build_interpolation_matrix, solve_epsilon
from src.weight_solver import RKPMSolver, stencil_centers


def mapping_from_dense(weights):
    # Nonzero, nonlocal global coordinates deliberately test cell-ID packing.
    return {
        mid: [
            {"i": 100 + cell, "j": -7, "k": 42, "w": float(weight),
             "Vcell": 1.0, "eps": 0.5}
            for cell, weight in enumerate(row) if weight != 0.0
        ]
        for mid, row in enumerate(weights)
    }


class EpsilonTests(unittest.TestCase):
    def test_dimensionless_operator_matches_legacy_spread_then_interpolate(self):
        lag_map = mapping_from_dense([[0.8, 0.3, -0.1], [0.0, 0.4, 0.6]])
        _, interpolation = build_interpolation_matrix(lag_map)
        cell_volume = 0.027
        physical_volumes = np.asarray([0.04, 0.07])
        # Legacy windows have units 1/volume. Share values only at identical
        # global Euler cells, including supports of different lengths.
        dispersion = {}
        for mid, rows in lag_map.items():
            for row in rows:
                cell = (row["i"], row["j"], row["k"])
                dispersion[cell] = dispersion.get(cell, 0.0) + physical_volumes[mid] * row["w"] / cell_volume
        legacy = np.asarray([
            sum(dispersion[(row["i"], row["j"], row["k"])] * row["w"] for row in rows)
            for rows in lag_map.values()
        ])
        actual = interpolation @ (interpolation.T @ (physical_volumes / cell_volume))
        np.testing.assert_allclose(actual, legacy, atol=1.0e-14)

    def test_coupled_solve_matches_dense_reference_and_volume_units(self):
        weights = np.asarray([[0.75, 0.25, 0.0], [0.0, 0.25, 0.75]])
        result = solve_epsilon(mapping_from_dense(weights), 0.125, spectrum=True)
        expected = np.linalg.solve(weights @ weights.T, np.ones(2))
        np.testing.assert_allclose(result.eps, expected, rtol=1.0e-12)
        np.testing.assert_allclose(result.volumes, expected * 0.125)
        self.assertLess(result.diagnostics["max_residual"], 1.0e-12)
        self.assertTrue(result.diagnostics["forcing_contracts"])

    def test_large_eps_can_be_stable_for_sparse_nonoverlapping_markers(self):
        weights = np.zeros((2, 54))
        weights[0, :27] = 1.0 / 27
        weights[1, 27:] = 1.0 / 27
        result = solve_epsilon(mapping_from_dense(weights), 0.1, spectrum=True)
        np.testing.assert_allclose(result.eps, [27, 27])
        self.assertLess(result.diagnostics["forcing_spectral_radius"], 1.0e-12)

    def test_exact_constant_reproduction_does_not_guarantee_stability(self):
        result = solve_epsilon(mapping_from_dense([[2, -1], [-1, 2]]), 1.0, spectrum=True)
        np.testing.assert_allclose(result.eps, [1, 1])
        self.assertAlmostEqual(result.diagnostics["lambda_max"], 9.0)
        self.assertAlmostEqual(result.diagnostics["forcing_spectral_radius"], 8.0)
        self.assertFalse(result.diagnostics["forcing_contracts"])

    def test_negative_volumes_are_rejected_even_with_small_residual(self):
        with self.assertRaisesRegex(EpsilonSolveError, "nonpositive marker volumes") as raised:
            solve_epsilon(mapping_from_dense([[1, 0], [0.8, 0.2]]), 1.0)
        self.assertFalse(raised.exception.result.diagnostics["accepted"])
        np.testing.assert_allclose(raised.exception.result.eps, [-3, 5])

    def test_nonconvergence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not converge"):
            solve_epsilon(mapping_from_dense([[1, 0], [0.2, 0.8]]), 1.0, maxiter=1)

    def test_redundant_markers_are_not_certified_contractive(self):
        result = solve_epsilon(mapping_from_dense([[0.5, 0.5], [0.5, 0.5]]), 1.0, spectrum=True)
        self.assertFalse(result.diagnostics["forcing_contracts"])
        self.assertAlmostEqual(result.diagnostics["lambda_min"], 0.0)

    def test_file_recompute_solves_eps_without_requiring_prescribed_volume(self):
        grid, positions, id_map, lag_map = build_small_fixture()
        for rows in lag_map.values():
            for row in rows:
                row["eps"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping.save_mappings_txt(id_map, lag_map, str(root / "seed"))
            args = main.parse_args([
                "--volume-mode", "solve-epsilon", "--epsilon-spectrum",
                "--id", str(root / "seed.id"), "--lag", str(root / "seed.lag"),
                "--output-prefix", str(root / "solved"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                main.recalculate_mapping(args, RKPMSolver(grid))
            solved = mapping.load_lag_map(root / "solved.lag")
            report = json.loads((root / "solved.epsilon.json").read_text())
        _, interpolation = build_interpolation_matrix(solved)
        eps = np.asarray([solved[mid][0]["eps"] for mid in sorted(solved)])
        np.testing.assert_allclose(interpolation @ (interpolation.T @ eps), 1.0, atol=1.0e-8)
        for mid in solved:
            self.assertTrue(all(row["eps"] == eps[mid] for row in solved[mid]))
            self.assertTrue(all(row["Vcell"] == 1.0 for row in solved[mid]))
            self.assertTrue(all(row["eps"] == 0.0 for row in lag_map[mid]))
        np.testing.assert_allclose([entry["V_lag"] for entry in report["markers"]], eps * np.prod(grid.dx))
        # Conservation diagnostics must accept nonuniform marker volumes.
        centers = stencil_centers(solved, grid, sorted(solved))
        supports = [np.column_stack((row, np.full(len(row), np.prod(grid.dx)))) for row in centers]
        weights = [[row["w"] for row in solved[mid]] for mid in sorted(solved)]
        with contextlib.redirect_stdout(io.StringIO()):
            force_error, torque_error = error.compute_conservation_check(
                positions.mean(axis=0), np.unique(centers.reshape(-1, 3), axis=0),
                positions, supports, weights, eps * np.prod(grid.dx),
            )
        self.assertLess(force_error, 1.0e-10)
        self.assertLess(torque_error, 1.0e-10)

    def test_geometry_generation_solves_volumes_for_planar_cloud(self):
        grid, positions, _, _ = build_small_fixture()
        positions[:, 2] = positions[0, 2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "points.txt", positions)
            args = main.parse_args([
                "--volume-mode", "solve-epsilon", "--world-frame",
                "--geometry", str(root / "points.txt"),
                "--output-prefix", str(root / "planar"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                main.generate_from_geometry(args, grid, RKPMSolver(grid))
            solved = mapping.load_lag_map(root / "planar.lag")
        self.assertTrue(all(rows[0]["eps"] > 0.0 for rows in solved.values()))

    def test_runtime_config_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "inputs.rkpm"
            config.write_text("rkpm.volume_mode = solve-epsilon\nrkpm.epsilon_rtol = 1e-9\n"
                              "epsilon_maxiter = 234\nepsilon_spectrum = true\n")
            args = main.parse_args(["--config", str(config), "--epsilon-rtol", "1e-7",
                                    "--no-epsilon-spectrum"])
        self.assertEqual(args.volume_mode, "solve-epsilon")
        self.assertEqual(args.epsilon_rtol, 1.0e-7)
        self.assertEqual(args.epsilon_maxiter, 234)
        self.assertFalse(args.epsilon_spectrum)

    def test_mpmd_and_ml_reject_volume_updates(self):
        for mode in (["--transport", "mpmd"], ["--solver", "ml"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.parse_args(["--volume-mode", "solve-epsilon", *mode])

    def test_real_case_retains_failed_diagnostics_without_writing_mapping(self):
        fixture = TEST_DIR / "fixtures" / "real_case"
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "nested" / "rejected"
            args = main.parse_args([
                "--volume-mode", "solve-epsilon",
                "--inputs", str(fixture / "inputs.3d.flow_past_ellipsoid"),
                "--id", str(fixture / "rkpm_mappings.id"),
                "--lag", str(fixture / "rkpm_mappings.lag"),
                "--output-prefix", str(prefix),
            ])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(EpsilonSolveError):
                main.run(args)
            report = json.loads(Path(str(prefix) + ".epsilon.json").read_text())
            self.assertFalse(Path(str(prefix) + ".id").exists())
            self.assertFalse(Path(str(prefix) + ".lag").exists())
        self.assertFalse(report["accepted"])
        self.assertEqual(report["marker_count"], 2830)
        self.assertGreater(report["nonpositive_count"], 0)
        self.assertLess(report["relative_residual"], 1.1e-8)


if __name__ == "__main__":
    unittest.main()
