"""Runtime backends for RKPM weights.

Solvers only convert ``marker positions + fixed Euler stencil`` into weights.
The caller selects file output or MPMD transport, allowing traditional RKPM
and ML to share validation and coordinate conversion.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from . import window
from .grid_index import containing_cell_indices


@dataclass(frozen=True)
class GridSpec:
    prob_lo: np.ndarray
    prob_hi: np.ndarray
    n_cell: np.ndarray
    max_level: int
    dx: np.ndarray

    @classmethod
    def from_inputs(cls, path: Path | str) -> "GridSpec":
        params = parse_inputs(path)
        prob_lo = _get_vector(params, "geometry.prob_lo", float)
        prob_hi = _get_vector(params, "geometry.prob_hi", float)
        n_cell = _get_vector(params, "amr.n_cell", int)
        max_level = int(_get_tokens(params, "amr.max_level")[0])

        if not (len(prob_lo) == len(prob_hi) == len(n_cell) == 3):
            raise ValueError("RKPM_weight currently supports only 3D grid parameters")
        if np.any(n_cell <= 0) or max_level < 0:
            raise ValueError("amr.n_cell must be positive and amr.max_level nonnegative")

        dx = (prob_hi - prob_lo) / (n_cell * (2 ** max_level))
        return cls(prob_lo, prob_hi, n_cell, max_level, dx)


def parse_inputs(path: Path | str) -> Dict[str, str]:
    """Parse an AMReX ``key = value`` inputs file."""
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            params[key.strip()] = value.strip()
    return params


def _get_tokens(params: Mapping[str, str], key: str) -> Sequence[str]:
    if key not in params:
        raise KeyError(f"Missing parameter in inputs file: {key}")
    return params[key].split()


def _get_vector(params: Mapping[str, str], key: str, dtype) -> np.ndarray:
    return np.asarray([dtype(value) for value in _get_tokens(params, key)])


def validate_stencils(lag_map, *, expected_size: int | None = None) -> list[int]:
    """Validate marker ID continuity, stencil sizes and grid-index fields."""
    marker_ids = sorted(lag_map)
    if marker_ids != list(range(len(marker_ids))):
        raise ValueError("Marker IDs must be contiguous from zero to match C++ ordering")

    for marker_id in marker_ids:
        rows = lag_map[marker_id]
        if not rows:
            raise ValueError(f"Marker {marker_id} has an empty stencil")
        if expected_size is not None and len(rows) != expected_size:
            raise ValueError(
                f"MPMD requires exactly {expected_size} stencil points per marker; "
                f"marker {marker_id} has {len(rows)}"
            )
        for row in rows:
            for key in ("i", "j", "k"):
                if key not in row:
                    raise ValueError(f"Marker {marker_id} stencil is missing {key}")
    return marker_ids


def positions_as_array(positions, marker_ids: Sequence[int]) -> np.ndarray:
    """Convert an ID mapping or array to an ID-ordered ``[N,3]`` array."""
    if isinstance(positions, Mapping):
        missing = set(marker_ids) - set(positions)
        extra = set(positions) - set(marker_ids)
        if missing or extra:
            raise ValueError(
                f"Position and stencil IDs differ; missing {sorted(missing)[:5]}, "
                f"extra {sorted(extra)[:5]}"
            )
        array = np.asarray(
            [positions[marker_id] for marker_id in marker_ids], dtype=float
        )
    else:
        array = np.asarray(positions, dtype=float)

    if array.shape != (len(marker_ids), 3):
        raise ValueError(
            f"Marker positions must have shape ({len(marker_ids)}, 3), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Marker positions contain NaN or Inf")
    return array


def stencil_centers(lag_map, grid: GridSpec, marker_ids: Sequence[int]):
    """Construct world-coordinate cell centers from global cell indices."""
    centers = []
    for marker_id in marker_ids:
        ijk = np.asarray([
            [row["i"], row["j"], row["k"]] for row in lag_map[marker_id]
        ], dtype=float)
        centers.append(grid.prob_lo + (ijk + 0.5) * grid.dx)
    return centers


def validate_fixed_stencils(positions, lag_map, grid: GridSpec, marker_ids):
    """Ensure each fixed 3x3x3 MPMD stencil still encloses its marker.

    The current C++ protocol returns weights but not ``i/j/k``. Once a marker
    crosses a cell boundary, the old stencil is invalid. Fail explicitly instead
    of silently applying weights to the wrong cells.
    """
    containing = containing_cell_indices(positions, grid.prob_lo, grid.dx)
    for row_index, marker_id in enumerate(marker_ids):
        ijk = np.asarray([
            [row["i"], row["j"], row["k"]] for row in lag_map[marker_id]
        ], dtype=int)
        unique_axes = [np.unique(ijk[:, axis]) for axis in range(3)]
        complete_stencil = len({tuple(cell) for cell in ijk}) == 27
        if (
            any(len(values) != 3 for values in unique_axes)
            or not complete_stencil
        ):
            raise ValueError(f"Marker {marker_id} is not a complete 3x3x3 stencil")
        expected_center = np.asarray([values[1] for values in unique_axes])
        if not np.array_equal(containing[row_index], expected_center):
            raise ValueError(
                f"Marker {marker_id} left the center cell of its fixed stencil: "
                f"current cell {containing[row_index].tolist()}, stencil center "
                f"{expected_center.tolist()}. The current MPMD protocol transfers "
                "weights only and cannot update i/j/k."
            )


class RKPMSolver:
    name = "rkpm"

    def __init__(self, grid: GridSpec):
        self.grid = grid

    def solve(self, positions, lag_map) -> Dict[int, np.ndarray]:
        marker_ids = validate_stencils(lag_map)
        marker_positions = positions_as_array(positions, marker_ids)
        centers = stencil_centers(lag_map, self.grid, marker_ids)

        # The .lag file stores eps = V_lag / Delta_V. Recover the physical cell
        # volume from the finest-grid spacing and then recover V_lag per marker.
        # Do not use row["Vcell"] here: in the current C++ mapping contract that
        # field is an interpolation/spreading multiplier fixed at 1.0, not the
        # physical Eulerian cell volume used by the RKPM moment equations.
        cell_volume = float(np.prod(self.grid.dx))
        weights = {}
        for index, marker_id in enumerate(marker_ids):
            eps = np.asarray(
                [row["eps"] for row in lag_map[marker_id]], dtype=float
            )
            if not np.all(np.isfinite(eps)) or np.any(eps <= 0.0):
                raise ValueError(
                    f"Marker {marker_id} has invalid eps values; cannot recover V_lag"
                )
            if not np.allclose(eps, eps[0], rtol=1.0e-12, atol=0.0):
                raise ValueError(
                    f"Marker {marker_id} has inconsistent eps values in its stencil"
                )

            lagrangian_volume = float(eps[0] * cell_volume)
            support_domain = np.column_stack((
                centers[index],
                np.full(len(centers[index]), cell_volume),
            ))
            solved = window.compute_all_modified_window_functions(
                [support_domain],  # only one marker
                marker_positions[index:index + 1],
                np.asarray([self.grid.dx[0] * 1.001]),
                np.asarray([self.grid.dx[1] * 1.001]),
                np.asarray([self.grid.dx[2] * 1.001]),
                V_lag=lagrangian_volume,
            )
            weights[marker_id] = np.asarray(solved[0], dtype=float)
        return weights


class MLWeightSolver:
    name = "ml"

    def __init__(
        self, grid: GridSpec, model_dir: Path | str, model_code: Path | str
    ):
        self.grid = grid
        self.model_dir = Path(model_dir).resolve()
        self.model_code = Path(model_code).resolve()
        self._load_model()

    def _load_model(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The ML solver requires PyTorch") from exc

        if not self.model_code.is_file():
            raise FileNotFoundError(f"Transolver model definition not found: {self.model_code}")
        spec = importlib.util.spec_from_file_location(
            "rkpm_weight_transolver_slim", self.model_code
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load model definition: {self.model_code}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        checkpoint_path = self.model_dir / "model_best.pt"
        stats_path = self.model_dir / "norm_stats.npz"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = module.TransolverSlim(**checkpoint["cfg"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        stats = np.load(stats_path)
        self.xm = stats["xm"]
        self.xs = stats["xs"]
        self.ym = stats["ym"]
        self.ys = stats["ys"]
        if np.any(self.xs == 0):
            raise ValueError(f"Model normalization parameter xs contains zero: {stats_path}")
        self.torch = torch

    def solve(self, positions, lag_map) -> Dict[int, np.ndarray]:
        marker_ids = validate_stencils(lag_map, expected_size=27)
        marker_positions = positions_as_array(positions, marker_ids)
        centers = stencil_centers(lag_map, self.grid, marker_ids)
        relative = np.asarray([
            (marker_centers - marker_positions[index]) / self.grid.dx
            for index, marker_centers in enumerate(centers)
        ], dtype=np.float32)

        normalized = (relative - self.xm) / self.xs
        batch = self.torch.as_tensor(normalized, dtype=self.torch.float32)
        with self.torch.no_grad():
            prediction = self.model(fx=batch, embedding=batch)
        weights = prediction.detach().cpu().numpy()[..., 0] * self.ys + self.ym

        if weights.shape != (len(marker_ids), 27):
            raise ValueError(
                f"ML output must have shape ({len(marker_ids)}, 27), got {weights.shape}"
            )
        sums = weights.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(weights)) or np.any(np.abs(sums) < 1.0e-12):
            raise ValueError("ML output contains NaN/Inf or a near-zero marker weight sum")
        weights = weights / sums
        return {
            marker_id: np.asarray(weights[index], dtype=float)
            for index, marker_id in enumerate(marker_ids)
        }


def build_solver(name: str, grid: GridSpec, *, model_dir=None, model_code=None):
    if name == "rkpm":
        return RKPMSolver(grid)
    if name == "ml":
        if model_dir is None or model_code is None:
            raise ValueError("The ML solver requires --model-dir and --model-code")
        return MLWeightSolver(grid, model_dir, model_code)
    raise ValueError(f"Unknown solver: {name}")
