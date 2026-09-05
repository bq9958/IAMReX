"""Build 3D RKPM mappings.

This module provides marker-ID-to-coordinate and Lagrangian-to-Eulerian
mappings. Eulerian points are global cell centers on the solver's finest grid,
so their indices are computed directly as
``floor((x - prob_lo) / dx_finest)`` without a local-grid offset. This avoids
the ambiguity caused by the former ``int(sx/dx)`` truncation.
"""

import ast
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def build_lagrangian_id_to_coord_map(lagrangian_points: np.ndarray) -> Dict[int, tuple]:
    """Build ``{marker_id: (xp, yp, zp)}`` from an ``(Ne, 3)`` array."""
    id_to_coord_map = {}

    for lag_id, coord in enumerate(lagrangian_points):
        id_to_coord_map[lag_id] = tuple(coord)

    return id_to_coord_map


def load_id_map(filename: Union[str, Path]) -> Dict[int, tuple]:
    """Read ``rkpm_mappings.id`` and validate marker IDs and coordinates."""
    with open(filename, "r", encoding="utf-8") as f:
        raw = ast.literal_eval(f.read())

    if not isinstance(raw, dict):
        raise ValueError(f"The top level of an ID file must be a dictionary: {filename}")

    result = {}
    for marker_id, coord in raw.items():
        marker_id = int(marker_id)
        if len(coord) != 3:
            raise ValueError(f"Marker {marker_id} does not have 3D coordinates: {coord}")
        result[marker_id] = tuple(float(value) for value in coord)
    return result


def load_lag_map(
    filename: Union[str, Path]
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Read stencil and weight metadata from ``rkpm_mappings.lag``."""
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    result = {}
    for marker_id, body in re.findall(r"(\d+)\s*:\s*\[(.*?)\]", text, re.S):
        rows = []
        for entry in re.findall(r"\{([^}]*)\}", body):
            values = dict(re.findall(
                rf'"(\w+)"\s*:\s*({NUMBER_PATTERN})', entry
            ))
            missing = {"i", "j", "k"} - values.keys()
            if missing:
                raise ValueError(
                    f"Marker {marker_id} stencil entry is missing fields: {sorted(missing)}"
                )
            rows.append({
                "i": int(values["i"]),
                "j": int(values["j"]),
                "k": int(values["k"]),
                "w": float(values.get("w", 0.0)),
                "Vcell": float(values.get("Vcell", 1.0)),
                "eps": float(values.get("eps", 0.0)),
            })
        result[int(marker_id)] = rows

    if not result:
        raise ValueError(f"No markers found in lag file: {filename}")
    return result


def validate_mapping_ids(
    id_to_coord_map: Mapping[int, Sequence[float]],
    lag_to_eul_map: Mapping[int, Sequence[Mapping[str, Union[int, float]]]],
    *,
    require_contiguous: bool = True,
) -> List[int]:
    """Validate matching marker sets and return their deterministic order."""
    coord_ids = set(id_to_coord_map)
    stencil_ids = set(lag_to_eul_map)
    if coord_ids != stencil_ids:
        only_id = sorted(coord_ids - stencil_ids)
        only_lag = sorted(stencil_ids - coord_ids)
        raise ValueError(
            f".id/.lag marker IDs differ; only in .id: {only_id[:5]}, "
            f"only in .lag: {only_lag[:5]}"
        )

    marker_ids = sorted(stencil_ids)
    if require_contiguous and marker_ids != list(range(len(marker_ids))):
        raise ValueError(
            "Marker IDs must be contiguous from zero to match the C++ flat array"
        )
    return marker_ids


def replace_mapping_weights(
    lag_to_eul_map: Mapping[int, Sequence[Mapping[str, Union[int, float]]]],
    weights: Mapping[int, Sequence[float]],
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Preserve stencil/volume metadata and replace only each entry's ``w``."""
    result = deepcopy(lag_to_eul_map)
    if set(result) != set(weights):
        raise ValueError("Weight-result IDs do not match the lag mapping IDs")

    for marker_id, rows in result.items():
        marker_weights = np.asarray(weights[marker_id], dtype=float).reshape(-1)
        if len(rows) != len(marker_weights):
            raise ValueError(
                f"Marker {marker_id} has {len(marker_weights)} weights but "
                f"{len(rows)} stencil entries"
            )
        for row, weight in zip(rows, marker_weights):
            row["w"] = float(weight)
    return result


def build_lag_to_eul_map(
    lagrangian_points: np.ndarray,
    all_S_I: List[np.ndarray],
    all_modified_w: List[List[float]],
    prob_lo: np.ndarray,
    dx_finest: np.ndarray,
    V_lag: float
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Build the Lagrangian-to-Eulerian force-spreading mapping.

    Eulerian points in ``all_S_I`` are global cell centers on the finest grid.
    Therefore, ``floor((x - prob_lo) / dx_finest)`` recovers global cell indices
    directly without a local-to-global offset.
    """
    prob_lo = np.asarray(prob_lo, dtype=float)
    dx_finest = np.asarray(dx_finest, dtype=float)
    lag_to_eul_map = {}

    for lag_id in range(len(lagrangian_points)):
        S_I = all_S_I[lag_id]  # Eulerian points and volumes in the support domain
        modified_w = all_modified_w[lag_id]  # Corrected window-function values

        eulerian_data = []

        for m, (x_mn, y_mn, z_mn, Vcell) in enumerate(S_I):
            # Global cell index = floor((cell center - prob_lo) / dx).
            i = int(np.floor((x_mn - prob_lo[0]) / dx_finest[0]))
            j = int(np.floor((y_mn - prob_lo[1]) / dx_finest[1]))
            k = int(np.floor((z_mn - prob_lo[2]) / dx_finest[2]))

            # Get the weight.
            w = modified_w[m]

            # Add the entry to the mapping.
            eulerian_data.append({
                "i": i,
                "j": j,
                "k": k,
                "w": float(w),
                "Vcell": 1.0,
                "eps": float(V_lag) / float(Vcell)
            })

        # Sort by k, then j, then i (k-major order).
        sort_keys = [(item["k"], item["j"], item["i"]) for item in eulerian_data]
        # Convert to an array for lexsort.
        sort_keys = np.array(sort_keys)  # shape: (N, 3)

        # lexsort uses the last key as primary, hence the (i, j, k) argument.
        indices = np.lexsort((sort_keys[:, 0], sort_keys[:, 1], sort_keys[:, 2]))

        # Reorder entries using the sorted indices.
        eulerian_data = [eulerian_data[idx] for idx in indices]

        # Store the marker mapping.
        lag_to_eul_map[lag_id] = eulerian_data

    return lag_to_eul_map

def save_mappings_txt(
    id_to_coord_map: Dict[int, tuple],
    lag_to_eul_map: Dict[int, List[Dict[str, Union[int, float]]]],
    filename: str
) -> None:
    """Write ID-to-coordinate and Lagrangian-to-Eulerian text mappings."""
    # Write the marker-ID-to-coordinate mapping.
    with open(filename + ".id", 'w') as f:
        f.write("{\n")
        for ids, (x, y, z) in id_to_coord_map.items():
            f.write(f"    {ids}: ({x}, {y}, {z}),\n")
        f.write("}")

    # Write lag_to_eul_map in the format consumed by IAMReX.
    with open(filename + ".lag", 'w') as f:
        f.write("{\n")
        for lag_id, eul_list in lag_to_eul_map.items():
            # Write the marker ID and its Eulerian stencil.
            f.write(f"    {lag_id}: [\n")
            for eul_info in eul_list:
                # Format one Eulerian cell entry.
                line = (
                    f"        {{"
                    f"\"i\": {eul_info['i']}, "
                    f"\"j\": {eul_info['j']}, "
                    f"\"k\": {eul_info['k']}, "
                    f"\"w\": {eul_info['w']}, "
                    f"\"Vcell\": {eul_info['Vcell']}, "
                    f"\"eps\": {eul_info['eps']}"
                    f"}},\n"
                )
                f.write(line)
            f.write("    ],\n")  # End the current marker list.
        f.write("}")  # End the mapping dictionary.
