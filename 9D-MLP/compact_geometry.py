"""Compact structured-Cartesian geometry features for the nonuniform RKPM model."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

OFFSETS = np.asarray(
    [(i, j, k) for k in (-1, 0, 1) for j in (-1, 0, 1) for i in (-1, 0, 1)],
    dtype=np.float32,
)


def compact_features(phase: np.ndarray, widths: np.ndarray | None,
                     sigma: np.ndarray | None = None, include_sigma: bool = False,
                     include_asymmetry: bool = False,
                     xi: np.ndarray | None = None) -> np.ndarray:
    """Return 9D (phase + six width ratios) or 12D inputs."""
    phase = np.asarray(phase, dtype=np.float32)
    if xi is None:
        if widths is None:
            raise ValueError("widths or xi are required to construct geometry ratios")
        widths = np.asarray(widths, dtype=np.float32)
        ratios = np.stack((widths[:, :, 0] / widths[:, :, 1],
                           widths[:, :, 2] / widths[:, :, 1]), axis=-1).reshape(len(widths), 6)
    else:
        # Cell-center differences cancel particle phase; this recovers the
        # source-precision width ratios from the stored production coordinates.
        xi = np.asarray(xi, dtype=np.float64)
        ratios = np.empty((len(xi), 3, 2), dtype=np.float64)
        center = xi[:, 13, :]  # (0,0,0) in k/j/i order
        for axis in range(3):
            off_left = np.zeros(3, dtype=np.float32); off_left[axis] = -1
            off_right = np.zeros(3, dtype=np.float32); off_right[axis] = 1
            left = int(np.flatnonzero(np.all(OFFSETS == off_left, axis=1))[0])
            right = int(np.flatnonzero(np.all(OFFSETS == off_right, axis=1))[0])
            ratios[:, axis, 0] = -2.0 * (xi[:, left, axis] - center[:, axis]) - 1.0
            ratios[:, axis, 1] = 2.0 * (xi[:, right, axis] - center[:, axis]) - 1.0
        ratios = ratios.reshape(len(xi), 6)
    values = [phase, ratios]
    if include_sigma:
        if sigma is None:
            raise ValueError("sigma is required for 12D features")
        values.append(np.asarray(sigma, dtype=np.float32))
    if include_asymmetry:
        if not include_sigma:
            raise ValueError("asymmetry extension is defined after support scales")
        ratios3 = ratios.reshape(len(ratios), 3, 2)
        values.append((ratios3[:, :, 1] - ratios3[:, :, 0]).astype(np.float32))
    return np.concatenate(values, axis=1).astype(np.float32, copy=False)


def torch_geometry_from_compact(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct particle-centered coordinates and central-cell-normalized volumes."""
    phase, ratios = features[:, :3], features[:, 3:9]
    n = len(features)
    offsets = torch.as_tensor(OFFSETS, device=features.device, dtype=features.dtype)
    centers = torch.zeros((n, 27, 3), device=features.device, dtype=features.dtype)
    widths = torch.ones((n, 3, 3), device=features.device, dtype=features.dtype)
    ratios3 = ratios.reshape(n, 3, 2)
    widths[:, :, 0] = ratios3[:, :, 0]
    widths[:, :, 2] = ratios3[:, :, 1]
    for axis in range(3):
        left, right = ratios3[:, axis, 0:1], ratios3[:, axis, 1:2]
        off = offsets[:, axis].unsqueeze(0)
        centers[:, :, axis] = torch.where(off < 0, -0.5 * (1.0 + left),
                                         torch.where(off > 0, 0.5 * (1.0 + right), 0.0))
    xi = centers - (phase - 0.5).unsqueeze(1)
    cell_widths = torch.stack([widths[:, axis, (offsets[:, axis].to(torch.long) + 1)]
                               for axis in range(3)], dim=-1)
    volumes = torch.prod(cell_widths, dim=-1)
    return xi, volumes


def compact_constraint(features: torch.Tensor) -> torch.Tensor:
    xi, _ = torch_geometry_from_compact(features)
    x, y, z = xi.unbind(dim=-1)
    basis = torch.stack((torch.ones_like(x), x, y, z, x*y, y*z, z*x,
                         x*x, y*y, z*z), dim=-1)
    return basis.transpose(1, 2)


class CompactGeometryMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim not in (9, 12, 15):
            raise ValueError(f"unsupported compact dimension {input_dim}")
        self.input_dim = input_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(),
                                    nn.Linear(128, 128), nn.GELU(),
                                    nn.Linear(128, 64), nn.GELU(),
                                    nn.Linear(64, 27))

    def raw(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)

    def forward(self, features: torch.Tensor, projected: bool = False) -> torch.Tensor:
        output = self.raw(features)
        if not projected:
            return output
        A = compact_constraint(features)
        gram = A @ A.transpose(1, 2)
        rhs = torch.zeros((len(features), 10, 1), device=features.device, dtype=output.dtype)
        rhs[:, 0, 0] = 1.0
        residual = rhs - A @ output.unsqueeze(-1)
        correction = torch.linalg.solve(gram, residual)
        return output + (A.transpose(1, 2) @ correction).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
