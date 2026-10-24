"""Standalone 9D-MLP inference for complete structured 3x3x3 stencils.

Call ``prepare`` whenever the particle-to-cell geometry or batch order changes.
The prepared cache can be reused while those particles move inside their cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cached_projection import GeometryCache
from compact_geometry import CompactGeometryMLP


DEFAULT_CHECKPOINT = Path(__file__).with_name("nonuniform_compact9_projected_targeted.pt")


class RKPM9D:
    """Predict 27 RKPM weights from particle phases and local cell widths."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if saved.get("input_dim") != 9 or saved.get("parameter_count") != 27803:
            raise ValueError("Checkpoint is not the expected 9D-MLP (27,803 parameters)")
        self.model = CompactGeometryMLP(9).to(self.device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.ratios: torch.Tensor | None = None
        self.cache: GeometryCache | None = None
        self.projection: str | None = None

    def prepare(self, cell_widths: np.ndarray, projection: str = "cached") -> None:
        """Prepare one row per particle; widths have shape (N,3,3).

        The last axis is [left, center, right], and the middle axis is [x,y,z].
        ``cached`` reuses a geometry-fixed FP32 10C projector. ``dynamic``
        rebuilds each particle-centered projector on every prediction.
        """
        widths = np.asarray(cell_widths, dtype=np.float64)
        if widths.ndim != 3 or widths.shape[1:] != (3, 3) or len(widths) == 0:
            raise ValueError("cell_widths must have shape (N,3,3), with N > 0")
        if not np.isfinite(widths).all() or not (widths > 0).all():
            raise ValueError("cell_widths must be finite and positive")
        if projection not in ("cached", "dynamic"):
            raise ValueError("projection must be 'cached' or 'dynamic'")
        ratios = np.stack((widths[:, :, 0] / widths[:, :, 1],
                           widths[:, :, 2] / widths[:, :, 1]), axis=-1).reshape(len(widths), 6)
        self.ratios = torch.as_tensor(ratios.astype(np.float32), device=self.device)
        self.cache = (GeometryCache.build(widths, self.device, precision="fp32",
                                          mode="template") if projection == "cached" else None)
        self.projection = projection

    @torch.inference_mode()
    def predict(self, phase: np.ndarray) -> np.ndarray:
        """Return weights with shape (N,27), ordered k outer, j middle, i inner.

        ``phase`` has shape (N,3), with each coordinate in [0,1). It is the
        particle position relative to the left edge of its center cell.
        """
        if self.ratios is None or self.projection is None:
            raise RuntimeError("Call prepare(cell_widths) before predict(phase)")
        phase = np.asarray(phase, dtype=np.float32)
        if phase.shape != (len(self.ratios), 3):
            raise ValueError(f"phase must have shape ({len(self.ratios)},3)")
        if not np.isfinite(phase).all() or not ((phase >= 0) & (phase < 1)).all():
            raise ValueError("phase coordinates must be finite and in [0,1)")
        phase_t = torch.as_tensor(phase, device=self.device)
        features = torch.cat((phase_t, self.ratios), dim=1)
        if self.projection == "dynamic":
            weights = self.model(features, projected=True)
        else:
            assert self.cache is not None
            weights = self.cache.apply(self.model.raw(features), phase_t,
                                       correction=True, output_fp32=True)
        return weights.cpu().numpy()
