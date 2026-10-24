"""Small CPU/GPU smoke example; run: python example_inference.py --device cpu"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from compact_geometry import compact_constraint
from inference import RKPM9D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--projection", choices=("cached", "dynamic"), default="cached")
    args = parser.parse_args()

    # Two complete 27-point stencils: one uniform, one locally nonuniform.
    widths = np.ones((2, 3, 3), dtype=np.float64)
    widths[1] = [[0.8, 1.0, 1.25], [1.1, 1.0, 0.9], [0.7, 1.0, 1.4]]
    phase = np.array([[0.2, 0.5, 0.8], [0.35, 0.6, 0.25]], dtype=np.float32)

    predictor = RKPM9D(device=args.device)
    predictor.prepare(widths, projection=args.projection)
    weights = predictor.predict(phase)

    ratios = np.stack((widths[:, :, 0] / widths[:, :, 1],
                       widths[:, :, 2] / widths[:, :, 1]), axis=-1).reshape(2, 6)
    features = torch.as_tensor(np.concatenate((phase, ratios.astype(np.float32)), axis=1))
    A = compact_constraint(features).numpy()
    rhs = np.zeros((2, 10), dtype=np.float64)
    rhs[:, 0] = 1.0
    residual = np.einsum("nij,nj->ni", A.astype(np.float64), weights.astype(np.float64)) - rhs
    print("weights shape:", weights.shape)
    print("sum of weights:", weights.sum(axis=1))
    print("maximum absolute 10C residual:", np.abs(residual).max())


if __name__ == "__main__":
    main()
