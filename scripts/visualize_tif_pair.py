#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), np.nan, arr)
    return arr


def get_vrange(gt: np.ndarray, pred: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    valid = np.concatenate([
        gt[np.isfinite(gt)],
        pred[np.isfinite(pred)],
    ])
    if valid.size == 0:
        return 0.0, 1.0

    auto_min = float(np.percentile(valid, 2.0))
    auto_max = float(np.percentile(valid, 98.0))
    if auto_max <= auto_min:
        auto_max = auto_min + 1e-6

    out_min = auto_min if vmin is None else float(vmin)
    out_max = auto_max if vmax is None else float(vmax)
    return out_min, out_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one GT .tif and one predicted .tif in a single figure")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground-truth TIFF")
    parser.add_argument("--pred", type=Path, required=True, help="Path to inferred TIFF")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument("--title", type=str, default="Flood Depth: Ground Truth vs Inference")
    args = parser.parse_args()

    import matplotlib

    # Headless-safe backend for servers/SSH sessions.
    if args.output is not None or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    gt = read_tif(args.gt)
    pred = read_tif(args.pred)

    vmin, vmax = get_vrange(gt, pred, args.vmin, args.vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(args.title)

    im0 = axes[0].imshow(gt, cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred, cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Inference")
    axes[1].axis("off")

    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Water depth (m)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print("[DONE] saved figure:", args.output)
    else:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("No DISPLAY found. Use --output to save figure in headless mode.")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
