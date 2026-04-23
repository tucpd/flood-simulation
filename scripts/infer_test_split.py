#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodcast.data import PakistanFloodDataset
from floodcast.models import SmallUNet


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unscale(x: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    return x * max(max_v - min_v, 1e-6) + min_v


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference only on test split and export .tif pairs")
    parser.add_argument("--config", type=Path, default=Path("configs/pakistan_train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pakistan_baseline/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_infer"))
    parser.add_argument("--start-index", type=int, default=0, help="Start index within test split")
    parser.add_argument("--max-samples", type=int, default=-1, help="How many test samples to export; -1 = all")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=-1,
        help="Patch size for test dataset. -1 means full-size frames (disable center crop).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    target_shape = data_cfg.get("target_shape", [1024, 1024])
    full_size_patch = int(max(int(target_shape[0]), int(target_shape[1])))
    patch_size = full_size_patch if int(args.patch_size) < 0 else int(args.patch_size)

    ds_test = PakistanFloodDataset(
        root=Path(data_cfg["root"]),
        cache_dir=Path(data_cfg["cache_dir"]),
        label_dir=data_cfg.get("label_dir", "flood_maps"),
        split="test",
        rain_history=data_cfg["rain_history"],
        rain_stride_steps=data_cfg["rain_stride_steps"],
        wd_history=data_cfg["wd_history"],
        pred_horizon=data_cfg["pred_horizon"],
        test_ratio=data_cfg["test_ratio"],
        val_ratio_from_train=data_cfg["val_ratio_from_train"],
        patch_size=patch_size,
        train_samples_per_epoch=data_cfg["train_samples_per_epoch"],
        eval_samples=data_cfg["val_samples"],
        dem_clip=data_cfg["dem_clip"],
        rain_clip=data_cfg["rain_clip"],
        wd_clip=data_cfg["wd_clip"],
        seed=cfg["seed"],
    )

    total_test = len(ds_test)
    start_idx = max(0, int(args.start_index))
    if start_idx >= total_test:
        raise ValueError(f"start-index {start_idx} is out of range for test set size {total_test}")

    if int(args.max_samples) < 0:
        end_idx = total_test
    else:
        end_idx = min(total_test, start_idx + int(args.max_samples))

    selected_indices = list(range(start_idx, end_idx))
    if not selected_indices:
        raise ValueError("No test samples selected")

    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = SmallUNet(
        in_channels=ds_test.input_channels,
        base_channels=int(cfg["model"]["base_channels"]),
    ).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wd_min, wd_max = float(data_cfg["wd_clip"][0]), float(data_cfg["wd_clip"][1])

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred"
    gt_dir = out_dir / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    pred_horizon = int(data_cfg["pred_horizon"])

    manifest = []
    for n, ds_idx in enumerate(selected_indices, start=1):
        x, y = ds_test[ds_idx]
        x_t = x.unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred = model(x_t)

        pred_scaled = torch.clamp(pred[0, 0], 0.0, 1.0).detach().cpu().numpy()
        gt_scaled = torch.clamp(y[0], 0.0, 1.0).detach().cpu().numpy()

        pred_m = unscale(pred_scaled, wd_min, wd_max).astype(np.float32)
        gt_m = unscale(gt_scaled, wd_min, wd_max).astype(np.float32)

        t_idx = int(ds_test.t_indices[ds_idx])
        target_idx = t_idx + pred_horizon
        target_path = ds_test.label_files[target_idx]

        with rasterio.open(target_path) as ref:
            profile = ref.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            compress="lzw",
            height=int(pred_m.shape[0]),
            width=int(pred_m.shape[1]),
        )

        file_id = f"test_{ds_idx:04d}_t_{int(target_path.stem):05d}"
        pred_path = pred_dir / f"{file_id}_pred.tif"
        gt_path = gt_dir / f"{file_id}_gt.tif"

        with rasterio.open(pred_path, "w", **profile) as dst:
            dst.write(pred_m, 1)
        with rasterio.open(gt_path, "w", **profile) as dst:
            dst.write(gt_m, 1)

        manifest.append(
            {
                "dataset_index": ds_idx,
                "time_index": t_idx,
                "target_label": str(target_path),
                "pred_tif": str(pred_path),
                "gt_tif": str(gt_path),
            }
        )

        if n % 20 == 0 or n == len(selected_indices):
            print(f"[INFER-TEST] exported {n}/{len(selected_indices)} samples")

    summary = {
        "checkpoint": str(args.checkpoint),
        "total_test_samples": total_test,
        "start_index": start_idx,
        "exported_samples": len(selected_indices),
        "patch_size": patch_size,
        "amp": amp,
        "output_dir": str(out_dir),
        "manifest": manifest,
    }

    summary_path = out_dir / "inference_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] test-split inference complete")
    print("[DONE] summary:", summary_path)


if __name__ == "__main__":
    main()
