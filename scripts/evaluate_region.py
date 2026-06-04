#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodcast.data import FloodDataset
from floodcast.models import SmallUNet


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate flood model on one region test split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    ds_test = FloodDataset(
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
        patch_size=data_cfg["patch_size"],
        train_samples_per_epoch=data_cfg["train_samples_per_epoch"],
        eval_samples=data_cfg["val_samples"],
        dem_clip=data_cfg["dem_clip"],
        rain_clip=data_cfg["rain_clip"],
        wd_clip=data_cfg["wd_clip"],
        seed=cfg["seed"],
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=max(1, int(train_cfg["num_workers"]) // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    model = SmallUNet(
        in_channels=ds_test.input_channels,
        base_channels=int(cfg["model"]["base_channels"]),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wd_min, wd_max = float(data_cfg["wd_clip"][0]), float(data_cfg["wd_clip"][1])
    wd_range = max(wd_max - wd_min, 1e-6)
    threshold_scaled = (float(args.threshold_m) - wd_min) / wd_range
    slope_eps = float(train_cfg.get("slope_epsilon", 0.002))
    wd_history = int(data_cfg.get("wd_history", 0))

    mse_sum = 0.0
    mae_sum = 0.0
    pix_count = 0
    inter_sum = 0
    union_sum = 0

    mass_abs_sum = 0.0
    mass_signed_sum = 0.0
    slope_valid_sum = 0
    slope_violation_sum = 0
    wet_pred_sum = 0
    wet_true_sum = 0
    delta_mae_sum = 0.0
    delta_pred_abs_sum = 0.0
    delta_true_abs_sum = 0.0

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred = model(x)

            pred = torch.clamp(pred, 0.0, 1.0)

            diff = pred - y
            mse_sum += float((diff * diff).sum().item())
            mae_sum += float(diff.abs().sum().item())
            pix_count += int(diff.numel())

            mass_diff = pred.mean(dim=(-1, -2)) - y.mean(dim=(-1, -2))
            mass_abs_sum += float(mass_diff.abs().sum().item())
            mass_signed_sum += float(mass_diff.sum().item())

            pred_bin = pred >= threshold_scaled
            y_bin = y >= threshold_scaled
            inter_sum += int((pred_bin & y_bin).sum().item())
            union_sum += int((pred_bin | y_bin).sum().item())
            wet_pred_sum += int(pred_bin.sum().item())
            wet_true_sum += int(y_bin.sum().item())

            dem = x[:, 0:1, :, :]
            dem_dx = dem[:, :, :, 1:] - dem[:, :, :, :-1]
            dem_dy = dem[:, :, 1:, :] - dem[:, :, :-1, :]

            pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

            valid_x = dem_dx.abs() > slope_eps
            valid_y = dem_dy.abs() > slope_eps
            viol_x = (pred_dx * torch.sign(dem_dx) > 0) & valid_x
            viol_y = (pred_dy * torch.sign(dem_dy) > 0) & valid_y

            slope_valid_sum += int(valid_x.sum().item() + valid_y.sum().item())
            slope_violation_sum += int(viol_x.sum().item() + viol_y.sum().item())

            if wd_history > 0:
                prev = x[:, -1:, :, :]
                pred_delta = (pred - prev).abs()
                true_delta = (y - prev).abs()
                delta_mae_sum += float((pred_delta - true_delta).abs().sum().item())
                delta_pred_abs_sum += float(pred_delta.sum().item())
                delta_true_abs_sum += float(true_delta.sum().item())

    rmse_scaled = math.sqrt(mse_sum / max(pix_count, 1))
    mae_scaled = mae_sum / max(pix_count, 1)

    rmse_m = rmse_scaled * wd_range
    mae_m = mae_scaled * wd_range

    iou = float(inter_sum / union_sum) if union_sum > 0 else 1.0
    mass_bias_abs_m = (mass_abs_sum / max(len(ds_test), 1)) * wd_range
    mass_bias_signed_m = (mass_signed_sum / max(len(ds_test), 1)) * wd_range
    slope_violation_rate = float(slope_violation_sum / max(slope_valid_sum, 1))
    wet_pred_ratio = float(wet_pred_sum / max(pix_count, 1))
    wet_true_ratio = float(wet_true_sum / max(pix_count, 1))

    if wd_history > 0:
        delta_mae_m = (delta_mae_sum / max(pix_count, 1)) * wd_range
        delta_pred_abs_m = (delta_pred_abs_sum / max(pix_count, 1)) * wd_range
        delta_true_abs_m = (delta_true_abs_sum / max(pix_count, 1)) * wd_range
    else:
        delta_mae_m = None
        delta_pred_abs_m = None
        delta_true_abs_m = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "test_samples": len(ds_test),
        "threshold_m": float(args.threshold_m),
        "rmse_m": float(rmse_m),
        "mae_m": float(mae_m),
        "iou": float(iou),
        "mass_bias_abs_m": float(mass_bias_abs_m),
        "mass_bias_signed_m": float(mass_bias_signed_m),
        "slope_violation_rate": float(slope_violation_rate),
        "wet_pred_ratio": float(wet_pred_ratio),
        "wet_true_ratio": float(wet_true_ratio),
        "delta_mae_m": None if delta_mae_m is None else float(delta_mae_m),
        "delta_pred_abs_m": None if delta_pred_abs_m is None else float(delta_pred_abs_m),
        "delta_true_abs_m": None if delta_true_abs_m is None else float(delta_true_abs_m),
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[TEST] samples=", len(ds_test))
    print("[TEST] RMSE(m)=", f"{rmse_m:.4f}")
    print("[TEST] MAE(m)=", f"{mae_m:.4f}")
    print("[TEST] IoU=", f"{iou:.4f}")
    print("[TEST] MassBiasAbs(m)=", f"{mass_bias_abs_m:.4f}")
    print("[TEST] SlopeViolationRate=", f"{slope_violation_rate:.4f}")
    print("[TEST] WetPredRatio=", f"{wet_pred_ratio:.4f}", "WetTrueRatio=", f"{wet_true_ratio:.4f}")
    if delta_mae_m is not None:
        print("[TEST] DeltaMAE(m)=", f"{delta_mae_m:.4f}")
    print("[TEST] saved metrics:", args.output)


if __name__ == "__main__":
    main()