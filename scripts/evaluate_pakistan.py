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

from floodcast.data import PakistanFloodDataset
from floodcast.models import SmallUNet


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pakistan flood model on test split")
    parser.add_argument("--config", type=Path, default=Path("configs/pakistan_train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pakistan_baseline/best.pt"))
    parser.add_argument("--threshold-m", type=float, default=0.1, help="Flood-depth threshold in meters for IoU")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wd_min, wd_max = float(data_cfg["wd_clip"][0]), float(data_cfg["wd_clip"][1])
    wd_range = max(wd_max - wd_min, 1e-6)
    threshold_scaled = (float(args.threshold_m) - wd_min) / wd_range

    mse_sum = 0.0
    mae_sum = 0.0
    pix_count = 0
    inter_sum = 0
    union_sum = 0

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(x)

            pred = torch.clamp(pred, 0.0, 1.0)

            diff = (pred - y)
            mse_sum += float((diff * diff).sum().item())
            mae_sum += float(diff.abs().sum().item())
            pix_count += int(diff.numel())

            pred_bin = pred >= threshold_scaled
            y_bin = y >= threshold_scaled
            inter_sum += int((pred_bin & y_bin).sum().item())
            union_sum += int((pred_bin | y_bin).sum().item())

    rmse_scaled = math.sqrt(mse_sum / max(pix_count, 1))
    mae_scaled = mae_sum / max(pix_count, 1)

    rmse_m = rmse_scaled * wd_range
    mae_m = mae_scaled * wd_range

    iou = float(inter_sum / union_sum) if union_sum > 0 else 1.0

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "test_samples": len(ds_test),
        "threshold_m": float(args.threshold_m),
        "rmse_m": float(rmse_m),
        "mae_m": float(mae_m),
        "iou": float(iou),
    }

    metrics_path = out_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[TEST] samples=", len(ds_test))
    print("[TEST] RMSE(m)=", f"{rmse_m:.4f}")
    print("[TEST] MAE(m)=", f"{mae_m:.4f}")
    print("[TEST] IoU=", f"{iou:.4f}")
    print("[TEST] saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
