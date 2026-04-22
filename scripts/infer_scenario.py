#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodcast.models import SmallUNet


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clip_scale(x: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    x = np.clip(x, min_v, max_v)
    denom = max(max_v - min_v, 1e-6)
    return (x - min_v) / denom


def unscale(x: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    return x * max(max_v - min_v, 1e-6) + min_v


def read_resize(path: Path, target_hw: tuple[int, int], resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1, out_shape=target_hw, resampling=resampling).astype(np.float32)
        nodata = ds.nodata
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), 0.0, arr)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoregressive scenario inference and export .tif flood maps")
    parser.add_argument("--config", type=Path, default=Path("configs/pakistan_train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pakistan_baseline/best.pt"))
    parser.add_argument("--rainfall-dir", type=Path, default=Path("data/pakistan/rainfall"))
    parser.add_argument("--dem-path", type=Path, default=Path("data/pakistan/DEM/Pakistan_DEM.tif"))
    parser.add_argument("--landuse-path", type=Path, default=Path("data/pakistan/land_use/Pakistan.tif"))
    parser.add_argument("--initial-path", type=Path, default=Path("data/pakistan/initial_conditions/Pakistan_480m.tif"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scenario_infer"))
    parser.add_argument("--steps", type=int, default=-1, help="Number of 5-min steps to rollout; -1 means full rainfall horizon")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    rain_history = int(data_cfg["rain_history"])
    rain_stride_steps = int(data_cfg["rain_stride_steps"])
    wd_history = int(data_cfg["wd_history"])

    dem_clip = tuple(data_cfg["dem_clip"])
    rain_clip = tuple(data_cfg["rain_clip"])
    wd_clip = tuple(data_cfg["wd_clip"])

    with rasterio.open(args.initial_path) as init_ds:
        target_h, target_w = init_ds.height, init_ds.width
        profile = init_ds.profile.copy()

    target_hw = (target_h, target_w)

    dem = read_resize(args.dem_path, target_hw, Resampling.bilinear)
    land = read_resize(args.landuse_path, target_hw, Resampling.nearest)
    init_depth = read_resize(args.initial_path, target_hw, Resampling.bilinear)

    dem_s = clip_scale(dem, float(dem_clip[0]), float(dem_clip[1]))
    land_s = land / max(float(np.nanmax(land)), 1.0)
    init_s = clip_scale(init_depth, float(wd_clip[0]), float(wd_clip[1]))

    rain_files = sorted(args.rainfall_dir.glob("*.tif"))
    if not rain_files:
        raise FileNotFoundError(f"No rainfall files found in {args.rainfall_dir}")

    rain_s = []
    for i, rf in enumerate(rain_files):
        arr = read_resize(rf, target_hw, Resampling.bilinear)
        arr = clip_scale(arr, float(rain_clip[0]), float(rain_clip[1]))
        rain_s.append(arr)
        if (i + 1) % 50 == 0 or i == len(rain_files) - 1:
            print(f"[RAIN] loaded {i + 1}/{len(rain_files)}")

    total_steps_full = len(rain_s) * rain_stride_steps
    total_steps = total_steps_full if args.steps < 0 else min(int(args.steps), total_steps_full)

    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = SmallUNet(
        in_channels=2 + rain_history + wd_history,
        base_channels=int(cfg["model"]["base_channels"]),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
    )

    # Save initial condition as t=0 map.
    initial_out = args.output_dir / "0.tif"
    with rasterio.open(initial_out, "w", **profile) as dst:
        dst.write(init_depth.astype(np.float32), 1)

    history = [init_s.copy() for _ in range(wd_history)]

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    for t in range(total_steps):
        channels = [dem_s, land_s]

        for k in range(rain_history):
            idx = t - (rain_history - 1 - k) * rain_stride_steps
            rain_idx = min(max(idx // rain_stride_steps, 0), len(rain_s) - 1)
            channels.append(rain_s[rain_idx])

        channels.extend(history)

        x = np.stack(channels, axis=0)[None, ...].astype(np.float32)
        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(x_t).detach().cpu().numpy()[0, 0]

        pred = np.clip(pred, 0.0, 1.0)
        history = history[1:] + [pred]

        pred_m = unscale(pred, float(wd_clip[0]), float(wd_clip[1])).astype(np.float32)

        sec = (t + 1) * 300
        out_path = args.output_dir / f"{sec}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred_m, 1)

        if (t + 1) % 100 == 0 or t == total_steps - 1:
            print(f"[ROLL] saved {t + 1}/{total_steps} -> {out_path.name}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "rainfall_frames": len(rain_s),
        "rain_stride_steps": rain_stride_steps,
        "rollout_steps": total_steps,
        "output_dir": str(args.output_dir),
    }

    with (args.output_dir / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] scenario inference complete")
    print("[DONE] outputs at:", args.output_dir)


if __name__ == "__main__":
    main()
