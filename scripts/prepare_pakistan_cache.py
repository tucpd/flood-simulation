#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import yaml


def _read_resize_raster(path: Path, target_hw: tuple[int, int], resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(
            1,
            out_shape=target_hw,
            resampling=resampling,
        ).astype(np.float32)
        nodata = ds.nodata

    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), 0.0, arr)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rain_seconds_from_name(name: str) -> int:
    dt = datetime.strptime(name, "%Y%m%d-S%H%M%S")
    return int(dt.timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Pakistan cache for FloodCast baseline training")
    parser.add_argument("--config", type=Path, default=Path("configs/pakistan_train.yaml"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_cfg = cfg["data"]

    root = Path(data_cfg["root"])
    cache_dir = Path(data_cfg["cache_dir"])
    label_dir = str(data_cfg.get("label_dir", "flood_maps"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted((root / label_dir).glob("*.tif"), key=lambda p: int(p.stem))
    rain_files = sorted((root / "rainfall").glob("*.tif"))

    if not label_files:
        raise FileNotFoundError(f"No label files found in {root / label_dir}")
    if not rain_files:
        raise FileNotFoundError("No rainfall files found")

    with rasterio.open(label_files[0]) as ds:
        target_h, target_w = ds.height, ds.width

    target_cfg = tuple(data_cfg.get("target_shape", [target_h, target_w]))
    if target_cfg != (target_h, target_w):
        print(f"[WARN] target_shape in config {target_cfg} differs from label grid {(target_h, target_w)}")
        print("[WARN] Using label grid to keep labels and inputs aligned.")
    target_hw = (target_h, target_w)

    # 1) Static features
    dem = _read_resize_raster(root / "DEM" / "Pakistan_DEM.tif", target_hw, Resampling.bilinear)
    land_use = _read_resize_raster(root / "land_use" / "Pakistan.tif", target_hw, Resampling.nearest)
    np.savez_compressed(cache_dir / "static_features.npz", dem=dem, land_use=land_use)
    print(f"[OK] Saved static features: {cache_dir / 'static_features.npz'}")

    # 2) Rainfall cache at 30-min resolution, resized to water-depth grid
    rain_cache_path = cache_dir / "rainfall_30min.npy"
    rain_mm = np.lib.format.open_memmap(
        rain_cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rain_files), target_h, target_w),
    )

    for i, rf in enumerate(rain_files):
        rain_i = _read_resize_raster(rf, target_hw, Resampling.bilinear)
        rain_mm[i] = rain_i.astype(np.float16)
        if (i + 1) % 50 == 0 or i == len(rain_files) - 1:
            print(f"[RAIN] processed {i + 1}/{len(rain_files)}")

    del rain_mm
    print(f"[OK] Saved rainfall cache: {rain_cache_path}")

    # 3) Metadata for temporal mapping and reproducibility
    label_seconds = [int(p.stem) for p in label_files]
    rain_epoch_seconds = [_rain_seconds_from_name(p.stem) for p in rain_files]

    rain_step = int(np.median(np.diff(rain_epoch_seconds))) if len(rain_epoch_seconds) > 1 else 1800
    label_step = int(np.median(np.diff(label_seconds))) if len(label_seconds) > 1 else 300

    metadata = {
        "target_hw": [target_h, target_w],
        "label_count": len(label_files),
        "rain_count": len(rain_files),
        "label_step_sec": label_step,
        "rain_step_sec": rain_step,
        "label_start_sec": label_seconds[0],
        "label_end_sec": label_seconds[-1],
        "rain_start": rain_files[0].stem,
        "rain_end": rain_files[-1].stem,
        "label_dir": str((root / label_dir).as_posix()),
        "rain_cache_path": str(rain_cache_path.as_posix()),

        # Backward-compatible aliases for older code paths.
        "wd_count": len(label_files),
        "wd_step_sec": label_step,
        "wd_start_sec": label_seconds[0],
        "wd_end_sec": label_seconds[-1],
        "wd_dir": str((root / label_dir).as_posix()),
    }

    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved metadata: {cache_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
