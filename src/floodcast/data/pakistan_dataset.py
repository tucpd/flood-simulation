from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def _clip_scale(x: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    x = np.clip(x, min_v, max_v)
    denom = max(max_v - min_v, 1e-6)
    return (x - min_v) / denom


def _read_wd(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


class PakistanFloodDataset(Dataset):
    """Temporal patch dataset for one-step flood depth prediction."""

    def __init__(
        self,
        root: Path,
        cache_dir: Path,
        label_dir: str,
        split: str,
        rain_history: int,
        rain_stride_steps: int,
        wd_history: int,
        pred_horizon: int,
        test_ratio: float,
        val_ratio_from_train: float,
        patch_size: int,
        train_samples_per_epoch: int,
        eval_samples: int,
        dem_clip: Sequence[float],
        rain_clip: Sequence[float],
        wd_clip: Sequence[float],
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.label_dir = str(label_dir)
        self.split = split
        self.rain_history = int(rain_history)
        self.rain_stride_steps = int(rain_stride_steps)
        self.wd_history = int(wd_history)
        self.pred_horizon = int(pred_horizon)
        self.patch_size = int(patch_size)
        self.train_samples_per_epoch = int(train_samples_per_epoch)
        self.eval_samples = int(eval_samples)

        self.dem_clip = (float(dem_clip[0]), float(dem_clip[1]))
        self.rain_clip = (float(rain_clip[0]), float(rain_clip[1]))
        self.wd_clip = (float(wd_clip[0]), float(wd_clip[1]))

        self.rng = random.Random(seed)

        metadata_path = self.cache_dir / "metadata.json"
        static_path = self.cache_dir / "static_features.npz"
        rain_cache_path = self.cache_dir / "rainfall_30min.npy"

        if not metadata_path.exists() or not static_path.exists() or not rain_cache_path.exists():
            raise FileNotFoundError(
                "Cache not found. Run scripts/prepare_pakistan_cache.py before training."
            )

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        static = np.load(static_path)
        self.dem = static["dem"].astype(np.float32)
        self.land_use = static["land_use"].astype(np.float32)

        self.dem_scaled = _clip_scale(self.dem, self.dem_clip[0], self.dem_clip[1])
        self.lu_scaled = self.land_use / max(float(np.nanmax(self.land_use)), 1.0)

        self.rain_30min = np.load(rain_cache_path, mmap_mode="r")
        self.rain_count = self.rain_30min.shape[0]

        self.label_files = sorted((self.root / self.label_dir).glob("*.tif"), key=lambda p: int(p.stem))
        expected_count = int(self.metadata.get("label_count", self.metadata.get("wd_count", -1)))
        if len(self.label_files) != expected_count:
            raise ValueError("Label files changed after cache preparation")

        self.h, self.w = self.dem.shape

        self.input_channels = 2 + self.rain_history + self.wd_history

        start_t = max((self.rain_history - 1) * self.rain_stride_steps, self.wd_history - 1)
        end_t = len(self.label_files) - self.pred_horizon
        valid_t = list(range(start_t, end_t))

        n = len(valid_t)
        n_train_full = int(n * (1.0 - float(test_ratio)))
        train_full = valid_t[:n_train_full]
        test_part = valid_t[n_train_full:]

        n_val = int(len(train_full) * float(val_ratio_from_train))
        if n_val <= 0 and len(train_full) > 1:
            n_val = 1

        n_train = max(len(train_full) - n_val, 1)
        train_part = train_full[:n_train]
        val_part = train_full[n_train:]

        if split == "train":
            self.t_indices = train_part
        elif split == "val":
            self.t_indices = val_part
        elif split == "test":
            self.t_indices = test_part
        else:
            raise ValueError(f"Unknown split: {split}")

        if not self.t_indices:
            raise ValueError(f"No time indices for split={split}")

    def __len__(self) -> int:
        if self.split == "train":
            return self.train_samples_per_epoch
        if self.split == "val":
            return min(self.eval_samples, len(self.t_indices))
        return len(self.t_indices)

    def _time_for_index(self, idx: int) -> int:
        if self.split == "train":
            return self.rng.choice(self.t_indices)
        # Deterministic sampling for validation/test.
        stride = max(len(self.t_indices) // max(self.__len__(), 1), 1)
        return self.t_indices[min(idx * stride, len(self.t_indices) - 1)]

    def _sample_patch(self, h: int, w: int) -> tuple[int, int]:
        if self.patch_size >= h or self.patch_size >= w:
            return 0, 0

        if self.split == "train":
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            return y0, x0

        # Center crop for eval
        y0 = (h - self.patch_size) // 2
        x0 = (w - self.patch_size) // 2
        return y0, x0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._time_for_index(idx)

        channels = []

        # Static channels
        channels.append(self.dem_scaled)
        channels.append(self.lu_scaled)

        # Rain history channels (mapped from WD timeline to rain timeline)
        for k in range(self.rain_history):
            wd_idx = t - (self.rain_history - 1 - k) * self.rain_stride_steps
            rain_idx = min(max(wd_idx // self.rain_stride_steps, 0), self.rain_count - 1)
            rain_k = self.rain_30min[rain_idx].astype(np.float32)
            rain_k = _clip_scale(rain_k, self.rain_clip[0], self.rain_clip[1])
            channels.append(rain_k)

        # Water-depth history channels
        for k in range(self.wd_history):
            wd_file_idx = t - (self.wd_history - 1 - k)
            wd_k = _read_wd(self.label_files[wd_file_idx])
            wd_k = _clip_scale(wd_k, self.wd_clip[0], self.wd_clip[1])
            channels.append(wd_k)

        target = _read_wd(self.label_files[t + self.pred_horizon])
        target = _clip_scale(target, self.wd_clip[0], self.wd_clip[1])

        x = np.stack(channels, axis=0)
        y = target[None, ...]

        y0, x0 = self._sample_patch(self.h, self.w)
        if self.patch_size < self.h and self.patch_size < self.w:
            x = x[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            y = y[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        return x_t, y_t
