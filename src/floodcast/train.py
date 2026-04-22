from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from floodcast.data import PakistanFloodDataset
from floodcast.models import SmallUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg["data"]

    ds_train = PakistanFloodDataset(
        root=Path(data_cfg["root"]),
        cache_dir=Path(data_cfg["cache_dir"]),
        label_dir=data_cfg.get("label_dir", "flood_maps"),
        split="train",
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

    ds_val = PakistanFloodDataset(
        root=Path(data_cfg["root"]),
        cache_dir=Path(data_cfg["cache_dir"]),
        label_dir=data_cfg.get("label_dir", "flood_maps"),
        split="val",
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

    num_workers = int(cfg["train"]["num_workers"])

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 1,
    )

    return dl_train, dl_val, ds_train.input_channels


def compute_loss(pred: torch.Tensor, target: torch.Tensor, lambda_mass: float) -> tuple[torch.Tensor, dict]:
    huber = nn.functional.smooth_l1_loss(pred, target)
    mass = torch.mean(torch.abs(pred.mean(dim=(-1, -2)) - target.mean(dim=(-1, -2))))
    loss = huber + lambda_mass * mass
    return loss, {"huber": float(huber.detach().item()), "mass": float(mass.detach().item())}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    lambda_mass: float,
) -> dict:
    model.train()
    total_loss = 0.0
    total_huber = 0.0
    total_mass = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(x)
            loss, metrics = compute_loss(pred, y, lambda_mass=lambda_mass)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_huber += metrics["huber"] * batch_size
        total_mass += metrics["mass"] * batch_size
        n += batch_size

    return {
        "loss": total_loss / max(n, 1),
        "huber": total_huber / max(n, 1),
        "mass": total_mass / max(n, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    lambda_mass: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_huber = 0.0
    total_mass = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(x)
            loss, metrics = compute_loss(pred, y, lambda_mass=lambda_mass)

        batch_size = x.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_huber += metrics["huber"] * batch_size
        total_mass += metrics["mass"] * batch_size
        n += batch_size

    return {
        "loss": total_loss / max(n, 1),
        "huber": total_huber / max(n, 1),
        "mass": total_mass / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pakistan flood baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/pakistan_train.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device_name = cfg["train"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    dl_train, dl_val, in_channels = build_dataloaders(cfg)

    model = SmallUNet(
        in_channels=in_channels,
        base_channels=int(cfg["model"]["base_channels"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    lambda_mass = float(cfg["train"].get("lambda_mass", 0.0))

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "effective_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    history = []

    print(f"[INFO] device={device} amp={amp} in_channels={in_channels}")
    print(f"[INFO] train_batches={len(dl_train)} val_batches={len(dl_val)}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
            lambda_mass=lambda_mass,
        )

        va = validate_one_epoch(
            model=model,
            loader=dl_val,
            device=device,
            amp=amp,
            lambda_mass=lambda_mass,
        )

        scheduler.step()

        epoch_time = time.time() - t0
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": tr,
            "val": va,
            "time_sec": epoch_time,
        }
        history.append(record)

        print(
            f"[E{epoch:03d}] "
            f"train_loss={tr['loss']:.5f} val_loss={va['loss']:.5f} "
            f"train_huber={tr['huber']:.5f} val_huber={va['huber']:.5f} "
            f"time={epoch_time:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "val_loss": va["loss"],
        }

        torch.save(ckpt, out_dir / "latest.pt")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[INFO] saved new best checkpoint with val_loss={best_val:.5f}")

        # if epoch % int(cfg["output"].get("save_every", 1)) == 0:
        #     torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("[DONE] training finished")


if __name__ == "__main__":
    # Keeps CPU thread usage stable on shared machines.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
