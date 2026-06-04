from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from floodcast.data import MultiRegionFloodDataset, PakistanFloodDataset
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

    regions_cfg = cfg.get("regions")
    if regions_cfg:
        required_keys = (
            "root",
            "cache_dir",
            "label_dir",
            "label_naming",
            "label_step_sec",
            "dem_clip",
            "rain_clip",
            "wd_clip",
        )
        merged_regions = []
        for i, region_cfg in enumerate(regions_cfg):
            merged_cfg = dict(data_cfg)
            merged_cfg.update(region_cfg)
            missing_keys = [k for k in required_keys if k not in merged_cfg]
            if missing_keys:
                raise KeyError(f"regions[{i}] missing keys: {missing_keys}")
            merged_cfg["seed"] = int(cfg["seed"])
            merged_regions.append(merged_cfg)

        ds_train = MultiRegionFloodDataset(
            region_configs=merged_regions,
            split="train",
            seed=int(cfg["seed"]),
        )
        ds_val = MultiRegionFloodDataset(
            region_configs=merged_regions,
            split="val",
            seed=int(cfg["seed"]),
        )
    else:
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


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_tensor(0.0)


def _slope_consistency_loss(pred: torch.Tensor, dem: torch.Tensor, slope_eps: float) -> torch.Tensor:
    # Penalize predicted depth gradients that move uphill with terrain slope direction.
    dem_dx = dem[:, :, :, 1:] - dem[:, :, :, :-1]
    dem_dy = dem[:, :, 1:, :] - dem[:, :, :-1, :]

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    valid_x = (dem_dx.abs() > slope_eps).float()
    valid_y = (dem_dy.abs() > slope_eps).float()

    viol_x = torch.relu(pred_dx * torch.sign(dem_dx))
    viol_y = torch.relu(pred_dy * torch.sign(dem_dy))

    loss_x = (viol_x * valid_x).sum() / valid_x.sum().clamp_min(1.0)
    loss_y = (viol_y * valid_y).sum() / valid_y.sum().clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def _wetdry_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold_scaled: float,
    temperature: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-4)
    logits = (pred - float(threshold_scaled)) / temp
    target_mask = (target >= float(threshold_scaled)).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, target_mask)


# -----------------------------------------------------------------------------
# Physics-informed residuals (discrete form used in training)
#
# Let h be scaled water depth, z be scaled DEM elevation, and eta = h + z.
# Spatial derivatives use first-order forward finite differences on unit grid.
#
# 1) Continuity residual with kinematic-wave approximation:
#    dh_dt(i,j) = (h_t(i,j) - h_{t-1}(i,j)) / dt
#    Sx, Sy     = dz/dx, dz/dy
#    S          = sqrt(Sx^2 + Sy^2 + eps)
#    q          = (1 / n) * h^(5/3) * sqrt(S)
#    qx         = -q * Sx / (sqrt(Sx^2 + Sy^2) + eps)
#    qy         = -q * Sy / (sqrt(Sx^2 + Sy^2) + eps)
#    div_q      = dqx/dx + dqy/dy
#    r_cont     = dh_dt + div_q - rain_proxy
#
# 2) Saint-Venant-inspired momentum residual (diffusive-wave simplification):
#    u = qx / (h + eps), v = qy / (h + eps)
#    Sf = n^2 * (u^2 + v^2) / (h^(4/3) + eps)
#    r_sv = g * |grad(eta)| - g * Sf
#
# 3) Manning roughness penalty:
#    L_manning = mean(n * |q|)
#
# rain_proxy = 0 because rainfall forcing is not in the one-step label increment
# target used by this loss term (the model receives rain as input channels already),
# so adding an external source term here would double-count forcing.
# -----------------------------------------------------------------------------
def _build_manning_map(land_use_scaled: torch.Tensor) -> torch.Tensor:
    lu = land_use_scaled
    n_map = torch.full_like(lu, 0.030)
    n_map = torch.where(lu < 0.15, lu.new_tensor(0.020), n_map)
    n_map = torch.where((lu >= 0.15) & (lu < 0.35), lu.new_tensor(0.030), n_map)
    n_map = torch.where((lu >= 0.35) & (lu < 0.55), lu.new_tensor(0.045), n_map)
    n_map = torch.where((lu >= 0.55) & (lu < 0.75), lu.new_tensor(0.060), n_map)
    n_map = torch.where(lu >= 0.75, lu.new_tensor(0.090), n_map)
    return n_map


def _continuity_sv_loss(
    pred: torch.Tensor,
    prev_wd: torch.Tensor,
    dem: torch.Tensor,
    n_map: torch.Tensor,
    dt: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt_safe = max(float(dt), 1e-6)
    eps_t = pred.new_tensor(float(eps))
    g = pred.new_tensor(9.81)

    h = pred.clamp_min(0.0)
    h_prev = prev_wd.clamp_min(0.0)
    z = dem
    eta = h + z

    dz_dx = z[:, :, :, 1:] - z[:, :, :, :-1]
    dz_dy = z[:, :, 1:, :] - z[:, :, :-1, :]

    h_x = 0.5 * (h[:, :, :, 1:] + h[:, :, :, :-1])
    h_y = 0.5 * (h[:, :, 1:, :] + h[:, :, :-1, :])
    n_x = 0.5 * (n_map[:, :, :, 1:] + n_map[:, :, :, :-1]).clamp_min(eps_t)
    n_y = 0.5 * (n_map[:, :, 1:, :] + n_map[:, :, :-1, :]).clamp_min(eps_t)

    slope_x = dz_dx.abs().clamp_min(eps_t)
    slope_y = dz_dy.abs().clamp_min(eps_t)
    q_x_mag = (h_x.clamp_min(0.0) ** (5.0 / 3.0)) * torch.sqrt(slope_x) / n_x
    q_y_mag = (h_y.clamp_min(0.0) ** (5.0 / 3.0)) * torch.sqrt(slope_y) / n_y

    dir_x = -torch.sign(dz_dx)
    dir_y = -torch.sign(dz_dy)
    q_x = q_x_mag * dir_x
    q_y = q_y_mag * dir_y

    dqx_dx = q_x[:, :, :, 1:] - q_x[:, :, :, :-1]
    dqy_dy = q_y[:, :, 1:, :] - q_y[:, :, :-1, :]
    div_q = dqx_dx[:, :, 1:-1, :] + dqy_dy[:, :, :, 1:-1]

    dh_dt = (h[:, :, 1:-1, 1:-1] - h_prev[:, :, 1:-1, 1:-1]) / dt_safe
    continuity_res = dh_dt + div_q

    deta_dx_c = eta[:, :, 1:-1, 1:-1] - eta[:, :, 1:-1, :-2]
    deta_dy_c = eta[:, :, 1:-1, 1:-1] - eta[:, :, :-2, 1:-1]
    grad_eta = torch.sqrt(deta_dx_c ** 2 + deta_dy_c ** 2 + eps_t)

    u_center = 0.5 * (q_x[:, :, 1:-1, :-1] + q_x[:, :, 1:-1, 1:]) / h[:, :, 1:-1, 1:-1].clamp_min(eps_t)
    v_center = 0.5 * (q_y[:, :, :-1, 1:-1] + q_y[:, :, 1:, 1:-1]) / h[:, :, 1:-1, 1:-1].clamp_min(eps_t)

    sf = (
        n_map[:, :, 1:-1, 1:-1].clamp_min(eps_t) ** 2
        * (u_center.pow(2) + v_center.pow(2))
        / h[:, :, 1:-1, 1:-1].clamp_min(eps_t).pow(4.0 / 3.0)
    )
    sv_momentum_res = g * grad_eta - g * sf

    sv_loss = continuity_res.abs().mean() + 0.5 * sv_momentum_res.abs().mean()

    q_mag_center = 0.5 * (
        0.5 * (q_x[:, :, 1:-1, :-1].abs() + q_x[:, :, 1:-1, 1:].abs())
        + 0.5 * (q_y[:, :, :-1, 1:-1].abs() + q_y[:, :, 1:, 1:-1].abs())
    )
    manning_penalty = (n_map[:, :, 1:-1, 1:-1] * q_mag_center).mean()
    return sv_loss, manning_penalty


def build_loss_config(cfg: dict) -> dict:
    train_cfg = cfg["train"]
    wd_min = float(cfg["data"]["wd_clip"][0])
    wd_max = float(cfg["data"]["wd_clip"][1])
    wd_range = max(wd_max - wd_min, 1e-6)

    wet_threshold_m = float(train_cfg.get("wet_threshold_m", 0.1))
    wet_threshold_scaled = float(np.clip((wet_threshold_m - wd_min) / wd_range, 0.0, 1.0))

    return {
        "lambda_huber": float(train_cfg.get("lambda_huber", 1.0)),
        "lambda_mass": float(train_cfg.get("lambda_mass", 0.0)),
        "lambda_slope": float(train_cfg.get("lambda_slope", 0.0)),
        "lambda_temporal": float(train_cfg.get("lambda_temporal", 0.0)),
        "lambda_wetdry": float(train_cfg.get("lambda_wetdry", 0.0)),
        "lambda_sv": float(train_cfg.get("lambda_sv", 0.02)),
        "lambda_manning": float(train_cfg.get("lambda_manning", 0.01)),
        "slope_epsilon": float(train_cfg.get("slope_epsilon", 1e-3)),
        "dt": float(train_cfg.get("dt", 300.0)),
        "wet_threshold_m": wet_threshold_m,
        "wet_threshold_scaled": wet_threshold_scaled,
        "wet_bce_temperature": float(train_cfg.get("wet_bce_temperature", 0.05)),
    }


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    wd_history: int,
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict]:
    huber = nn.functional.smooth_l1_loss(pred, target)
    mass = torch.mean(torch.abs(pred.mean(dim=(-1, -2)) - target.mean(dim=(-1, -2))))

    if loss_cfg["lambda_slope"] > 0:
        dem = x[:, 0:1, :, :]
        slope = _slope_consistency_loss(pred, dem, slope_eps=loss_cfg["slope_epsilon"])
    else:
        slope = _zero_like(pred)

    if loss_cfg["lambda_temporal"] > 0 and wd_history > 0:
        prev_wd = x[:, -1:, :, :]
        temporal = nn.functional.smooth_l1_loss(pred - prev_wd, target - prev_wd)
    else:
        temporal = _zero_like(pred)

    if loss_cfg["lambda_wetdry"] > 0:
        wetdry = _wetdry_bce_loss(
            pred=pred,
            target=target,
            threshold_scaled=loss_cfg["wet_threshold_scaled"],
            temperature=loss_cfg["wet_bce_temperature"],
        )
    else:
        wetdry = _zero_like(pred)

    if (loss_cfg["lambda_sv"] > 0 or loss_cfg["lambda_manning"] > 0) and wd_history > 0:
        dem = x[:, 0:1, :, :]
        land_use = x[:, 1:2, :, :]
        prev_wd = x[:, -1:, :, :]
        n_map = _build_manning_map(land_use)
        sv, manning = _continuity_sv_loss(
            pred=pred,
            prev_wd=prev_wd,
            dem=dem,
            n_map=n_map,
            dt=loss_cfg["dt"],
            eps=loss_cfg["slope_epsilon"],
        )
    else:
        sv = _zero_like(pred)
        manning = _zero_like(pred)

    loss = (
        loss_cfg["lambda_huber"] * huber
        + loss_cfg["lambda_mass"] * mass
        + loss_cfg["lambda_slope"] * slope
        + loss_cfg["lambda_temporal"] * temporal
        + loss_cfg["lambda_wetdry"] * wetdry
        + loss_cfg["lambda_sv"] * sv
        + loss_cfg["lambda_manning"] * manning
    )

    return loss, {
        "loss": float(loss.detach().item()),
        "huber": float(huber.detach().item()),
        "mass": float(mass.detach().item()),
        "slope": float(slope.detach().item()),
        "temporal": float(temporal.detach().item()),
        "wetdry": float(wetdry.detach().item()),
        "sv": float(sv.detach().item()),
        "manning": float(manning.detach().item()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    wd_history: int,
    loss_cfg: dict,
    log_first_n_steps: int = 0,
) -> tuple[dict, list[dict]]:
    model.train()
    totals = {
        "loss": 0.0,
        "huber": 0.0,
        "mass": 0.0,
        "slope": 0.0,
        "temporal": 0.0,
        "wetdry": 0.0,
        "sv": 0.0,
        "manning": 0.0,
    }
    n = 0
    first_steps: list[dict] = []

    for step_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            pred = model(x)
            loss, metrics = compute_loss(pred, y, x, wd_history=wd_history, loss_cfg=loss_cfg)

        for k, v in metrics.items():
            if not math.isfinite(float(v)):
                raise ValueError(f"Non-finite metric at step={step_idx}: {k}={v}")

        if log_first_n_steps > 0 and step_idx <= log_first_n_steps:
            first_steps.append({"step": int(step_idx), **{k: float(v) for k, v in metrics.items()}})

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        for k in totals:
            totals[k] += metrics[k] * batch_size
        n += batch_size

    return {k: v / max(n, 1) for k, v in totals.items()}, first_steps


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    wd_history: int,
    loss_cfg: dict,
) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "huber": 0.0,
        "mass": 0.0,
        "slope": 0.0,
        "temporal": 0.0,
        "wetdry": 0.0,
        "sv": 0.0,
        "manning": 0.0,
    }
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            pred = model(x)
            _, metrics = compute_loss(pred, y, x, wd_history=wd_history, loss_cfg=loss_cfg)

        batch_size = x.size(0)
        for k in totals:
            totals[k] += metrics[k] * batch_size
        n += batch_size

    return {k: v / max(n, 1) for k, v in totals.items()}


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
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    wd_history = int(cfg["data"].get("wd_history", 0))
    loss_cfg = build_loss_config(cfg)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "effective_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    history = []

    print(f"[INFO] device={device} amp={amp} in_channels={in_channels}")
    print(f"[INFO] train_batches={len(dl_train)} val_batches={len(dl_val)}")
    print(
        "[INFO] loss_weights="
        f"huber:{loss_cfg['lambda_huber']}, "
        f"mass:{loss_cfg['lambda_mass']}, "
        f"slope:{loss_cfg['lambda_slope']}, "
        f"temporal:{loss_cfg['lambda_temporal']}, "
        f"wetdry:{loss_cfg['lambda_wetdry']}, "
        f"sv:{loss_cfg['lambda_sv']}, "
        f"manning:{loss_cfg['lambda_manning']}"
    )

    log_first_n_steps = int(cfg["train"].get("log_first_n_steps", 0))
    if log_first_n_steps > 0:
        print(f"[INFO] log_first_n_steps={log_first_n_steps}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr, step_metrics = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
            wd_history=wd_history,
            loss_cfg=loss_cfg,
            log_first_n_steps=log_first_n_steps if epoch == 1 else 0,
        )

        for item in step_metrics:
            print(
                f"[S{item['step']:03d}] "
                f"loss={item['loss']:.5f} "
                f"huber={item['huber']:.5f} "
                f"mass={item['mass']:.5f} "
                f"slope={item['slope']:.5f} "
                f"temporal={item['temporal']:.5f} "
                f"wetdry={item['wetdry']:.5f} "
                f"sv={item['sv']:.5f} "
                f"manning={item['manning']:.5f}"
            )

        va = validate_one_epoch(
            model=model,
            loader=dl_val,
            device=device,
            amp=amp,
            wd_history=wd_history,
            loss_cfg=loss_cfg,
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
            f"train_mass={tr['mass']:.5f} val_mass={va['mass']:.5f} "
            f"train_slope={tr['slope']:.5f} val_slope={va['slope']:.5f} "
            f"train_temp={tr['temporal']:.5f} val_temp={va['temporal']:.5f} "
            f"train_wet={tr['wetdry']:.5f} val_wet={va['wetdry']:.5f} "
            f"train_sv={tr['sv']:.5f} val_sv={va['sv']:.5f} "
            f"train_manning={tr['manning']:.5f} val_manning={va['manning']:.5f} "
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
