#!/usr/bin/env python3
"""Block 5 — Visual comparison: Ground Truth vs. Prediction for all 4 regions.

Generates a 6-panel figure per test sample:
  Row 1: GT depth (m)  |  Pred depth (m)  |  |Error| map (m)
  Row 2: GT flood mask |  Pred flood mask  |  Per-sample metrics

Usage (from project root, conda floodcast activated):
    python scripts/infer_compare_regions.py
    python scripts/infer_compare_regions.py --n-samples 8 --regions pakistan uk
    python scripts/infer_compare_regions.py --patch-size 256 --n-samples 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / SSH safe
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodcast.data import FloodDataset, PakistanFloodDataset
from floodcast.models import SmallUNet

# ---------------------------------------------------------------------------
# Region specs — best checkpoint from Block 5 summary per region
# ---------------------------------------------------------------------------
REGION_SPECS: dict[str, dict] = {
    "pakistan": {
        "config": ROOT / "configs/pakistan_train.yaml",
        "checkpoint": ROOT / "outputs/physics_loss/best.pt",
        "dataset_cls": "pakistan",  # PakistanFloodDataset
    },
    "australia": {
        "config": ROOT / "configs/australia_train.yaml",
        "checkpoint": ROOT / "outputs/multiregion/best.pt",
        "dataset_cls": "flood",  # FloodDataset
    },
    "mozambique": {
        "config": ROOT / "configs/mozambique_train.yaml",
        "checkpoint": ROOT / "outputs/multiregion/best.pt",
        "dataset_cls": "flood",
    },
    "uk": {
        "config": ROOT / "configs/uk_train.yaml",
        "checkpoint": ROOT / "outputs/multiregion/best.pt",
        "dataset_cls": "flood",
    },
}

FLOOD_THRESHOLD_M = 0.1
CMAP_DEPTH = "Blues"
CMAP_ERROR = "Reds"
CMAP_MASK = "Blues"
BG_DARK = "#12121e"
BG_PANEL = "#0d0d1a"
BG_TEXT = "#1e1e34"
FG_WHITE = "white"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def unscale(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return arr * max(vmax - vmin, 1e-6) + vmin


def sample_test_indices(total: int, n: int, seed: int = 42) -> list[int]:
    """Select n evenly spaced indices across [0, total)."""
    if n >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    # Evenly spaced across whole test range, plus small jitter to avoid always
    # picking the exact same time-step boundaries.
    base = np.linspace(0, total - 1, n, dtype=float)
    jitter = rng.uniform(-0.3, 0.3, size=n)
    indices = np.clip(np.round(base + jitter), 0, total - 1).astype(int).tolist()
    return sorted(set(indices))


def build_dataset(cfg: dict, dataset_cls: str, patch_size: int):
    dc = cfg["data"]
    kwargs = dict(
        root=Path(dc["root"]),
        cache_dir=Path(dc["cache_dir"]),
        label_dir=dc.get("label_dir", "flood_maps"),
        split="test",
        rain_history=dc["rain_history"],
        rain_stride_steps=dc["rain_stride_steps"],
        wd_history=dc["wd_history"],
        pred_horizon=dc["pred_horizon"],
        test_ratio=dc["test_ratio"],
        val_ratio_from_train=dc["val_ratio_from_train"],
        patch_size=patch_size,
        train_samples_per_epoch=dc["train_samples_per_epoch"],
        eval_samples=dc["val_samples"],
        dem_clip=dc["dem_clip"],
        rain_clip=dc["rain_clip"],
        wd_clip=dc["wd_clip"],
        seed=cfg["seed"],
    )
    if dataset_cls == "pakistan":
        return PakistanFloodDataset(**kwargs)
    return FloodDataset(**kwargs)


def per_sample_metrics(pred_m: np.ndarray, gt_m: np.ndarray, thr: float) -> dict:
    diff = pred_m - gt_m
    rmse = float(math.sqrt(float(np.mean(diff ** 2))))
    mae = float(np.mean(np.abs(diff)))
    mass_bias = float(pred_m.mean() - gt_m.mean())
    pred_bin = pred_m >= thr
    gt_bin = gt_m >= thr
    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    iou = float(inter / union) if union > 0 else 1.0
    return {"rmse_m": rmse, "mae_m": mae, "iou": iou, "mass_bias_m": mass_bias}


# ---------------------------------------------------------------------------
# Figure generator — dark theme, 6-panel
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes, bg: str = BG_PANEL) -> None:
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5a")
        spine.set_linewidth(0.8)


def _add_colorbar(fig: plt.Figure, im, ax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.9)
    cbar.set_label(label, color=FG_WHITE, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=FG_WHITE, labelcolor=FG_WHITE, labelsize=7)
    cbar.outline.set_edgecolor("#3a3a5a")


def make_comparison_figure(
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    region: str,
    sample_idx: int,
    metrics: dict,
    threshold_m: float,
    checkpoint_name: str,
) -> plt.Figure:

    err_abs = np.abs(pred_m - gt_m)
    gt_bin = (gt_m >= threshold_m).astype(np.float32)
    pred_bin = (pred_m >= threshold_m).astype(np.float32)

    vmax_depth = float(max(
        np.nanpercentile(np.concatenate([gt_m.ravel(), pred_m.ravel()]), 99),
        threshold_m * 2,
        0.05,
    ))
    vmax_err = float(max(np.nanpercentile(err_abs.ravel(), 97), 0.01))

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG_DARK)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.38, wspace=0.22,
        left=0.04, right=0.97, top=0.90, bottom=0.04,
    )

    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_err  = fig.add_subplot(gs[0, 2])
    ax_gtm  = fig.add_subplot(gs[1, 0])
    ax_pm   = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])

    title_kw = dict(color=FG_WHITE, fontsize=10, fontweight="bold", pad=5)
    aspect = "auto"

    # --- row 0: depth maps ---
    for ax in (ax_gt, ax_pred, ax_err, ax_gtm, ax_pm):
        _style_ax(ax)

    im_gt = ax_gt.imshow(gt_m, cmap=CMAP_DEPTH, vmin=0, vmax=vmax_depth, aspect=aspect)
    ax_gt.set_title("Ground Truth  (m)", **title_kw)
    ax_gt.axis("off")
    _add_colorbar(fig, im_gt, ax_gt, "depth (m)")

    im_pr = ax_pred.imshow(pred_m, cmap=CMAP_DEPTH, vmin=0, vmax=vmax_depth, aspect=aspect)
    ax_pred.set_title("Prediction  (m)", **title_kw)
    ax_pred.axis("off")
    _add_colorbar(fig, im_pr, ax_pred, "depth (m)")

    im_er = ax_err.imshow(err_abs, cmap=CMAP_ERROR, vmin=0, vmax=vmax_err, aspect=aspect)
    ax_err.set_title("|Error|  (m)", **title_kw)
    ax_err.axis("off")
    _add_colorbar(fig, im_er, ax_err, "|error| (m)")

    # --- row 1: binary flood masks ---
    ax_gtm.imshow(gt_bin, cmap=CMAP_MASK, vmin=0, vmax=1, aspect=aspect)
    ax_gtm.set_title(f"GT Flood Mask  (≥ {threshold_m} m)", **title_kw)
    ax_gtm.axis("off")

    ax_pm.imshow(pred_bin, cmap=CMAP_MASK, vmin=0, vmax=1, aspect=aspect)
    ax_pm.set_title(f"Pred Flood Mask  (≥ {threshold_m} m)", **title_kw)
    ax_pm.axis("off")

    # --- info panel ---
    _style_ax(ax_info, bg=BG_TEXT)
    ax_info.axis("off")
    wet_gt_pct = float(gt_bin.mean() * 100)
    wet_pr_pct = float(pred_bin.mean() * 100)
    shape_str = f"{gt_m.shape[0]} × {gt_m.shape[1]}"
    ckpt_short = checkpoint_name.replace("outputs/", "").replace("/best.pt", "")
    info = (
        f"Region    : {region.upper()}\n"
        f"Sample idx: #{sample_idx}\n"
        f"Frame size: {shape_str} px\n"
        f"Checkpoint: {ckpt_short}\n\n"
        f"{'─' * 28}\n\n"
        f"  RMSE     {metrics['rmse_m']:>8.4f} m\n"
        f"  MAE      {metrics['mae_m']:>8.4f} m\n"
        f"  IoU      {metrics['iou']:>8.4f}\n"
        f"  Bias     {metrics['mass_bias_m']:>+8.4f} m\n\n"
        f"{'─' * 28}\n\n"
        f"  Wet GT   {wet_gt_pct:>7.2f} %\n"
        f"  Wet Pred {wet_pr_pct:>7.2f} %"
    )
    ax_info.text(
        0.5, 0.5, info,
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=10.5, color=FG_WHITE, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#252545", edgecolor="#5a5a90", linewidth=1.2),
    )

    fig.suptitle(
        f"Flood Depth Comparison — {region.capitalize()}   |   Sample #{sample_idx}",
        color=FG_WHITE, fontsize=13, fontweight="bold",
    )

    return fig


# ---------------------------------------------------------------------------
# Simple 2-panel pair figure — GT vs Pred only (for report)
# ---------------------------------------------------------------------------

def make_pair_figure(
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    region: str,
    sample_idx: int,
    metrics: dict,
    dpi: int = 150,
) -> plt.Figure:
    """Clean 2-panel side-by-side: Ground Truth | Prediction."""
    vmax = float(max(
        np.nanpercentile(np.concatenate([gt_m.ravel(), pred_m.ravel()]), 99),
        FLOOD_THRESHOLD_M * 2,
        0.05,
    ))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)

    labels = ["Ground Truth (m)", "Prediction (m)"]
    for ax, data, label in zip(axes, [gt_m, pred_m], labels):
        _style_ax(ax)
        im = ax.imshow(data, cmap=CMAP_DEPTH, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(label, color=FG_WHITE, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.88)
        cbar.set_label("depth (m)", color=FG_WHITE, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=FG_WHITE, labelcolor=FG_WHITE, labelsize=8)
        cbar.outline.set_edgecolor("#3a3a5a")

    fig.suptitle(
        f"{region.capitalize()}  —  IoU {metrics['iou']:.3f}  |  RMSE {metrics['rmse_m']:.3f} m",
        color=FG_WHITE, fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Region mosaic — tile all samples in a single overview image
# ---------------------------------------------------------------------------

def make_mosaic(region: str, figure_paths: list[Path], output_path: Path) -> None:
    """Stack all per-sample figures vertically into one tall PNG for the appendix."""
    images = []
    for p in figure_paths:
        try:
            img = plt.imread(str(p))
            images.append(img)
        except Exception:  # noqa: BLE001
            continue

    if not images:
        return

    # Uniform width: pad to widest
    max_w = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        if im.shape[1] < max_w:
            pad = np.full((im.shape[0], max_w - im.shape[1], im.shape[2]), 0.07, dtype=im.dtype)
            im = np.concatenate([im, pad], axis=1)
        padded.append(im)

    mosaic = np.concatenate(padded, axis=0)

    fig_m, ax_m = plt.subplots(figsize=(mosaic.shape[1] / 100, mosaic.shape[0] / 100), dpi=100)
    fig_m.patch.set_facecolor(BG_DARK)
    ax_m.set_facecolor(BG_DARK)
    ax_m.imshow(mosaic)
    ax_m.axis("off")
    fig_m.tight_layout(pad=0)
    fig_m.savefig(output_path, dpi=100, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig_m)
    print(f"  [MOSAIC] {output_path.name}  ({mosaic.shape[0]}x{mosaic.shape[1]}px)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Block 5 — generate GT vs Pred comparison figures for 4 regions"
    )
    p.add_argument(
        "--regions", nargs="+", default=list(REGION_SPECS.keys()),
        choices=list(REGION_SPECS.keys()),
        help="Regions to visualize (default: all 4)",
    )
    p.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of test samples to visualize per region (default: 5)",
    )
    p.add_argument(
        "--patch-size", type=int, default=4096,
        help="Patch size passed to dataset — use 4096 (default) for full-frame inference,"
             " or a smaller value (e.g. 256) to match training crop",
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/comparison/figures",
        help="Directory to save PNG figures",
    )
    p.add_argument("--threshold-m", type=float, default=FLOOD_THRESHOLD_M)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-mosaic", action="store_true", help="Skip mosaic overview image")
    p.add_argument(
        "--pairs-only", action="store_true",
        help="Generate one clean 2-panel (GT | Pred) image per region and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"[INFER] device={device_name}  patch_size={args.patch_size}  n_samples={args.n_samples}")

    manifest: dict = {}
    pairs_dir = args.output_dir / "pairs"
    if args.pairs_only:
        pairs_dir.mkdir(parents=True, exist_ok=True)

    for region in args.regions:
        spec = REGION_SPECS[region]
        region_dir = args.output_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[REGION] {region.upper()}")
        print(f"{'='*60}")

        # ── Validate paths ──────────────────────────────────────────────
        config_path: Path = spec["config"]   # type: ignore[assignment]
        ckpt_path: Path   = spec["checkpoint"]  # type: ignore[assignment]

        if not config_path.exists():
            print(f"  [SKIP] config not found: {config_path}")
            manifest[region] = {"status": "missing_config", "path": str(config_path)}
            continue
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            manifest[region] = {"status": "missing_checkpoint", "path": str(ckpt_path)}
            continue

        cfg = load_config(config_path)
        dc = cfg["data"]
        wd_min = float(dc["wd_clip"][0])
        wd_max = float(dc["wd_clip"][1])

        # ── Dataset ─────────────────────────────────────────────────────
        try:
            ds = build_dataset(cfg, spec["dataset_cls"], args.patch_size)
        except FileNotFoundError as exc:
            print(f"  [SKIP] cache missing → {exc}")
            manifest[region] = {"status": "missing_cache", "reason": str(exc)}
            continue

        n_test = len(ds)
        print(f"  dataset  : {type(ds).__name__}  |  test samples: {n_test}")

        # ── Model ────────────────────────────────────────────────────────
        model = SmallUNet(
            in_channels=ds.input_channels,
            base_channels=int(cfg["model"]["base_channels"]),
        ).to(device)
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
        ckpt_name = str(ckpt_path.relative_to(ROOT))

        # ── Sample indices ───────────────────────────────────────────────
        n_req = 1 if args.pairs_only else args.n_samples
        indices = sample_test_indices(n_test, n_req)
        print(f"  checkpoint: {ckpt_name}")
        out_label = pairs_dir if args.pairs_only else region_dir
        print(f"  visualize : {len(indices)} sample(s) → {out_label}")

        figure_paths: list[Path] = []
        records: list[dict] = []

        for j, idx in enumerate(indices):
            x, y = ds[idx]
            x_t = x.unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    pred_t = model(x_t)

            pred_scaled = torch.clamp(pred_t[0, 0], 0.0, 1.0).cpu().numpy()
            gt_scaled   = torch.clamp(y[0], 0.0, 1.0).numpy()

            pred_m = unscale(pred_scaled, wd_min, wd_max)
            gt_m   = unscale(gt_scaled,   wd_min, wd_max)

            metrics = per_sample_metrics(pred_m, gt_m, args.threshold_m)

            if args.pairs_only:
                fig = make_pair_figure(gt_m, pred_m, region, idx, metrics, args.dpi)
                fig_path = pairs_dir / f"{region}.png"
            else:
                fig = make_comparison_figure(
                    gt_m, pred_m, region, idx, metrics, args.threshold_m, ckpt_name
                )
                fig_path = region_dir / f"sample_{j:03d}_idx{idx:04d}.png"

            fig.savefig(
                fig_path, dpi=args.dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            figure_paths.append(fig_path)

            records.append({
                "figure": str(fig_path.relative_to(ROOT)),
                "dataset_index": idx,
                "frame_shape": list(gt_m.shape),
                "metrics": metrics,
            })

            print(
                f"  [{j+1:02d}/{len(indices):02d}] idx={idx:04d} "
                f"RMSE={metrics['rmse_m']:.4f}m  MAE={metrics['mae_m']:.4f}m  "
                f"IoU={metrics['iou']:.4f}  → {fig_path.name}"
            )

        # ── Mosaic ───────────────────────────────────────────────────────
        if not args.no_mosaic and figure_paths:
            mosaic_path = region_dir / f"_mosaic_{region}.png"
            make_mosaic(region, figure_paths, mosaic_path)
        else:
            mosaic_path = None

        manifest[region] = {
            "status": "ok",
            "n_test": n_test,
            "samples_visualized": len(indices),
            "checkpoint": ckpt_name,
            "mosaic": str(mosaic_path.relative_to(ROOT)) if mosaic_path else None,
            "samples": records,
        }

    # ── Write manifest ──────────────────────────────────────────────────
    manifest_path = args.output_dir / "figures_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ok  = [r for r, v in manifest.items() if v.get("status") == "ok"]
    err = [r for r, v in manifest.items() if v.get("status") != "ok"]
    total_figs = sum(len(v.get("samples", [])) for v in manifest.values())

    print(f"\n{'='*60}")
    print(f"[DONE] regions OK : {ok}")
    if err:
        print(f"[WARN] regions SKIP: {err}")
    print(f"[DONE] total figures : {total_figs}")
    print(f"[DONE] manifest      : {manifest_path.relative_to(ROOT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
