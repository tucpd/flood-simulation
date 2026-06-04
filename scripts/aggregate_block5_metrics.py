#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENTS = ["non_physics", "physics_loss", "sv_loss", "multiregion"]
REQUIRED_REGIONS = ["pakistan", "australia", "mozambique", "uk"]

EXPERIMENT_ALIASES = {
    "non_physics": "non_physics",
    "physics_loss": "physics_loss",
    "sv_loss": "sv_loss",
    "multiregion": "multiregion",
}

SCHEMA_COLUMNS = [
    "experiment",
    "region",
    "checkpoint",
    "rmse_m",
    "mae_m",
    "iou",
    "mass_bias_abs_m",
    "mass_bias_signed_m",
    "slope_violation_rate",
    "wet_pred_ratio",
    "wet_true_ratio",
    "delta_mae_m",
    "has_nan",
    "has_inf",
    "source_path",
]

DELTA_COLUMNS = [
    "delta_iou_vs_non_physics",
    "delta_iou_vs_physics_loss",
    "delta_rmse_vs_non_physics",
    "delta_mae_vs_non_physics",
    "delta_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 5 metrics aggregation and comparison")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/comparison"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out


def normalize_experiment(raw_name: str) -> str | None:
    key = raw_name.strip().lower()
    return EXPERIMENT_ALIASES.get(key)


def infer_region_from_payload_or_path(path: Path, payload: dict[str, Any], fallback: str | None = None) -> str | None:
    stem = path.stem.lower()
    for region in REQUIRED_REGIONS:
        if f"_{region}" in stem or stem.endswith(region):
            return region

    cfg = str(payload.get("config", "")).lower()
    for region in REQUIRED_REGIONS:
        if region in cfg:
            return region

    return fallback


def finite_flags(values: list[float | None]) -> tuple[bool, bool]:
    has_nan = False
    has_inf = False
    for v in values:
        if v is None:
            continue
        if math.isnan(v):
            has_nan = True
        if math.isinf(v):
            has_inf = True
    return has_nan, has_inf


def base_row(experiment: str, region: str, source_path: str) -> dict[str, Any]:
    row = {k: None for k in SCHEMA_COLUMNS + DELTA_COLUMNS}
    row["experiment"] = experiment
    row["region"] = region
    row["source_path"] = source_path
    return row


def apply_metric_payload(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    row["checkpoint"] = payload.get("checkpoint")
    row["rmse_m"] = to_float(payload.get("rmse_m"))
    row["mae_m"] = to_float(payload.get("mae_m"))
    row["iou"] = to_float(payload.get("iou"))
    row["mass_bias_abs_m"] = to_float(payload.get("mass_bias_abs_m"))
    row["mass_bias_signed_m"] = to_float(payload.get("mass_bias_signed_m"))
    row["slope_violation_rate"] = to_float(payload.get("slope_violation_rate"))
    row["wet_pred_ratio"] = to_float(payload.get("wet_pred_ratio"))
    row["wet_true_ratio"] = to_float(payload.get("wet_true_ratio"))
    row["delta_mae_m"] = to_float(payload.get("delta_mae_m"))

    has_nan, has_inf = finite_flags([row["rmse_m"], row["mae_m"], row["iou"]])
    row["has_nan"] = bool(has_nan)
    row["has_inf"] = bool(has_inf)
    return row


def discover_artifacts(outputs_dir: Path) -> dict[str, list[Path]]:
    artifact_map: dict[str, list[Path]] = {
        "manifest": [],
        "history": [],
        "effective_config": [],
        "test_metrics": [],
        "region_metrics": [],
    }

    for path in outputs_dir.rglob("*.json"):
        name = path.name
        if "manifest" in name:
            artifact_map["manifest"].append(path)
        if name == "history.json":
            artifact_map["history"].append(path)
        if name == "effective_config.json":
            artifact_map["effective_config"].append(path)
        if name == "test_metrics.json":
            artifact_map["test_metrics"].append(path)
        if path.parent.name == "report" and name.startswith("metrics_"):
            artifact_map["region_metrics"].append(path)

    for k in artifact_map:
        artifact_map[k] = sorted(artifact_map[k])
    return artifact_map


def collect_rows(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    missing_or_error: list[dict[str, str]] = []
    skipped: list[str] = []
    stats = {
        "parsed_ok": 0,
        "parse_error": 0,
        "rows_valid": 0,
        "rows_missing_primary": 0,
        "rows_nan_or_inf": 0,
    }

    for path in sorted(outputs_dir.rglob("test_metrics.json")):
        rel = path.relative_to(outputs_dir)
        if len(rel.parts) < 2:
            skipped.append(str(path))
            continue

        exp_raw = rel.parts[0]
        exp = normalize_experiment(exp_raw)
        if exp is None:
            skipped.append(str(path))
            continue

        region = "pakistan"
        row = base_row(exp, region, str(path))
        try:
            payload = read_json(path)
            row = apply_metric_payload(row, payload)
            rows.append(row)
            stats["parsed_ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["parse_error"] += 1
            missing_or_error.append(
                {
                    "type": "parse_error",
                    "path": str(path),
                    "reason": f"Failed to parse test metrics: {exc}",
                }
            )

    for path in sorted(outputs_dir.rglob("metrics_*.json")):
        if path.parent.name != "report":
            continue

        rel = path.relative_to(outputs_dir)
        if len(rel.parts) < 3:
            skipped.append(str(path))
            continue

        exp = normalize_experiment(rel.parts[0])
        if exp is None:
            skipped.append(str(path))
            continue

        try:
            payload = read_json(path)
            default_region = "pakistan" if exp != "multiregion" else None
            region = infer_region_from_payload_or_path(path, payload, fallback=default_region)
            if region not in REQUIRED_REGIONS:
                skipped.append(str(path))
                continue

            row = base_row(exp, region, str(path))
            row = apply_metric_payload(row, payload)
            rows.append(row)
            stats["parsed_ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["parse_error"] += 1
            missing_or_error.append(
                {
                    "type": "parse_error",
                    "path": str(path),
                    "reason": f"Failed to parse multiregion report metrics: {exc}",
                }
            )

    for row in rows:
        primary = [row["rmse_m"], row["mae_m"], row["iou"]]
        if any(v is None for v in primary):
            stats["rows_missing_primary"] += 1
        elif row["has_nan"] or row["has_inf"]:
            stats["rows_nan_or_inf"] += 1
        else:
            stats["rows_valid"] += 1

    return rows, missing_or_error, skipped, stats


def add_missing_artifact_statuses(outputs_dir: Path, missing_or_error: list[dict[str, str]]) -> None:
    expected = {
        "non_physics": [
            outputs_dir / "non_physics" / "history.json",
            outputs_dir / "non_physics" / "effective_config.json",
            outputs_dir / "non_physics" / "test_metrics.json",
        ],
        "physics_loss": [
            outputs_dir / "physics_loss" / "history.json",
            outputs_dir / "physics_loss" / "effective_config.json",
            outputs_dir / "physics_loss" / "test_metrics.json",
        ],
        "sv_loss": [
            outputs_dir / "sv_loss" / "history.json",
            outputs_dir / "sv_loss" / "effective_config.json",
            outputs_dir / "sv_loss" / "test_metrics.json",
        ],
        "multiregion": [
            outputs_dir / "multiregion" / "history.json",
            outputs_dir / "multiregion" / "effective_config.json",
            outputs_dir / "multiregion" / "report" / "metrics_pakistan.json",
            outputs_dir / "multiregion" / "report" / "metrics_australia.json",
            outputs_dir / "multiregion" / "report" / "metrics_mozambique.json",
            outputs_dir / "multiregion" / "report" / "metrics_uk.json",
        ],
        "global": [outputs_dir / "block4_manifest.json"],
    }

    for _, paths in expected.items():
        for path in paths:
            if not path.exists():
                missing_or_error.append(
                    {
                        "type": "missing_artifact",
                        "path": str(path),
                        "reason": "Expected artifact is missing",
                    }
                )


def compute_deltas(rows: list[dict[str, Any]]) -> None:
    baseline_non_physics: dict[str, float] = {}
    baseline_physics_loss: dict[str, float] = {}

    for row in rows:
        if row["region"] != "pakistan":
            continue
        if row["experiment"] == "non_physics":
            baseline_non_physics = {
                "iou": row["iou"],
                "rmse_m": row["rmse_m"],
                "mae_m": row["mae_m"],
            }
        if row["experiment"] == "physics_loss":
            baseline_physics_loss = {
                "iou": row["iou"],
            }

    for row in rows:
        reason_parts: list[str] = []
        region = row["region"]

        if region == "pakistan" and baseline_non_physics.get("iou") is not None and row["iou"] is not None:
            row["delta_iou_vs_non_physics"] = row["iou"] - baseline_non_physics["iou"]
        else:
            row["delta_iou_vs_non_physics"] = None
            reason_parts.append("missing pakistan baseline non_physics iou")

        if region == "pakistan" and baseline_physics_loss.get("iou") is not None and row["iou"] is not None:
            row["delta_iou_vs_physics_loss"] = row["iou"] - baseline_physics_loss["iou"]
        else:
            row["delta_iou_vs_physics_loss"] = None
            reason_parts.append("missing pakistan baseline physics_loss iou")

        if region == "pakistan" and baseline_non_physics.get("rmse_m") is not None and row["rmse_m"] is not None:
            row["delta_rmse_vs_non_physics"] = row["rmse_m"] - baseline_non_physics["rmse_m"]
        else:
            row["delta_rmse_vs_non_physics"] = None
            reason_parts.append("missing pakistan baseline non_physics rmse")

        if region == "pakistan" and baseline_non_physics.get("mae_m") is not None and row["mae_m"] is not None:
            row["delta_mae_vs_non_physics"] = row["mae_m"] - baseline_non_physics["mae_m"]
        else:
            row["delta_mae_vs_non_physics"] = None
            reason_parts.append("missing pakistan baseline non_physics mae")

        row["delta_reason"] = None if row["region"] == "pakistan" else "non-pakistan region has no baseline pair"
        if row["region"] == "pakistan" and reason_parts:
            if not any(row[col] is not None for col in DELTA_COLUMNS[:-1]):
                row["delta_reason"] = "; ".join(reason_parts)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("iou") is not None and r.get("rmse_m") is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: (-float(r["iou"]), float(r["rmse_m"])))[0]


def build_wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["experiment"])][str(row["region"])].append(row)

    wide_rows: list[dict[str, Any]] = []
    for experiment in REQUIRED_EXPERIMENTS:
        out: dict[str, Any] = {"experiment": experiment}
        for region in REQUIRED_REGIONS:
            best = choose_best_row(grouped[experiment].get(region, []))
            out[f"{region}_rmse_m"] = None if best is None else best.get("rmse_m")
            out[f"{region}_mae_m"] = None if best is None else best.get("mae_m")
            out[f"{region}_iou"] = None if best is None else best.get("iou")
        wide_rows.append(out)
    return wide_rows


def rank_pakistan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pakistan_rows = [
        r
        for r in rows
        if r.get("region") == "pakistan" and r.get("iou") is not None and r.get("rmse_m") is not None
    ]

    # Keep one representative row per experiment (best IoU, then lowest RMSE).
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pakistan_rows:
        by_experiment[str(row["experiment"])].append(row)

    deduped: list[dict[str, Any]] = []
    for experiment, exp_rows in by_experiment.items():
        best_row = sorted(exp_rows, key=lambda r: (-float(r["iou"]), float(r["rmse_m"])))[0]
        deduped.append(best_row)

    pakistan_rows = sorted(deduped, key=lambda r: (-float(r["iou"]), float(r["rmse_m"])))
    ranking: list[dict[str, Any]] = []
    for idx, r in enumerate(pakistan_rows, start=1):
        ranking.append(
            {
                "rank": idx,
                "experiment": r.get("experiment"),
                "iou": r.get("iou"),
                "rmse_m": r.get("rmse_m"),
                "mae_m": r.get("mae_m"),
                "source_path": r.get("source_path"),
            }
        )
    return ranking


def best_per_region(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_region[str(r["region"])].append(r)

    result: dict[str, dict[str, Any] | None] = {}
    for region in REQUIRED_REGIONS:
        best = choose_best_row(by_region.get(region, []))
        if best is None:
            result[region] = None
        else:
            result[region] = {
                "experiment": best.get("experiment"),
                "iou": best.get("iou"),
                "rmse_m": best.get("rmse_m"),
                "mae_m": best.get("mae_m"),
                "source_path": best.get("source_path"),
            }
    return result


def build_result_bullets(best_region: dict[str, dict[str, Any] | None], pakistan_rank: list[dict[str, Any]]) -> list[str]:
    bullets: list[str] = []
    for region in REQUIRED_REGIONS:
        best = best_region.get(region)
        if best is None:
            bullets.append(f"No valid metrics available for region {region}.")
            continue
        bullets.append(
            (
                f"Region {region}: best experiment is {best['experiment']} "
                f"(IoU={best['iou']:.4f}, RMSE={best['rmse_m']:.4f} m, MAE={best['mae_m']:.4f} m)."
            )
        )
    if pakistan_rank:
        top = pakistan_rank[0]
        bullets.append(
            (
                f"Pakistan ranking leader is {top['experiment']} "
                f"(IoU={top['iou']:.4f}, RMSE={top['rmse_m']:.4f} m)."
            )
        )
    return bullets


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir
    comparison_dir = args.comparison_dir
    comparison_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_artifacts(outputs_dir)
    rows, missing_or_error, skipped, stats = collect_rows(outputs_dir)
    add_missing_artifact_statuses(outputs_dir, missing_or_error)
    compute_deltas(rows)

    rows = sorted(rows, key=lambda r: (str(r["experiment"]), str(r["region"]), str(r.get("source_path", ""))))

    long_columns = SCHEMA_COLUMNS + DELTA_COLUMNS
    long_csv = comparison_dir / "metrics_long.csv"
    write_csv(long_csv, rows, long_columns)

    wide_rows = build_wide_rows(rows)
    wide_columns = ["experiment"]
    for region in REQUIRED_REGIONS:
        wide_columns.extend([f"{region}_rmse_m", f"{region}_mae_m", f"{region}_iou"])
    wide_csv = comparison_dir / "metrics_wide.csv"
    write_csv(wide_csv, wide_rows, wide_columns)

    comparison_table = {
        "schema_columns": long_columns,
        "expected_experiments": REQUIRED_EXPERIMENTS,
        "expected_regions": REQUIRED_REGIONS,
        "artifact_inventory": {k: [str(p) for p in v] for k, v in artifacts.items()},
        "rows": rows,
        "stats": stats,
        "skipped_paths": skipped,
        "missing_or_error": missing_or_error,
    }
    comparison_json = comparison_dir / "comparison_table.json"
    with comparison_json.open("w", encoding="utf-8") as f:
        json.dump(comparison_table, f, indent=2)

    best_region = best_per_region(rows)
    pakistan_ranking = rank_pakistan(rows)
    summary = {
        "best_experiment_by_region": best_region,
        "pakistan_experiment_ranking": pakistan_ranking,
        "results_bullets": build_result_bullets(best_region, pakistan_ranking),
        "missing_or_error": missing_or_error,
        "stats": {
            **stats,
            "artifact_counts": {k: len(v) for k, v in artifacts.items()},
            "rows_total": len(rows),
            "experiments_found": sorted({str(r["experiment"]) for r in rows}),
            "regions_found": sorted({str(r["region"]) for r in rows}),
        },
    }

    summary_json = comparison_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[BLOCK5] Completed metrics aggregation")
    print(f"[BLOCK5] rows_total={len(rows)} parsed_ok={stats['parsed_ok']} parse_error={stats['parse_error']}")
    print(
        "[BLOCK5] sanity rows_valid="
        f"{stats['rows_valid']} rows_missing_primary={stats['rows_missing_primary']} rows_nan_or_inf={stats['rows_nan_or_inf']}"
    )
    print(f"[BLOCK5] outputs: {long_csv}, {wide_csv}, {comparison_json}, {summary_json}")


if __name__ == "__main__":
    main()
