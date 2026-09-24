#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    metadata: dict = {}
    for key in data.files:
        value = data[key]
        if value.shape == ():
            try:
                metadata[key] = value.item()
            except Exception:
                metadata[key] = value
        else:
            metadata[key] = value
    return metadata


def extract_date(path_or_name: str) -> str:
    match = DATE_RE.search(str(path_or_name))
    if match is None:
        raise ValueError(f"Cannot parse YYYY-MM-DD from: {path_or_name}")
    return match.group(1)


def resolve_feature_images(metadata: dict, n_times: int) -> tuple[list[str], list[Path]]:
    if "image_paths" not in metadata:
        raise KeyError("metadata.npz does not contain image_paths.")

    image_paths = [Path(str(value)).expanduser().resolve() for value in metadata["image_paths"]]

    if "feature_dates" in metadata:
        feature_dates = [str(value) for value in metadata["feature_dates"]]
    else:
        all_dates = [extract_date(str(path)) for path in image_paths]
        exclude_first = bool(metadata.get("exclude_first", True))
        feature_dates = all_dates[1:] if exclude_first else all_dates

    if len(feature_dates) != n_times:
        raise ValueError(
            f"Feature tensor has T={n_times}, but metadata contains "
            f"{len(feature_dates)} feature dates: {feature_dates}"
        )

    image_by_date: dict[str, Path] = {}
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        image_by_date[extract_date(path.name)] = path

    feature_images: list[Path] = []
    for date in feature_dates:
        if date not in image_by_date:
            raise ValueError(
                f"No input SAR path matching feature date {date}. "
                f"Available dates: {sorted(image_by_date)}"
            )
        feature_images.append(image_by_date[date])

    return feature_dates, feature_images


def load_sample_blocks(summary_path: Path, n_rows: int, n_times: int) -> dict[int, list[tuple[int, int]]]:
    """Reuse the exact row blocks used to train the saved KMeans."""
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    raw_blocks = summary.get("sample_blocks")
    if not raw_blocks:
        raise ValueError(
            f"{summary_path} has no sample_blocks. Re-run train_global_kmeans.py "
            "or provide a compatible training summary."
        )

    blocks_by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for block in raw_blocks:
        time_index = int(block["time_index"])
        start = int(block["start"])
        end = int(block["end"])

        if not (0 <= time_index < n_times):
            raise ValueError(f"Invalid time_index in sample block: {block}")
        if not (0 <= start < end <= n_rows):
            raise ValueError(f"Invalid row range in sample block: {block}")

        blocks_by_time[time_index].append((start, end))

    for time_index in range(n_times):
        if not blocks_by_time[time_index]:
            raise ValueError(f"No sample blocks found for time index {time_index}.")
        blocks_by_time[time_index].sort()

    return dict(blocks_by_time)


def strided_valid_values(
    image: np.ndarray,
    valid: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    """Take a spatially distributed sample without constructing huge index arrays."""
    h, w = image.shape
    if max_samples <= 0:
        stride = 1
    else:
        stride = max(1, int(math.ceil(math.sqrt((h * w) / max_samples))))

    sampled_image = image[::stride, ::stride]
    sampled_valid = valid[::stride, ::stride] & np.isfinite(sampled_image)
    values = np.asarray(sampled_image[sampled_valid], dtype=np.float32)

    if values.size == 0:
        raise ValueError("No valid sampled SAR values were found.")
    return values


def prepare_otsu_surface(
    image_path: Path,
    p_lower: float,
    p_upper: float,
    gaussian_sigma: float,
    max_threshold_samples: int,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Read one SAR image, robustly scale it to [0,1], optionally apply a NaN-aware
    Gaussian filter, and compute a dark-region Otsu threshold.

    Returns:
        surface: float32 [H,W], low values are dark/water-like
        valid: bool [H,W]
        threshold: Otsu threshold on the scaled/filtered surface
        raster_info: profile and diagnostic information
    """
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32, copy=False)
        valid = (src.read_masks(1) > 0) & np.isfinite(image)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            valid &= image != np.float32(nodata)
        profile = src.profile.copy()
        height, width = src.height, src.width

    raw_sample = strided_valid_values(image, valid, max_threshold_samples)
    low, high = np.percentile(raw_sample, [p_lower, p_upper]).astype(np.float64)
    del raw_sample

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(
            f"Invalid percentile range for {image_path}: low={low}, high={high}"
        )

    # In-place robust scaling. Invalid pixels are zero only during filtering.
    image[~valid] = 0.0
    image -= np.float32(low)
    image /= np.float32(high - low)
    np.clip(image, 0.0, 1.0, out=image)

    if gaussian_sigma > 0:
        weights = valid.astype(np.float32)
        surface = gaussian_filter(image, sigma=gaussian_sigma, mode="nearest")
        denominator = gaussian_filter(weights, sigma=gaussian_sigma, mode="nearest")

        good = denominator > 1e-6
        np.divide(surface, denominator, out=surface, where=good)
        surface[~good] = np.nan

        del denominator, weights, image
    else:
        surface = image
        surface[~valid] = np.nan

    threshold_values = strided_valid_values(
        surface,
        valid & np.isfinite(surface),
        max_threshold_samples,
    )

    if np.nanmax(threshold_values) <= np.nanmin(threshold_values):
        raise ValueError(f"Otsu is undefined for constant image values: {image_path}")

    threshold = float(threshold_otsu(threshold_values))
    del threshold_values

    info = {
        "profile": profile,
        "height": int(height),
        "width": int(width),
        "percentile_low_value": float(low),
        "percentile_high_value": float(high),
    }
    return surface.astype(np.float32, copy=False), valid, threshold, info


def save_otsu_mask_tif(
    surface: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    output_path: Path,
    profile: dict,
) -> None:
    output = np.full(surface.shape, 255, dtype=np.uint8)
    good = valid & np.isfinite(surface)
    output[good] = (surface[good] <= threshold).astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(output, 1)


def save_otsu_preview(
    surface: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str,
    max_plot_size: int,
) -> None:
    h, w = surface.shape
    stride = max(1, int(math.ceil(max(h, w) / max_plot_size)))

    sampled = surface[::stride, ::stride]
    sampled_valid = valid[::stride, ::stride] & np.isfinite(sampled)
    mask = np.zeros(sampled.shape, dtype=np.uint8)
    mask[sampled_valid] = (sampled[sampled_valid] <= threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=140)
    axes[0].imshow(sampled, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Scaled Gaussian SAR")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Otsu dark mask\nthreshold={threshold:.6f}")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean[None, :]) / std[None, :]

def evaluate_cluster_subsets(
    per_date_rows,
    n_clusters,
    max_selected_clusters=None,
):
    rows_by_date = {}

    for row in per_date_rows:
        time_index = int(row["time_index"])
        cluster_id = int(row["cluster"])

        rows_by_date.setdefault(time_index, {})[
            cluster_id
        ] = row

    if max_selected_clusters is None:
        max_selected_clusters = n_clusters

    max_selected_clusters = min(
        max_selected_clusters,
        n_clusters,
    )

    subset_rows = []

    for subset_size in range(
        1,
        max_selected_clusters + 1,
    ):
        for cluster_ids in combinations(
            range(n_clusters),
            subset_size,
        ):
            date_ious = []
            date_precisions = []
            date_coverages = []

            pooled_intersection = 0
            pooled_cluster_pixels = 0
            pooled_otsu_pixels = 0

            for time_index in sorted(rows_by_date):
                cluster_rows = rows_by_date[time_index]

                selected_rows = [
                    cluster_rows[cluster_id]
                    for cluster_id in cluster_ids
                ]

                intersection = sum(
                    int(row["intersection"])
                    for row in selected_rows
                )

                cluster_pixels = sum(
                    int(row["cluster_pixels"])
                    for row in selected_rows
                )

                otsu_pixels = int(
                    selected_rows[0]["otsu_pixels"]
                )

                union = (
                    cluster_pixels
                    + otsu_pixels
                    - intersection
                )

                iou = (
                    intersection / union
                    if union > 0
                    else 0.0
                )

                precision = (
                    intersection / cluster_pixels
                    if cluster_pixels > 0
                    else 0.0
                )

                coverage = (
                    intersection / otsu_pixels
                    if otsu_pixels > 0
                    else 0.0
                )

                date_ious.append(iou)
                date_precisions.append(precision)
                date_coverages.append(coverage)

                pooled_intersection += intersection
                pooled_cluster_pixels += cluster_pixels
                pooled_otsu_pixels += otsu_pixels

            pooled_union = (
                pooled_cluster_pixels
                + pooled_otsu_pixels
                - pooled_intersection
            )

            pooled_iou = (
                pooled_intersection / pooled_union
                if pooled_union > 0
                else 0.0
            )

            pooled_precision = (
                pooled_intersection
                / pooled_cluster_pixels
                if pooled_cluster_pixels > 0
                else 0.0
            )

            pooled_coverage = (
                pooled_intersection
                / pooled_otsu_pixels
                if pooled_otsu_pixels > 0
                else 0.0
            )

            subset_rows.append({
                "clusters": list(cluster_ids),
                "num_clusters": len(cluster_ids),
                "mean_date_iou": float(
                    np.mean(date_ious)
                ),
                "median_date_iou": float(
                    np.median(date_ious)
                ),
                "std_date_iou": float(
                    np.std(date_ious)
                ),
                "min_date_iou": float(
                    np.min(date_ious)
                ),
                "pooled_iou": float(pooled_iou),
                "pooled_cluster_overlap_ratio": float(
                    pooled_precision
                ),
                "pooled_otsu_coverage_ratio": float(
                    pooled_coverage
                ),
                "mean_date_cluster_overlap_ratio": float(
                    np.mean(date_precisions)
                ),
                "mean_date_otsu_coverage_ratio": float(
                    np.mean(date_coverages)
                ),
                "pooled_intersection": int(
                    pooled_intersection
                ),
                "pooled_cluster_pixels": int(
                    pooled_cluster_pixels
                ),
                "pooled_otsu_pixels": int(
                    pooled_otsu_pixels
                ),
                "num_dates": len(date_ious),
            })

    return subset_rows

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use per-date training Otsu masks to identify the water cluster in "
            "a saved global KMeans model."
        )
    )
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--kmeans-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--p-lower", type=float, default=1.0)
    parser.add_argument("--p-upper", type=float, default=99.0)
    parser.add_argument(
        "--otsu-sample-size",
        type=int,
        default=5_000_000,
        help="Approximate maximum spatial samples used to estimate percentiles and Otsu threshold.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mean_date_iou", "pooled_iou"],
        default="mean_date_iou",
    )
    parser.add_argument("--save-otsu-tif", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-plot-size", type=int, default=2500)
    args = parser.parse_args()

    feature_dir = args.feature_dir.expanduser().resolve()
    kmeans_dir = args.kmeans_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = feature_dir / "features.npy"
    coords_path = feature_dir / "coords.npy"
    metadata_path = feature_dir / "metadata.npz"
    model_path = kmeans_dir / "global_kmeans.pkl"
    mean_path = kmeans_dir / "global_feature_mean.npy"
    std_path = kmeans_dir / "global_feature_std.npy"
    summary_path = kmeans_dir / "global_kmeans_training_summary.json"

    for path in [features_path, coords_path, metadata_path, model_path, mean_path, std_path, summary_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    features = np.load(features_path, mmap_mode="r")
    coords = np.load(coords_path, mmap_mode="r")
    metadata = load_metadata(metadata_path)

    if features.ndim != 3:
        raise ValueError(f"Expected features [N,T,C], got {features.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords [N,2], got {coords.shape}")
    if coords.shape[0] != features.shape[0]:
        raise ValueError(
            f"Feature rows {features.shape[0]} do not match coords rows {coords.shape[0]}"
        )

    n_rows, n_times, feature_dim = map(int, features.shape)
    feature_dates, feature_images = resolve_feature_images(metadata, n_times)
    blocks_by_time = load_sample_blocks(summary_path, n_rows, n_times)

    with model_path.open("rb") as file:
        kmeans = pickle.load(file)
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
        raise ValueError(
            f"Scaler shapes mean={mean.shape}, std={std.shape}, expected {(feature_dim,)}"
        )

    k = int(kmeans.n_clusters)
    if k < 2:
        raise ValueError("KMeans must contain at least two clusters.")

    metadata_p_lower = float(metadata.get("p_lower", 1.0))
    metadata_p_upper = float(metadata.get("p_upper", 99.0))
    p_lower = metadata_p_lower if args.p_lower is None else args.p_lower
    p_upper = metadata_p_upper if args.p_upper is None else args.p_upper

    if not (0.0 <= p_lower < p_upper <= 100.0):
        raise ValueError("Require 0 <= p-lower < p-upper <= 100.")
    if args.gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be non-negative.")

    print(f"Features: {features_path}")
    print(f"Feature shape: {features.shape}")
    print(f"KMeans: {model_path}")
    print(f"K: {k}")
    print(f"Feature dates: {feature_dates}")
    print(f"Percentile scaling: {p_lower}, {p_upper}")
    print(f"Gaussian sigma: {args.gaussian_sigma}")

    # totals[cluster] aggregates all sampled date intersections/unions.
    pooled_intersection = np.zeros(k, dtype=np.int64)
    pooled_union = np.zeros(k, dtype=np.int64)
    pooled_cluster_pixels = np.zeros(k, dtype=np.int64)
    pooled_otsu_pixels = np.zeros(k, dtype=np.int64)
    per_date_rows: list[dict] = []
    thresholds: list[dict] = []

    otsu_output_dir = output_dir / "otsu_masks"
    if args.save_otsu_tif or args.save_preview:
        otsu_output_dir.mkdir(parents=True, exist_ok=True)

    for time_index, (date, image_path) in enumerate(zip(feature_dates, feature_images)):
        print("=" * 100)
        print(f"[Date {time_index + 1}/{n_times}] {date}")
        print(f"SAR: {image_path}")

        surface, image_valid, threshold, raster_info = prepare_otsu_surface(
            image_path=image_path,
            p_lower=p_lower,
            p_upper=p_upper,
            gaussian_sigma=args.gaussian_sigma,
            max_threshold_samples=args.otsu_sample_size,
        )

        thresholds.append({
            "time_index": int(time_index),
            "date": date,
            "image_path": str(image_path),
            "otsu_threshold": float(threshold),
            "percentile_low_value": raster_info["percentile_low_value"],
            "percentile_high_value": raster_info["percentile_high_value"],
        })
        print(f"Otsu threshold: {threshold:.8f}")

        date_intersection = np.zeros(k, dtype=np.int64)
        date_union = np.zeros(k, dtype=np.int64)
        date_cluster_pixels = np.zeros(k, dtype=np.int64)
        date_otsu_pixels = np.zeros(k, dtype=np.int64)
        date_valid_count = 0

        blocks = blocks_by_time[time_index]
        for block_index, (start, end) in enumerate(blocks):
            x = np.asarray(features[start:end, time_index, :], dtype=np.float32)
            labels = np.asarray(kmeans.predict(standardize(x, mean, std)), dtype=np.int32)

            batch_coords = np.asarray(coords[start:end], dtype=np.int64)
            rows = batch_coords[:, 0]
            cols = batch_coords[:, 1]

            coord_valid = (
                (rows >= 0)
                & (rows < surface.shape[0])
                & (cols >= 0)
                & (cols < surface.shape[1])
            )

            surface_values = np.full(rows.shape[0], np.nan, dtype=np.float32)
            if coord_valid.any():
                surface_values[coord_valid] = surface[rows[coord_valid], cols[coord_valid]]

            valid = coord_valid & np.isfinite(surface_values)
            otsu_at_coords = valid & (surface_values <= threshold)
            date_valid_count += int(valid.sum())

            for cluster_id in range(k):
                cluster_mask = valid & (labels == cluster_id)
                intersection = int((cluster_mask & otsu_at_coords).sum())
                union = int((cluster_mask | otsu_at_coords).sum())
                cluster_count = int(cluster_mask.sum())
                otsu_count = int(otsu_at_coords.sum())

                date_intersection[cluster_id] += intersection
                date_union[cluster_id] += union
                date_cluster_pixels[cluster_id] += cluster_count
                date_otsu_pixels[cluster_id] += otsu_count

            if block_index == 0 or (block_index + 1) % 20 == 0 or block_index + 1 == len(blocks):
                print(
                    f"  block {block_index + 1}/{len(blocks)}, "
                    f"sampled rows processed={sum(e - s for s, e in blocks[:block_index + 1]):,}",
                    flush=True,
                )

        for cluster_id in range(k):
            iou = (
                date_intersection[cluster_id] / date_union[cluster_id]
                if date_union[cluster_id] > 0
                else 0.0
            )
            precision_like = (
                date_intersection[cluster_id] / date_cluster_pixels[cluster_id]
                if date_cluster_pixels[cluster_id] > 0
                else 0.0
            )
            recall_like = (
                date_intersection[cluster_id] / date_otsu_pixels[cluster_id]
                if date_otsu_pixels[cluster_id] > 0
                else 0.0
            )

            row = {
                "time_index": int(time_index),
                "date": date,
                "cluster": int(cluster_id),
                "iou_with_otsu": float(iou),
                "cluster_overlap_ratio": float(precision_like),
                "otsu_coverage_ratio": float(recall_like),
                "intersection": int(date_intersection[cluster_id]),
                "union": int(date_union[cluster_id]),
                "cluster_pixels": int(date_cluster_pixels[cluster_id]),
                "otsu_pixels": int(date_otsu_pixels[cluster_id]),
                "valid_sample_pixels": int(date_valid_count),
            }
            per_date_rows.append(row)
            print(
                f"  cluster {cluster_id}: IoU={iou:.6f}, "
                f"cluster-overlap={precision_like:.6f}, "
                f"Otsu-coverage={recall_like:.6f}"
            )

            pooled_intersection[cluster_id] += date_intersection[cluster_id]
            pooled_union[cluster_id] += date_union[cluster_id]
            pooled_cluster_pixels[cluster_id] += date_cluster_pixels[cluster_id]
            pooled_otsu_pixels[cluster_id] += date_otsu_pixels[cluster_id]

        if args.save_otsu_tif:
            save_otsu_mask_tif(
                surface=surface,
                valid=image_valid,
                threshold=threshold,
                output_path=otsu_output_dir / f"{date}_otsu_water_mask.tif",
                profile=raster_info["profile"],
            )

        if args.save_preview:
            save_otsu_preview(
                surface=surface,
                valid=image_valid,
                threshold=threshold,
                output_path=otsu_output_dir / f"{date}_otsu_preview.png",
                title=f"Training Otsu water-like mask: {date}",
                max_plot_size=args.max_plot_size,
            )

        del surface, image_valid
        gc.collect()

    aggregate_rows: list[dict] = []
    for cluster_id in range(k):
        date_ious = [
            row["iou_with_otsu"]
            for row in per_date_rows
            if row["cluster"] == cluster_id
        ]
        mean_date_iou = float(np.mean(date_ious)) if date_ious else 0.0
        median_date_iou = float(np.median(date_ious)) if date_ious else 0.0
        pooled_iou = (
            pooled_intersection[cluster_id] / pooled_union[cluster_id]
            if pooled_union[cluster_id] > 0
            else 0.0
        )
        aggregate_rows.append({
            "cluster": int(cluster_id),
            "mean_date_iou": mean_date_iou,
            "median_date_iou": median_date_iou,
            "pooled_iou": float(pooled_iou),
            "pooled_intersection": int(pooled_intersection[cluster_id]),
            "pooled_union": int(pooled_union[cluster_id]),
            "pooled_cluster_pixels": int(pooled_cluster_pixels[cluster_id]),
            "pooled_otsu_pixels": int(pooled_otsu_pixels[cluster_id]),
            "num_dates": int(len(date_ious)),
        })

    subset_rows = evaluate_cluster_subsets(
        per_date_rows=per_date_rows,
        n_clusters=k,
    )

    best_subset = max(
        subset_rows,
        key=lambda row: (
            row[args.selection_metric],
            -row["num_clusters"],
        ),
    )

    water_cluster_ids = [
        int(cluster_id)
        for cluster_id in best_subset["clusters"]
    ]

    print(
        "Selected water clusters: "
        f"{water_cluster_ids}"
    )

    print(
        f"Combined mean-date IoU: "
        f"{best_subset['mean_date_iou']:.6f}"
    )

    print(
        f"Combined pooled IoU: "
        f"{best_subset['pooled_iou']:.6f}"
    )

    print(
        f"Combined cluster-overlap: "
        f"{best_subset['pooled_cluster_overlap_ratio']:.6f}"
    )

    print(
        f"Combined Otsu-coverage: "
        f"{best_subset['pooled_otsu_coverage_ratio']:.6f}"
    )
    subset_csv = (
        output_dir
        / "cluster_subset_otsu_aggregate.csv"
    )

    subset_csv_rows = []

    for row in subset_rows:
        csv_row = dict(row)
        csv_row["clusters"] = ",".join(
            str(cluster_id)
            for cluster_id in row["clusters"]
        )
        subset_csv_rows.append(csv_row)

    with subset_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                subset_csv_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(subset_csv_rows)
    per_date_csv = output_dir / "cluster_otsu_per_date.csv"
    with per_date_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_date_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_date_rows)

    aggregate_csv = output_dir / "cluster_otsu_aggregate.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    threshold_csv = output_dir / "otsu_thresholds.csv"
    with threshold_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(thresholds[0].keys()))
        writer.writeheader()
        writer.writerows(thresholds)

    semantics = {
        "water_cluster_ids": water_cluster_ids,
        "selected_subset_stats": best_subset,
        "cluster_subset_stats_file": str(subset_csv),
        "selection_metric": args.selection_metric,
        "method": (
            "Saved global KMeans predictions were compared with per-date "
            "Gaussian-Otsu dark masks on the same sampled training feature blocks."
        ),
        "feature_dir": str(feature_dir),
        "kmeans_dir": str(kmeans_dir),
        "feature_shape": [n_rows, n_times, feature_dim],
        "feature_dates": feature_dates,
        "feature_images": [str(path) for path in feature_images],
        "k": k,
        "gaussian_sigma": float(args.gaussian_sigma),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "otsu_sample_size": int(args.otsu_sample_size),
        "aggregate_cluster_stats": aggregate_rows,
        "per_date_cluster_stats_file": str(per_date_csv),
        "aggregate_cluster_stats_file": str(aggregate_csv),
        "otsu_thresholds_file": str(threshold_csv),
        "note": (
            "KMeans centers were not refitted. Otsu was used "
            "to assign water semantics to a subset of saved "
            "global clusters. Test features must use the same "
            "saved feature mean/std and KMeans model."
        ),
    }

    semantics_path = output_dir / "water_cluster_semantics.json"
    with semantics_path.open("w", encoding="utf-8") as file:
        json.dump(semantics, file, indent=2)

    print("=" * 100)
    print("Aggregate cluster statistics:")
    for row in aggregate_rows:
        print(
            f"  cluster {row['cluster']}: "
            f"mean_date_iou={row['mean_date_iou']:.6f}, "
            f"pooled_iou={row['pooled_iou']:.6f}"
        )
    print(f"Selected water cluster: {water_cluster_ids}")
    print(f"Saved semantics: {semantics_path}")


if __name__ == "__main__":
    main()
