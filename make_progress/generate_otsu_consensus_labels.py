#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse YYYY-MM-DD from filename: {path.name}")
    return match.group(1)


def collect_sar_paths(
    input_dir: Path | None,
    images: Iterable[Path] | None,
) -> list[Path]:
    if input_dir is not None:
        paths = [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ]
    else:
        paths = list(images or [])

    paths = [Path(path).expanduser().resolve() for path in paths]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    paths.sort(key=parse_date)

    if len(paths) < 2:
        raise ValueError(f"At least two SAR images are required, found {len(paths)}.")

    dates = [parse_date(path) for path in paths]
    if len(dates) != len(set(dates)):
        raise ValueError(f"Duplicate SAR dates were found: {dates}")

    return paths


def read_sar(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        masked = src.read(1, masked=True)
        image = np.asarray(masked.filled(np.nan), dtype=np.float32)
        valid = ~np.ma.getmaskarray(masked)
        valid &= np.isfinite(image)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    image[~valid] = np.nan

    info = {
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "height": height,
        "width": width,
    }
    return image, valid, info


def check_alignment(reference: dict, current: dict, path: Path) -> None:
    if current["height"] != reference["height"] or current["width"] != reference["width"]:
        raise ValueError(
            f"Raster shape mismatch for {path}: "
            f"{(current['height'], current['width'])} vs "
            f"{(reference['height'], reference['width'])}"
        )

    if current["transform"] != reference["transform"]:
        raise ValueError(f"Raster transform mismatch for {path}")

    if current["crs"] != reference["crs"]:
        raise ValueError(f"Raster CRS mismatch for {path}")


def percentile_clip(
    image: np.ndarray,
    valid: np.ndarray,
    p_lower: float,
    p_upper: float,
) -> tuple[np.ndarray, float, float]:
    values = image[valid]

    if values.size == 0:
        raise ValueError("No valid SAR pixels.")

    lower = float(np.nanpercentile(values, p_lower))
    upper = float(np.nanpercentile(values, p_upper))

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"Invalid clipping range: lower={lower}, upper={upper}")

    clipped = image.copy()
    clipped[valid] = np.clip(clipped[valid], lower, upper)
    clipped[~valid] = np.nan

    return clipped, lower, upper


def nan_gaussian_filter(
    image: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    valid = valid & np.isfinite(image)

    values = np.zeros(image.shape, dtype=np.float32)
    values[valid] = image[valid]

    weights = valid.astype(np.float32)

    numerator = gaussian_filter(values, sigma=sigma)
    denominator = gaussian_filter(weights, sigma=sigma)

    blurred = np.full(image.shape, np.nan, dtype=np.float32)
    good = valid & (denominator > 1e-6)
    blurred[good] = numerator[good] / denominator[good]

    return blurred


def make_dark_otsu_mask(
    image: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, float]:
    otsu_valid = valid & np.isfinite(image)
    values = image[otsu_valid]

    if values.size == 0:
        raise ValueError("No valid pixels for Otsu thresholding.")

    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))

    if value_max <= value_min:
        raise ValueError("Valid SAR values are constant; Otsu is undefined.")

    threshold = float(threshold_otsu(values))

    water = np.zeros(image.shape, dtype=bool)
    water[otsu_valid] = image[otsu_valid] <= threshold

    return water, threshold


def make_consensus_labels(
    raw_water: np.ndarray,
    gaussian_water: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    labels = np.full(raw_water.shape, -1, dtype=np.int8)

    agree_water = valid & raw_water & gaussian_water
    agree_background = valid & (~raw_water) & (~gaussian_water)

    labels[agree_background] = 0
    labels[agree_water] = 1

    return labels


def save_label_tif(
    labels: np.ndarray,
    output_path: Path,
    reference_profile: dict,
) -> None:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="int8",
        nodata=-1,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(labels.astype(np.int8, copy=False), 1)


def percentile_display(
    image: np.ndarray,
    valid: np.ndarray,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
) -> np.ndarray:
    output = np.full(image.shape, np.nan, dtype=np.float32)
    values = image[valid & np.isfinite(image)]

    if values.size == 0:
        return output

    low = np.nanpercentile(values, p_lower)
    high = np.nanpercentile(values, p_upper)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return output

    good = valid & np.isfinite(image)
    output[good] = np.clip((image[good] - low) / (high - low), 0.0, 1.0)
    return output


def save_preview(
    image: np.ndarray,
    valid: np.ndarray,
    raw_water: np.ndarray,
    gaussian_water: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    max_plot_size: int,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for --save-preview.")

    height, width = image.shape
    stride = max(1, int(np.ceil(max(height, width) / max_plot_size)))

    display = percentile_display(image, valid)[::stride, ::stride]
    raw_small = raw_water[::stride, ::stride]
    gaussian_small = gaussian_water[::stride, ::stride]
    labels_small = labels[::stride, ::stride]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=140)

    axes[0].imshow(display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"SAR\n{title}")

    axes[1].imshow(raw_small, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Raw Otsu water")

    axes[2].imshow(gaussian_small, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Gaussian Otsu water")

    image_for_plot = np.ma.masked_where(labels_small == -1, labels_small)
    axes[3].imshow(image_for_plot, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Consensus label\n-1 disagreement")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Otsu/Gaussian-Otsu consensus labels from a SAR sequence."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing the SAR sequence as GeoTIFF files.",
    )
    input_group.add_argument(
        "--images",
        nargs="+",
        type=Path,
        help="Explicit SAR GeoTIFF sequence.",
    )

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument(
        "--p-lower",
        type=float,
        default=1.0,
        help="Lower clipping percentile. Use 0 to disable lower clipping.",
    )
    parser.add_argument(
        "--p-upper",
        type=float,
        default=99.0,
        help="Upper clipping percentile. Use 100 to disable upper clipping.",
    )
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-plot-size", type=int, default=2500)

    args = parser.parse_args()

    if not (0.0 <= args.p_lower < args.p_upper <= 100.0):
        raise ValueError(
            "--p-lower and --p-upper must satisfy 0 <= p-lower < p-upper <= 100."
        )

    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else None
    )
    if input_dir is not None and not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_date_dir = output_dir / "per_date_labels"
    per_date_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir / "previews"
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    sar_paths = collect_sar_paths(input_dir=input_dir, images=args.images)
    dates = [parse_date(path) for path in sar_paths]

    print(f"Found {len(sar_paths)} SAR images:")
    for date, path in zip(dates, sar_paths):
        print(f"  {date}: {path}")

    _, _, first_info = read_sar(sar_paths[0])
    height = first_info["height"]
    width = first_info["width"]
    num_dates = len(sar_paths)

    labels_all_path = output_dir / "labels_all_dates.npy"
    labels_all = np.lib.format.open_memmap(
        labels_all_path,
        mode="w+",
        dtype=np.int8,
        shape=(num_dates, height, width),
    )

    records: list[dict] = []

    for time_index, (date, sar_path) in enumerate(zip(dates, sar_paths)):
        print(f"[{time_index + 1}/{num_dates}] Processing {date}", flush=True)

        image, valid, info = read_sar(sar_path)
        check_alignment(first_info, info, sar_path)

        clipped, clip_lower, clip_upper = percentile_clip(
            image=image,
            valid=valid,
            p_lower=args.p_lower,
            p_upper=args.p_upper,
        )

        raw_water, raw_threshold = make_dark_otsu_mask(
            image=clipped,
            valid=valid,
        )

        blurred = nan_gaussian_filter(
            image=clipped,
            valid=valid,
            sigma=args.gaussian_sigma,
        )

        gaussian_water, gaussian_threshold = make_dark_otsu_mask(
            image=blurred,
            valid=valid,
        )

        labels = make_consensus_labels(
            raw_water=raw_water,
            gaussian_water=gaussian_water,
            valid=valid,
        )

        labels_all[time_index] = labels

        save_label_tif(
            labels=labels,
            output_path=per_date_dir / f"{date}_consensus_label.tif",
            reference_profile=info["profile"],
        )

        if args.save_preview:
            save_preview(
                image=image,
                valid=valid,
                raw_water=raw_water,
                gaussian_water=gaussian_water,
                labels=labels,
                output_path=preview_dir / f"{date}_consensus_preview.png",
                title=date,
                max_plot_size=args.max_plot_size,
            )

        valid_count = int(valid.sum())
        water_count = int((labels == 1).sum())
        background_count = int((labels == 0).sum())
        uncertain_count = int((labels == -1).sum())

        records.append(
            {
                "date": date,
                "sar_path": str(sar_path),
                "clip_lower": clip_lower,
                "clip_upper": clip_upper,
                "raw_otsu_threshold": raw_threshold,
                "gaussian_otsu_threshold": gaussian_threshold,
                "valid_pixels": valid_count,
                "water_pixels": water_count,
                "background_pixels": background_count,
                "uncertain_or_invalid_pixels": uncertain_count,
                "water_fraction_of_valid": (
                    water_count / valid_count if valid_count > 0 else np.nan
                ),
                "background_fraction_of_valid": (
                    background_count / valid_count if valid_count > 0 else np.nan
                ),
            }
        )

        print(
            f"  raw threshold={raw_threshold:.6f}, "
            f"Gaussian threshold={gaussian_threshold:.6f}, "
            f"water={water_count:,}, "
            f"background={background_count:,}, "
            f"uncertain/invalid={uncertain_count:,}",
            flush=True,
        )

        del image, valid, clipped, raw_water, blurred, gaussian_water, labels

    labels_all.flush()

    labels_feature_path = output_dir / "labels_feature_dates.npy"
    labels_feature = np.lib.format.open_memmap(
        labels_feature_path,
        mode="w+",
        dtype=np.int8,
        shape=(num_dates - 1, height, width),
    )
    labels_feature[:] = labels_all[1:]
    labels_feature.flush()

    thresholds_path = output_dir / "otsu_thresholds.csv"
    with open(thresholds_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    np.savez(
        output_dir / "metadata.npz",
        sar_paths=np.asarray([str(path) for path in sar_paths]),
        dates=np.asarray(dates),
        feature_dates=np.asarray(dates[1:]),
        num_dates=np.asarray(num_dates, dtype=np.int64),
        height=np.asarray(height, dtype=np.int64),
        width=np.asarray(width, dtype=np.int64),
        gaussian_sigma=np.asarray(args.gaussian_sigma, dtype=np.float32),
        p_lower=np.asarray(args.p_lower, dtype=np.float32),
        p_upper=np.asarray(args.p_upper, dtype=np.float32),
        label_definition=np.asarray(
            "1=both water, 0=both background, -1=disagreement or invalid"
        ),
    )

    print("\nDone.")
    print(f"Saved all-date labels:     {labels_all_path}")
    print(f"Saved feature-date labels: {labels_feature_path}")
    print(f"Saved thresholds:          {thresholds_path}")
    print(f"labels_all_dates shape:     {(num_dates, height, width)}")
    print(f"labels_feature_dates shape: {(num_dates - 1, height, width)}")


if __name__ == "__main__":
    main()
