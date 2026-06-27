import json
import random
import csv
from glob import glob
from pathlib import Path
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import re
import numpy as np
import rasterio
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_channels(value):
    """Parse a comma-separated channel list, e.g. '32,32,32'."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def collect_geotiff_paths(inputs):
    """Return a sorted list of GeoTIFF paths from files, directories, or globs."""
    paths = []
    print(inputs)
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.tif")))
            paths.extend(sorted(path.glob("*.tiff")))
        else:
            matches = glob(item)
            paths.extend(Path(match) for match in matches)
    print(paths)
    unique_paths = sorted({str(path) for path in paths})
    if not unique_paths:
        raise ValueError("No GeoTIFF files found.")
    return unique_paths


def percentile_normalize(image, valid_mask, lower=1.0, upper=99.0):
    """Normalize one SAR image to [0, 1] using valid pixels only."""
    values = image[valid_mask]
    if values.size == 0:
        raise ValueError("No valid pixels available for normalization.")

    lo, hi = np.percentile(values, [lower, upper])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - lo) / (hi - lo)
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)

def nan_gaussian_filter(image, valid, sigma=1.0, eps=1e-6):
    """
    Apply Gaussian blur without spreading NaN into valid pixels.
    """
    image = image.astype(np.float32)
    valid = valid & np.isfinite(image)

    filled = np.where(valid, image, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)

    blurred_sum = gaussian_filter(filled, sigma=sigma, mode="nearest")
    blurred_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

    blurred = blurred_sum / (blurred_weight + eps)
    blurred[~valid] = np.nan

    return blurred.astype(np.float32)

def read_sar_stack(paths, lower=1.0, upper=99.0, reference_path=None, resampling="nearest", gaussian_sigma=1.0, verbose=True):
    """
    Read co-registered single-band GeoTIFFs.

    Returns:
        stack: float32 array [T, H, W], normalized independently per time step.
        valid_mask: bool array [H, W], valid for every time step.
        profile: rasterio profile copied from the first image.
    """
    arrays = []
    masks = []
    profile = None
    reference_shape = None
    reference_transform = None
    reference_crs = None
    
    paths = [str(p) for p in paths]
    if reference_path is None:
        reference_path = paths[0]

    with rasterio.open(reference_path) as ref:
        reference_profile = ref.profile.copy()
        reference_shape = (ref.height, ref.width)
        reference_transform = ref.transform
        reference_crs = ref.crs
    
    if reference_crs is None:
        raise ValueError(f"Reference image has no CRS: {reference_path}")

    profile = reference_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        height=reference_shape[0],
        width=reference_shape[1],
        transform=reference_transform,
        crs=reference_crs,
        nodata=np.nan,
    )

    if resampling == "nearest":
        resampling_method = Resampling.nearest
    elif resampling == "bilinear":
        resampling_method = Resampling.bilinear
    else:
        raise ValueError("resampling must be 'nearest' or 'bilinear'.")

    for path in paths:
        with rasterio.open(path) as src:
            image = src.read(1).astype(np.float32)
            same_grid = (
                image.shape == reference_shape
                and src.transform == reference_transform
                and src.crs == reference_crs
            )
            if same_grid:
                aligned = image
            else:
                if verbose:
                    print("[Aligning]", path, "shape", image.shape, "->", reference_shape, flush=True)

                aligned = np.full(reference_shape, np.nan, dtype=np.float32)
                reproject(
                    source=image,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=reference_transform,
                    dst_crs=reference_crs,
                    dst_nodata=np.nan,
                    resampling=resampling_method,
                )
            valid = np.isfinite(aligned)
            if src.nodata is not None:
                valid &= aligned != src.nodata

            aligned = aligned.astype(np.float32)
            aligned[~valid] = np.nan
            # Apply gaussian blur to input images
            if gaussian_sigma is not None and gaussian_sigma > 0:
                print("Applying Gaussian blur")
                aligned = nan_gaussian_filter(
                    image=aligned,
                    valid=valid,
                    sigma=gaussian_sigma,
                )
                valid = np.isfinite(aligned)
            arrays.append(aligned)
            masks.append(valid)

    valid_mask = np.logical_and.reduce(masks)
    if int(valid_mask.sum()) == 0:
        raise ValueError(
            "Common valid_mask is empty after alignment. "
            "Check whether the images actually overlap."
        )
    normalized = [
        percentile_normalize(image, valid_mask & mask, lower=lower, upper=upper)
        for image, mask in zip(arrays, masks)
    ]
    stack = np.stack(normalized, axis=0).astype(np.float32)  # [T, H, W]
    return stack, valid_mask, profile


def valid_pixel_coords(valid_mask):
    """Return valid pixel coordinates as int64 [N, 2] with columns [row, col]."""
    return np.argwhere(valid_mask).astype(np.int64)


def make_tile_split(valid_mask, tile_size=128, val_fraction=0.20, seed=42):
    """
    Split valid pixels by spatial tiles to reduce spatial leakage.

    Whole tiles are assigned to one split. Fractions are approximate because tile
    sizes and valid-pixel counts vary.
    """
    if val_fraction < 0 or val_fraction  >= 1:
        raise ValueError("val_fraction must be non-negative and sum to less than 1.")

    height, width = valid_mask.shape
    tiles = []
    for row0 in range(0, height, tile_size):
        row1 = min(row0 + tile_size, height)
        for col0 in range(0, width, tile_size):
            col1 = min(col0 + tile_size, width)
            tile_mask = valid_mask[row0:row1, col0:col1]
            count = int(tile_mask.sum())
            if count == 0:
                continue
            tiles.append((row0, row1, col0, col1, count))

    if not tiles:
        raise ValueError("No valid tiles found.")

    rng = np.random.default_rng(seed)
    rng.shuffle(tiles)

    total = sum(tile[-1] for tile in tiles)
    targets = {
        "val": total * val_fraction
    }
    counts = {"train": 0, "val": 0}
    split_tiles = {"train": [], "val": []}

    for index, tile in enumerate(tiles):
        remaining_tiles = len(tiles) - index
        if remaining_tiles <= 1:
            split = "train"
        elif counts["val"] < targets["val"]:
            split = "val"
        else:
            split = "train"
        split_tiles[split].append(tile)
        counts[split] += tile[-1]

    coords = {}
    for split, assigned_tiles in split_tiles.items():
        split_coords = []
        for row0, row1, col0, col1, _ in assigned_tiles:
            local = np.argwhere(valid_mask[row0:row1, col0:col1])
            if local.size == 0:
                continue
            local[:, 0] += row0
            local[:, 1] += col0
            split_coords.append(local)
        if split_coords:
            coords[split] = np.concatenate(split_coords, axis=0).astype(np.int64)
        else:
            coords[split] = np.empty((0, 2), dtype=np.int64)

    return coords


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_cluster_geotiff(path, cluster_map, reference_profile):
    """Write an int16 [H, W] cluster map, using -1 as nodata."""
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=rasterio.int16,
        nodata=-1,
        compress="lzw",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cluster_map.astype(np.int16), 1)

def save_preview_png(cluster_map_path, output_png, n_clusters, max_plot_size=2500):
    cluster_map = np.load(cluster_map_path, mmap_mode="r")
    height, width = cluster_map.shape

    stride = max(1, int(np.ceil(max(height, width) / max_plot_size)))
    preview = np.asarray(cluster_map[::stride, ::stride])

    # -1: invalid / no data，0 ~ n_clusters-1: cluster
    base_cmap = plt.get_cmap("tab20", n_clusters)
    colors = ["black"]  # for -1
    colors += [base_cmap(i) for i in range(n_clusters)]
    cmap = ListedColormap(colors)
    bounds = np.arange(-1.5, n_clusters + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(preview, cmap=cmap, norm=norm, interpolation="nearest")
    plt.title(f"Cluster map preview, {n_clusters} clusters")
    plt.axis("off")

    cbar = plt.colorbar(im, ticks=np.arange(-1, n_clusters), fraction=0.046, pad=0.04)
    tick_labels = ["invalid"] + [f"cluster {i}" for i in range(n_clusters)]
    cbar.ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Saved preview PNG to {output_png}")

def save_cluster_png(path, cluster_map):
    """Save a quick PNG preview. Invalid pixels (-1) are shown as black."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = cluster_map.astype(np.float32)
    preview[preview < 0] = np.nan

    cmap = plt.get_cmap("tab20")
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="black")
    plt.imsave(path, preview, cmap=cmap)

def parse_dates(image_paths):
    dates = []

    for p in image_paths:
        p_str = str(p)
        match = re.search(r"\d{4}-\d{2}-\d{2}", p_str)

        if match is None:
            raise ValueError(
                f"Cannot parse date from path: {p_str}. "
                "Please provide dates manually with --recency-dates."
            )

        dates.append(match.group(0))

    return dates

def make_causal_recency_matrix(dates, half_life_days=6.0, device="cpu"):
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    T = len(date_objs)
    delta_days = torch.zeros((T, T), dtype=torch.float32, device=device)
    for t in range(T):
        for i in range(T):
            delta_days[t, i] = (date_objs[t] - date_objs[i]).days
    # causal mask: only current and previous dates can contribute
    causal_mask = delta_days >= 0
    # half-life decay:
    # every half_life_days older, weight becomes half
    weights = 0.5 ** (delta_days / half_life_days)
    # remove future information
    weights = torch.where(causal_mask, weights, torch.zeros_like(weights))
    # normalize each row
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    return weights


def add_recency_weighted_channel(x, recency_matrix):
    if recency_matrix is None:
        return x

    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError(f"Expected x shape [B, T, 1], got {x.shape}")

    B, T, _ = x.shape

    if recency_matrix.shape != (T, T):
        raise ValueError(
            f"recency_matrix shape {recency_matrix.shape} does not match T={T}"
        )

    raw = x.squeeze(-1)  # [B, T]

    # weighted[:, t] = sum_i raw[:, i] * A[t, i]
    weighted = torch.matmul(raw, recency_matrix.T)  # [B, T]
    weighted = weighted.unsqueeze(-1)               # [B, T, 1]

    x_input = torch.cat([x, weighted], dim=-1)       # [B, T, 2]

    return x_input

def save_cluster_counts_csv(cluster_counts, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "pixel_count", "fraction"])

        total = int(cluster_counts.sum())
        for i, count in enumerate(cluster_counts):
            fraction = count / total if total > 0 else 0.0
            writer.writerow([i, int(count), fraction])


def save_geotiff(cluster_map_path, output_tif, reference_path):
    cluster_map = np.load(cluster_map_path, mmap_mode="r")
    height, width = cluster_map.shape
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype=rasterio.int16, nodata=-1, compress="lzw", height=height, width=width)

    row_chunk = 2048

    with rasterio.open(output_tif, "w", **profile) as dst:
        for row_start in range(0, height, row_chunk):
            row_end = min(row_start + row_chunk, height)
            arr = np.asarray(cluster_map[row_start:row_end, :], dtype=np.int16)
            window = Window(col_off=0, row_off=row_start, width=width, height=row_end - row_start,)
            dst.write(arr, 1, window=window)

    print(f"Saved GeoTIFF to {output_tif}")


def compute_feature_mean_std(features, batch_size):
    """
    Streaming mean/std over all features.
    """
    n, c = features.shape

    total = 0
    sum_x = np.zeros(c, dtype=np.float64)
    sum_x2 = np.zeros(c, dtype=np.float64)

    for start, end in iter_chunk_ranges(n, batch_size, shuffle_chunks=False):
        x = np.asarray(features[start:end], dtype=np.float32)
        x64 = x.astype(np.float64, copy=False)

        sum_x += x64.sum(axis=0)
        sum_x2 += (x64 * x64).sum(axis=0)
        total += x.shape[0]

        if start % (batch_size * 200) == 0:
            print(f"standardization pass: processed {end}/{n}")

    mean = sum_x / max(total, 1)
    var = sum_x2 / max(total, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)

def load_metadata(metadata_path):
    meta = np.load(metadata_path, allow_pickle=True)
    height = int(meta["height"])
    width = int(meta["width"])
    reference_path = None
    if "reference_path" in meta:
        reference_path = str(meta["reference_path"].item())

    return meta, height, width, reference_path

def iter_chunk_ranges(n, batch_size, shuffle_chunks=False, seed=42):
    starts = np.arange(0, n, batch_size)

    if shuffle_chunks:
        rng = np.random.default_rng(seed)
        rng.shuffle(starts)

    for start in starts:
        end = min(start + batch_size, n)
        yield int(start), int(end)