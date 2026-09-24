#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import MiniBatchKMeans


def find_feature_dirs(feature_root=None, feature_dirs=None):
    """
    Find date-level feature directories.

    Each directory must contain:
        features.npy
        coords.npy
        metadata.npz
    """
    if feature_dirs is not None and len(feature_dirs) > 0:
        dirs = [Path(p) for p in feature_dirs]
    else:
        feature_root = Path(feature_root)
        dirs = sorted([
            p for p in feature_root.iterdir()
            if p.is_dir()
            and (p / "features.npy").exists()
            and (p / "coords.npy").exists()
        ])

    valid_dirs = []
    for d in dirs:
        if not (d / "features.npy").exists():
            raise FileNotFoundError(f"Missing features.npy in {d}")
        if not (d / "coords.npy").exists():
            raise FileNotFoundError(f"Missing coords.npy in {d}")
        valid_dirs.append(d)

    if len(valid_dirs) == 0:
        raise ValueError("No valid feature directories found.")

    return valid_dirs


def load_metadata(path):
    path = Path(path)
    if not path.exists():
        return {}

    data = np.load(path, allow_pickle=True)
    metadata = {}
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


def infer_hw_from_metadata_or_coords(metadata, coords, reference_tif=None):
    """
    Infer raster height and width.
    Priority:
        1. reference_tif
        2. metadata keys
        3. max coords + 1
    """
    if reference_tif is not None:
        with rasterio.open(reference_tif) as src:
            return src.height, src.width

    candidate_keys = [
        ("height", "width"),
        ("H", "W"),
    ]

    for hk, wk in candidate_keys:
        if hk in metadata and wk in metadata:
            return int(metadata[hk]), int(metadata[wk])

    for key in ["image_shape", "raster_shape", "shape"]:
        if key in metadata:
            shape = metadata[key]
            if len(shape) >= 2:
                return int(shape[0]), int(shape[1])

    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    return h, w


def iter_feature_chunks(features, chunk_size):
    n = features.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end, features[start:end]


def compute_global_mean_std(feature_dirs, chunk_size=1_000_000, eps=1e-6):
    """
    Compute global mean/std across all dates and all valid pixels.

    This avoids loading all dates into one huge array.
    """
    total_count = 0
    total_sum = None
    total_sumsq = None

    for d in feature_dirs:
        features = np.load(d / "features.npy", mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(f"{d}/features.npy should have shape [N, C], got {features.shape}")

        for _, _, chunk in iter_feature_chunks(features, chunk_size):
            chunk = np.asarray(chunk, dtype=np.float64)

            if total_sum is None:
                c = chunk.shape[1]
                total_sum = np.zeros(c, dtype=np.float64)
                total_sumsq = np.zeros(c, dtype=np.float64)

            total_sum += chunk.sum(axis=0)
            total_sumsq += (chunk ** 2).sum(axis=0)
            total_count += chunk.shape[0]

    mean = total_sum / max(total_count, 1)
    var = total_sumsq / max(total_count, 1) - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    std[~np.isfinite(std)] = 1.0
    std[std < eps] = 1.0

    return mean.astype(np.float32), std.astype(np.float32), int(total_count)


def standardize_chunk(chunk, mean, std):
    chunk = np.asarray(chunk, dtype=np.float32)
    return (chunk - mean[None, :]) / std[None, :]


def train_global_kmeans(
    feature_dirs,
    k,
    mean,
    std,
    chunk_size=262_144,
    batch_size=8192,
    epochs=10,
    seed=42,
    n_init=3,
):
    """
    Train one global MiniBatchKMeans over all dates.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        n_init=n_init,
        max_no_improvement=20,
        reassignment_ratio=0.01,
        verbose=0,
    )

    initialized = False

    for epoch in range(epochs):
        print(f"[KMeans] epoch {epoch + 1}/{epochs}", flush=True)

        rng = np.random.default_rng(seed + epoch)
        shuffled_dirs = list(feature_dirs)
        rng.shuffle(shuffled_dirs)

        for d in shuffled_dirs:
            features = np.load(d / "features.npy", mmap_mode="r")

            for _, _, chunk in iter_feature_chunks(features, chunk_size):
                x = standardize_chunk(chunk, mean, std)

                if not initialized:
                    if x.shape[0] < k:
                        continue
                    initialized = True

                kmeans.partial_fit(x)

    if not initialized:
        raise RuntimeError("KMeans was not initialized. Check feature size or k.")

    return kmeans


def labels_to_cluster_map(labels, coords, height, width, invalid_value=-1):
    """
    Convert 1D labels back to raster map.
    """
    if labels.shape[0] != coords.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match coords length {coords.shape[0]}"
        )

    if labels.max() <= np.iinfo(np.int16).max:
        dtype = np.int16
    else:
        dtype = np.int32

    cluster_map = np.full((height, width), invalid_value, dtype=dtype)

    rows = coords[:, 0].astype(np.int64)
    cols = coords[:, 1].astype(np.int64)

    valid = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    cluster_map[rows[valid], cols[valid]] = labels[valid].astype(dtype)

    return cluster_map


def predict_labels_for_date(kmeans, feature_dir, mean, std, chunk_size=262_144):
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    labels = np.empty(features.shape[0], dtype=np.int32)

    for start, end, chunk in iter_feature_chunks(features, chunk_size):
        x = standardize_chunk(chunk, mean, std)
        labels[start:end] = kmeans.predict(x).astype(np.int32)

    return labels


def save_cluster_tif(cluster_map, output_path, reference_tif, nodata=-1):
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype=rasterio.int16 if cluster_map.max() <= np.iinfo(np.int16).max else rasterio.int32,
        nodata=nodata,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cluster_map, 1)


def save_cluster_preview(cluster_map, output_path, k, title=None, max_plot_size=2500):
    h, w = cluster_map.shape
    stride = max(1, int(np.ceil(max(h, w) / max_plot_size)))

    cm = cluster_map[::stride, ::stride]

    base_colors = plt.cm.get_cmap("tab20", max(k, 1))(np.arange(k))
    invalid_color = np.array([[0.0, 0.0, 0.0, 1.0]])
    colors = np.vstack([invalid_color, base_colors])

    cmap = ListedColormap(colors)
    bounds = np.arange(-1.5, k + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    im = ax.imshow(cm, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title if title is not None else "Global cluster map")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cluster id; -1 = invalid")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Global clustering over TCN features from all dates."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--feature-root",
        type=str,
        help="Root directory containing one feature directory per date.",
    )
    group.add_argument(
        "--feature-dirs",
        nargs="+",
        help="Explicit list of feature directories, each containing features.npy and coords.npy.",
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, required=True)

    parser.add_argument(
        "--reference-tif",
        default=None,
        help="Optional reference GeoTIFF for output shape/profile and saving cluster_map.tif.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=262_144,
        help="Number of feature rows processed per chunk.",
    )
    parser.add_argument(
        "--mean-std-chunk-size",
        type=int,
        default=1_000_000,
        help="Chunk size for global mean/std computation.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=3)

    parser.add_argument("--save-tif", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-plot-size", type=int, default=2500)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dirs = find_feature_dirs(
        feature_root=args.feature_root,
        feature_dirs=args.feature_dirs,
    )

    print("Found feature dirs:")
    for d in feature_dirs:
        print("  ", d)

    # Check feature dimensions.
    feature_dims = []
    for d in feature_dirs:
        f = np.load(d / "features.npy", mmap_mode="r")
        feature_dims.append(f.shape[1])

    if len(set(feature_dims)) != 1:
        raise ValueError(f"Feature dimensions are inconsistent: {feature_dims}")

    in_channels = int(feature_dims[0])

    print(f"Number of dates: {len(feature_dirs)}")
    print(f"Feature dimension: {in_channels}")
    print(f"K: {args.k}")

    # 1. Compute global feature mean/std.
    print("[1/4] Computing global mean/std...", flush=True)
    mean, std, total_count = compute_global_mean_std(
        feature_dirs,
        chunk_size=args.mean_std_chunk_size,
    )

    np.save(output_dir / "global_feature_mean.npy", mean)
    np.save(output_dir / "global_feature_std.npy", std)

    print(f"Total feature vectors across all dates: {total_count}")
    print("Mean/std saved.")

    # 2. Train global MiniBatchKMeans.
    print("[2/4] Training global MiniBatchKMeans...", flush=True)
    kmeans = train_global_kmeans(
        feature_dirs=feature_dirs,
        k=args.k,
        mean=mean,
        std=std,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        n_init=args.n_init,
    )

    with open(output_dir / "global_kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    np.save(output_dir / "global_cluster_centers_standardized.npy", kmeans.cluster_centers_)

    # 3. Predict cluster maps for each date.
    print("[3/4] Predicting cluster maps for each date...", flush=True)

    all_counts = []

    for date_idx, d in enumerate(feature_dirs):
        date_name = d.name
        print(f"  Predicting {date_idx + 1}/{len(feature_dirs)}: {date_name}", flush=True)

        coords = np.load(d / "coords.npy", mmap_mode="r")
        metadata = load_metadata(d / "metadata.npz")

        height, width = infer_hw_from_metadata_or_coords(
            metadata=metadata,
            coords=coords,
            reference_tif=args.reference_tif,
        )

        labels = predict_labels_for_date(
            kmeans=kmeans,
            feature_dir=d,
            mean=mean,
            std=std,
            chunk_size=args.chunk_size,
        )

        cluster_map = labels_to_cluster_map(
            labels=labels,
            coords=coords,
            height=height,
            width=width,
            invalid_value=-1,
        )

        date_out = output_dir / date_name
        date_out.mkdir(parents=True, exist_ok=True)

        np.save(date_out / "global_cluster_labels.npy", labels.astype(np.int32))
        np.save(date_out / "global_cluster_map.npy", cluster_map)

        counts = np.bincount(labels, minlength=args.k)
        pd.DataFrame({
            "cluster": np.arange(args.k),
            "count": counts,
            "date": date_name,
        }).to_csv(date_out / "global_cluster_counts.csv", index=False)

        for c, cnt in enumerate(counts):
            all_counts.append({
                "date": date_name,
                "cluster": int(c),
                "count": int(cnt),
            })

        if args.save_tif:
            if args.reference_tif is None:
                raise ValueError("--save-tif requires --reference-tif.")
            save_cluster_tif(
                cluster_map,
                date_out / "global_cluster_map.tif",
                reference_tif=args.reference_tif,
                nodata=-1,
            )

        if args.save_preview:
            save_cluster_preview(
                cluster_map=cluster_map,
                output_path=date_out / "global_cluster_map_preview.png",
                k=args.k,
                title=f"Global cluster map: {date_name}",
                max_plot_size=args.max_plot_size,
            )

    pd.DataFrame(all_counts).to_csv(output_dir / "global_cluster_counts_all_dates.csv", index=False)

    # 4. Save summary.
    print("[4/4] Saving summary...", flush=True)

    summary = {
        "method": "global clustering over all-date TCN features",
        "num_dates": len(feature_dirs),
        "feature_dirs": [str(d) for d in feature_dirs],
        "k": int(args.k),
        "in_channels": int(in_channels),
        "total_feature_vectors": int(total_count),
        "chunk_size": int(args.chunk_size),
        "mean_std_chunk_size": int(args.mean_std_chunk_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "reference_tif": str(args.reference_tif) if args.reference_tif is not None else None,
        "outputs": {
            "kmeans": str(output_dir / "global_kmeans.pkl"),
            "mean": str(output_dir / "global_feature_mean.npy"),
            "std": str(output_dir / "global_feature_std.npy"),
            "cluster_centers": str(output_dir / "global_cluster_centers_standardized.npy"),
            "per_date_outputs": "Each date folder contains global_cluster_labels.npy and global_cluster_map.npy",
        },
        "note": (
            "KMeans is fitted once using features from all dates. "
            "Each date is then assigned to the same global cluster centers. "
            "This makes cluster ids comparable across dates."
        ),
    }

    save_json(output_dir / "global_cluster_summary.json", summary)

    print("Done.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

# python global_cluster.py \
#   --feature-root "/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo/test_data/svartadalen/svartadalen_2018/04_features" \
#   --output-dir "/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo/test_data/svartadalen/svartadalen_2018/04_features/global_cluster_k4" \
#   --k 4 \
#   --reference-tif "/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo/test_data/svartadalen/svartadalen_2018/0411/svartadalen_2018-04-11.tif" \
#   --save-tif \
#   --save-preview