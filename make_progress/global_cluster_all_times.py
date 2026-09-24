#!/usr/bin/env python3
import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.cluster import MiniBatchKMeans

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_metadata(path: Path):
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        if value.shape == ():
            try:
                out[key] = value.item()
            except Exception:
                out[key] = value
        else:
            out[key] = value
    return out


def parse_date_from_path(value):
    m = DATE_RE.search(str(value))
    return m.group(1) if m else None


def infer_dates(metadata, t_out, exclude_first=True):
    """Infer one date name for each temporal feature slice."""
    candidates = None

    if "recency_dates" in metadata:
        arr = np.asarray(metadata["recency_dates"]).reshape(-1)
        if arr.size > 0:
            candidates = [str(x) for x in arr.tolist()]

    if candidates is None and "image_paths" in metadata:
        arr = np.asarray(metadata["image_paths"]).reshape(-1)
        candidates = []
        for p in arr.tolist():
            date = parse_date_from_path(p)
            candidates.append(date if date is not None else Path(str(p)).stem)

    if candidates is None:
        return [f"time_{i:02d}" for i in range(t_out)]

    if exclude_first and len(candidates) == t_out + 1:
        candidates = candidates[1:]

    if len(candidates) != t_out:
        print(
            f"[Warning] Found {len(candidates)} dates in metadata but features have "
            f"T_out={t_out}; using generic time names.",
            flush=True,
        )
        return [f"time_{i:02d}" for i in range(t_out)]

    return candidates


def infer_hw(metadata, coords, reference_tif=None):
    if reference_tif is not None:
        with rasterio.open(reference_tif) as src:
            return src.height, src.width

    if "height" in metadata and "width" in metadata:
        return int(metadata["height"]), int(metadata["width"])

    for key in ("image_shape", "raster_shape", "shape"):
        if key in metadata:
            shape = np.asarray(metadata[key]).reshape(-1)
            if shape.size >= 2:
                return int(shape[0]), int(shape[1])

    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def iter_time_pixel_chunks(features, chunk_size):
    """
    Iterate over a 3D feature array [N, T, C] without flattening it in RAM.
    Each yielded chunk has shape [B, C].
    """
    n, t_out, _ = features.shape
    for t in range(t_out):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield t, start, end, features[start:end, t, :]


def compute_global_mean_std(features, chunk_size=262_144, eps=1e-6):
    total_count = 0
    total_sum = None
    total_sumsq = None

    for _, _, _, chunk in iter_time_pixel_chunks(features, chunk_size):
        x = np.asarray(chunk, dtype=np.float64)
        if total_sum is None:
            c = x.shape[1]
            total_sum = np.zeros(c, dtype=np.float64)
            total_sumsq = np.zeros(c, dtype=np.float64)
        total_sum += x.sum(axis=0)
        total_sumsq += np.square(x).sum(axis=0)
        total_count += x.shape[0]

    if total_count == 0:
        raise ValueError("No feature vectors found.")

    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    std = np.sqrt(np.maximum(var, 0.0))
    std[~np.isfinite(std)] = 1.0
    std[std < eps] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), int(total_count)


def standardize_chunk(chunk, mean, std):
    x = np.asarray(chunk, dtype=np.float32)
    return (x - mean[None, :]) / std[None, :]


def train_global_kmeans(
    features,
    k,
    mean,
    std,
    chunk_size=262_144,
    batch_size=8192,
    epochs=5,
    seed=42,
    n_init=3,
):
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
    n, t_out, _ = features.shape

    for epoch in range(epochs):
        print(f"[KMeans] epoch {epoch + 1}/{epochs}", flush=True)
        rng = np.random.default_rng(seed + epoch)
        time_order = rng.permutation(t_out)

        for t in time_order:
            starts = np.arange(0, n, chunk_size)
            rng.shuffle(starts)
            for start in starts:
                end = min(start + chunk_size, n)
                x = standardize_chunk(features[start:end, t, :], mean, std)
                if not initialized:
                    if x.shape[0] < k:
                        continue
                    initialized = True
                kmeans.partial_fit(x)

    if not initialized:
        raise RuntimeError("KMeans was not initialized. Check feature size or k.")
    return kmeans


def predict_labels_for_time(kmeans, features, time_index, mean, std, chunk_size):
    n = features.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = standardize_chunk(features[start:end, time_index, :], mean, std)
        labels[start:end] = kmeans.predict(x).astype(np.int32)
    return labels


def labels_to_cluster_map(labels, coords, height, width, invalid_value=-1):
    if labels.shape[0] != coords.shape[0]:
        raise ValueError("labels and coords have different lengths")
    dtype = np.int16 if labels.max(initial=0) <= np.iinfo(np.int16).max else np.int32
    out = np.full((height, width), invalid_value, dtype=dtype)
    rows = coords[:, 0].astype(np.int64)
    cols = coords[:, 1].astype(np.int64)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    out[rows[valid], cols[valid]] = labels[valid].astype(dtype)
    return out


def save_cluster_tif(cluster_map, output_path, reference_tif, nodata=-1):
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    dtype = rasterio.int16 if cluster_map.dtype == np.int16 else rasterio.int32
    profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cluster_map, 1)


def save_cluster_preview(cluster_map, output_path, k, title=None, max_plot_size=2500):
    h, w = cluster_map.shape
    stride = max(1, int(np.ceil(max(h, w) / max_plot_size)))
    cm = cluster_map[::stride, ::stride]

    base_colors = plt.cm.get_cmap("tab20", max(k, 1))(np.arange(k))
    colors = np.vstack([np.array([[0.0, 0.0, 0.0, 1.0]]), base_colors])
    cmap = ListedColormap(colors)
    bounds = np.arange(-1.5, k + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(cm, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title or "Global cluster map")
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
        description="Global MiniBatchKMeans for one all-time feature array [N,T,C]."
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="Directory containing features.npy [N,T,C], coords.npy and metadata.npz.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--reference-tif", default=None, type=Path)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--mean-std-chunk-size", type=int, default=262_144)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--include-first-date", action="store_true", help="Use when features.npy includes h0 as well as later dates.")
    parser.add_argument("--save-tif", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-plot-size", type=int, default=2500)
    args = parser.parse_args()

    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = feature_dir / "features.npy"
    coords_path = feature_dir / "coords.npy"
    metadata_path = feature_dir / "metadata.npz"

    for p in (features_path, coords_path):
        if not p.exists():
            raise FileNotFoundError(p)

    features = np.load(features_path, mmap_mode="r")
    coords = np.load(coords_path, mmap_mode="r")
    metadata = load_metadata(metadata_path)

    if features.ndim != 3:
        raise ValueError(
            f"Expected features.npy shape [N,T,C], got {features.shape}. "
            "Use the original global_cluster.py for separate [N,C] date folders."
        )

    n, t_out, c = features.shape
    if coords.shape != (n, 2):
        raise ValueError(f"Expected coords shape {(n, 2)}, got {coords.shape}")

    dates = infer_dates(metadata, t_out, exclude_first=not args.include_first_date)
    height, width = infer_hw(metadata, coords, args.reference_tif)

    print(f"Feature shape: {features.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Dates ({len(dates)}): {dates}")
    print(f"Raster shape: {(height, width)}")
    print(f"K: {args.k}")

    print("[1/4] Computing global mean/std...", flush=True)
    mean, std, total_count = compute_global_mean_std(
        features,
        chunk_size=args.mean_std_chunk_size,
    )
    np.save(output_dir / "global_feature_mean.npy", mean)
    np.save(output_dir / "global_feature_std.npy", std)
    print(f"Total feature vectors: {total_count}")

    print("[2/4] Training global MiniBatchKMeans...", flush=True)
    kmeans = train_global_kmeans(
        features=features,
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

    print("[3/4] Predicting cluster maps for each date...", flush=True)
    all_counts = []
    labels_all = np.lib.format.open_memmap(
        output_dir / "global_cluster_labels_all_dates.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n, t_out),
    )

    for t, date_name in enumerate(dates):
        print(f"  Predicting {t + 1}/{t_out}: {date_name}", flush=True)
        labels = predict_labels_for_time(
            kmeans, features, t, mean, std, args.chunk_size
        )
        labels_all[:, t] = labels

        cluster_map = labels_to_cluster_map(labels, coords, height, width, invalid_value=-1)
        date_out = output_dir / date_name
        date_out.mkdir(parents=True, exist_ok=True)

        np.save(date_out / "global_cluster_labels.npy", labels)
        np.save(date_out / "global_cluster_map.npy", cluster_map)

        counts = np.bincount(labels, minlength=args.k)
        pd.DataFrame({
            "cluster": np.arange(args.k),
            "count": counts,
            "date": date_name,
        }).to_csv(date_out / "global_cluster_counts.csv", index=False)

        for cluster_id, count in enumerate(counts):
            all_counts.append({
                "date": date_name,
                "cluster": int(cluster_id),
                "count": int(count),
            })

        if args.save_tif:
            if args.reference_tif is None:
                raise ValueError("--save-tif requires --reference-tif")
            save_cluster_tif(
                cluster_map,
                date_out / "global_cluster_map.tif",
                args.reference_tif,
            )

        if args.save_preview:
            save_cluster_preview(
                cluster_map,
                date_out / "global_cluster_map_preview.png",
                args.k,
                title=f"Global cluster map: {date_name}",
                max_plot_size=args.max_plot_size,
            )

    labels_all.flush()
    pd.DataFrame(all_counts).to_csv(
        output_dir / "global_cluster_counts_all_dates.csv", index=False
    )

    print("[4/4] Saving summary...", flush=True)
    summary = {
        "method": "global MiniBatchKMeans over one [N,T,C] feature array",
        "feature_dir": str(feature_dir),
        "feature_shape": [int(n), int(t_out), int(c)],
        "dates": dates,
        "k": int(args.k),
        "total_feature_vectors": int(total_count),
        "chunk_size": int(args.chunk_size),
        "mean_std_chunk_size": int(args.mean_std_chunk_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "reference_tif": str(args.reference_tif) if args.reference_tif else None,
    }
    save_json(output_dir / "global_cluster_summary.json", summary)

    print("Done.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
