#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def load_metadata(path: Path) -> dict:
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


def infer_height_width(metadata: dict, coords: np.ndarray, reference_tif: Path | None):
    if reference_tif is not None:
        with rasterio.open(reference_tif) as src:
            return int(src.height), int(src.width)

    for h_key, w_key in [("height", "width"), ("H", "W")]:
        if h_key in metadata and w_key in metadata:
            return int(metadata[h_key]), int(metadata[w_key])

    for key in ["image_shape", "raster_shape", "shape"]:
        if key in metadata:
            shape = np.asarray(metadata[key]).reshape(-1)
            if shape.size >= 2:
                return int(shape[0]), int(shape[1])

    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def infer_reference_tif(metadata: dict) -> Path | None:
    image_paths = metadata.get("image_paths")
    if image_paths is not None:
        paths = np.asarray(image_paths).reshape(-1)
        if paths.size > 0:
            candidate = Path(str(paths[-1]))
            if candidate.exists():
                return candidate

    reference_path = metadata.get("reference_path")
    if reference_path is not None:
        candidate = Path(str(reference_path))
        if candidate.exists():
            return candidate

    return None


def save_semantic_tif(
    semantic_map: np.ndarray,
    output_path: Path,
    reference_tif: Path,
    nodata: int = 255,
):
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()

    if semantic_map.shape != (profile["height"], profile["width"]):
        raise ValueError(
            f"Semantic map shape {semantic_map.shape} does not match reference raster "
            f"shape {(profile['height'], profile['width'])}."
        )

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=nodata,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(semantic_map.astype(np.uint8, copy=False), 1)


def save_preview(semantic_map: np.ndarray, output_path: Path):
    display = np.ma.masked_where(semantic_map == 255, semantic_map)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Predicted final-date semantics\n0 = background, 1 = water-like")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels(["background", "water-like"])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Predict final-date water/background labels with an unbundled trained KMeans."
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="Test feature directory containing features.npy, coords.npy, and metadata.npz.",
    )
    parser.add_argument(
        "--kmeans-dir",
        required=True,
        type=Path,
        help=(
            "Training output directory containing global_kmeans.pkl, "
            "global_feature_mean.npy, and global_feature_std.npy."
        ),
    )
    parser.add_argument(
        "--semantics-json",
        type=Path,
        default=None,
        help=(
            "Path to water_cluster_semantics.json. Default: "
            "<kmeans-dir>/otsu_semantics/water_cluster_semantics.json"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-tif",
        type=Path,
        default=None,
        help="Reference GeoTIFF used to save the output map. If omitted, metadata is checked.",
    )
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--save-tif", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    feature_dir = args.feature_dir.expanduser().resolve()
    kmeans_dir = args.kmeans_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = feature_dir / "features.npy"
    coords_path = feature_dir / "coords.npy"
    metadata_path = feature_dir / "metadata.npz"

    kmeans_path = kmeans_dir / "global_kmeans.pkl"
    mean_path = kmeans_dir / "global_feature_mean.npy"
    std_path = kmeans_dir / "global_feature_std.npy"
    semantics_path = (
        args.semantics_json.expanduser().resolve()
        if args.semantics_json is not None
        else kmeans_dir / "otsu_semantics" / "water_cluster_semantics.json"
    )

    required_paths = [
        features_path,
        coords_path,
        kmeans_path,
        mean_path,
        std_path,
        semantics_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    features = np.load(features_path, mmap_mode="r")
    coords = np.load(coords_path, mmap_mode="r")
    metadata = load_metadata(metadata_path)

    if features.ndim not in (2, 3):
        raise ValueError(
            f"features.npy must have shape [N,C] or [N,T,C], got {features.shape}."
        )
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords.npy must have shape [N,2], got {coords.shape}.")
    if coords.shape[0] != features.shape[0]:
        raise ValueError(
            f"Number of coords {coords.shape[0]} does not match number of features {features.shape[0]}."
        )

    with open(kmeans_path, "rb") as file:
        kmeans = pickle.load(file)

    mean = np.asarray(np.load(mean_path), dtype=np.float32)
    std = np.asarray(np.load(std_path), dtype=np.float32)
    std = std.copy()
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-6] = 1.0

    with open(semantics_path, "r", encoding="utf-8") as file:
        semantics = json.load(file)

    if "water_cluster_ids" in semantics:
        raw_water_cluster_ids = semantics["water_cluster_ids"]
    elif "water_cluster_id" in semantics:
        raw_water_cluster_ids = [semantics["water_cluster_id"]]
    else:
        raise KeyError(
            f"{semantics_path} must contain "
            "'water_cluster_ids' or 'water_cluster_id'."
        )

    if not isinstance(raw_water_cluster_ids, (list, tuple)):
        raise TypeError(
            "'water_cluster_ids' must be a list, "
            f"but got {type(raw_water_cluster_ids).__name__}."
        )

    water_cluster_ids = sorted({
        int(cluster_id)
        for cluster_id in raw_water_cluster_ids
    })

    if not water_cluster_ids:
        raise ValueError(
            "'water_cluster_ids' cannot be empty."
        )

    n_pixels = int(features.shape[0])
    feature_dim = int(features.shape[-1])
    n_times = int(features.shape[1]) if features.ndim == 3 else 1

    if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
        raise ValueError(
            f"Scaler dimension mismatch: feature_dim={feature_dim}, "
            f"mean={mean.shape}, std={std.shape}."
        )
    if kmeans.cluster_centers_.shape[1] != feature_dim:
        raise ValueError(
            f"KMeans expects {kmeans.cluster_centers_.shape[1]} features, "
            f"but test features have {feature_dim}."
        )
    n_clusters = int(kmeans.cluster_centers_.shape[0])

    invalid_water_cluster_ids = [
        cluster_id
        for cluster_id in water_cluster_ids
        if not 0 <= cluster_id < n_clusters
    ]

    if invalid_water_cluster_ids:
        raise ValueError(
            f"Invalid water cluster IDs: "
            f"{invalid_water_cluster_ids}. "
            f"Valid IDs for K={n_clusters} are "
            f"0 to {n_clusters - 1}."
        )

    print(f"Test features shape: {features.shape}")
    print(f"Using final feature time index: {n_times - 1}")
    print(
        "Water cluster IDs learned from training Otsu masks: "
        f"{water_cluster_ids}"
    )
    cluster_labels_path = output_dir / "final_cluster_labels.npy"
    semantic_labels_path = output_dir / "final_semantic_labels.npy"

    cluster_labels_mm = np.lib.format.open_memmap(
        cluster_labels_path,
        mode="w+",
        dtype=np.int32,
        shape=(n_pixels,),
    )
    semantic_labels_mm = np.lib.format.open_memmap(
        semantic_labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_pixels,),
    )

    for start in range(0, n_pixels, args.chunk_size):
        end = min(start + args.chunk_size, n_pixels)

        if features.ndim == 3:
            # KMeans input is [B,C], using only the final-date temporal feature.
            x = np.asarray(features[start:end, -1, :], dtype=np.float32)
        else:
            x = np.asarray(features[start:end, :], dtype=np.float32)

        x = (x - mean[None, :]) / std[None, :]
        cluster_labels = kmeans.predict(x).astype(
            np.int32,
            copy=False,
        )
        semantic_labels = np.isin(
            cluster_labels,
            water_cluster_ids,
        ).astype(np.uint8)

        cluster_labels_mm[start:end] = cluster_labels
        semantic_labels_mm[start:end] = semantic_labels

        if start == 0 or end == n_pixels or (start // args.chunk_size) % 20 == 0:
            print(
                f"Predicted {end}/{n_pixels} pixels "
                f"({100.0 * end / n_pixels:.1f}%)",
                flush=True,
            )

    cluster_labels_mm.flush()
    semantic_labels_mm.flush()

    reference_tif = (
        args.reference_tif.expanduser().resolve()
        if args.reference_tif is not None
        else infer_reference_tif(metadata)
    )

    height, width = infer_height_width(metadata, coords, reference_tif)
    semantic_map = np.full((height, width), 255, dtype=np.uint8)
    cluster_map = np.full((height, width), -1, dtype=np.int16)

    rows = np.asarray(coords[:, 0], dtype=np.int64)
    cols = np.asarray(coords[:, 1], dtype=np.int64)
    valid_coords = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    semantic_map[rows[valid_coords], cols[valid_coords]] = np.asarray(
        semantic_labels_mm[valid_coords], dtype=np.uint8
    )
    cluster_map[rows[valid_coords], cols[valid_coords]] = np.asarray(
        cluster_labels_mm[valid_coords], dtype=np.int16
    )

    np.save(output_dir / "final_semantic_map.npy", semantic_map)
    np.save(output_dir / "final_cluster_map.npy", cluster_map)

    water_pixels = int((semantic_labels_mm[:] == 1).sum())
    background_pixels = int((semantic_labels_mm[:] == 0).sum())

    summary = {
        "feature_dir": str(feature_dir),
        "features_shape": [int(v) for v in features.shape],
        "used_time_index": int(n_times - 1),
        "kmeans_path": str(kmeans_path),
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "semantics_path": str(semantics_path),
        "water_cluster_ids": water_cluster_ids,
        "number_of_water_clusters": len(water_cluster_ids),
        "semantic_mapping": {"0": "background", "1": "water-like", "255": "invalid"},
        "water_pixels": water_pixels,
        "background_pixels": background_pixels,
        "water_ratio_valid": water_pixels / max(water_pixels + background_pixels, 1),
        "reference_tif": str(reference_tif) if reference_tif is not None else None,
    }
    with open(output_dir / "prediction_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    if args.save_tif:
        if reference_tif is None:
            raise ValueError(
                "Cannot save GeoTIFF because no reference raster was provided or found in metadata."
            )
        save_semantic_tif(
            semantic_map,
            output_dir / "final_semantic_map.tif",
            reference_tif,
        )

    if args.save_preview:
        save_preview(
            semantic_map,
            output_dir / "final_semantic_map_preview.png",
        )

    print("Done.")
    print(f"Saved cluster labels:  {cluster_labels_path}")
    print(f"Saved semantic labels: {semantic_labels_path}")
    print(f"Saved semantic map:    {output_dir / 'final_semantic_map.npy'}")
    print(f"Water ratio among valid pixels: {summary['water_ratio_valid']:.6f}")
    print("Semantic output: 0 = background, 1 = water-like")


if __name__ == "__main__":
    main()
