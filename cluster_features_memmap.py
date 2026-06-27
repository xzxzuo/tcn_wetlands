import argparse
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

try:
    from .utils import save_preview_png, save_cluster_counts_csv, save_geotiff, compute_feature_mean_std, load_metadata, iter_chunk_ranges
except ImportError:
    from utils import save_preview_png, save_cluster_counts_csv, save_geotiff, compute_feature_mean_std, load_metadata, iter_chunk_ranges


def get_feature_batch(features, start, end, mean=None, std=None):
    x = np.asarray(features[start:end], dtype=np.float32)

    if mean is not None and std is not None:
        x = (x - mean) / std

    return x


def fit_minibatch_kmeans_all(
    features,
    n_clusters,
    batch_size,
    fit_epochs,
    seed,
    standardize=False,
    shuffle_chunks=True,
):
    n, c = features.shape

    mean = None
    std = None

    if standardize:
        print("Computing feature mean/std over all pixels...")
        mean, std = compute_feature_mean_std(features, batch_size)
        print("Finished standardization statistics.")
    else:
        print("Standardization disabled.")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )

    print("Fitting MiniBatchKMeans using all features...")

    for epoch in range(fit_epochs):
        print(f"Fit epoch {epoch + 1}/{fit_epochs}")

        for batch_id, (start, end) in enumerate(
            iter_chunk_ranges(
                n,
                batch_size,
                shuffle_chunks=shuffle_chunks,
                seed=seed + epoch,
            )
        ):
            x = get_feature_batch(features, start, end, mean, std)
            kmeans.partial_fit(x)

            if batch_id % 100 == 0:
                print(f"  partial_fit batch {batch_id}, pixels {end}/{n}")

    return kmeans, mean, std


def initialize_cluster_map(output_path, height, width):
    cluster_map = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int16,
        shape=(height, width),
    )

    # Initialize in row chunks to avoid creating a large temporary array.
    row_chunk = 2048
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        cluster_map[row_start:row_end, :] = -1

    cluster_map.flush()
    return cluster_map

def predict_all_to_cluster_map(
    features,
    coords,
    kmeans,
    cluster_map,
    n_clusters,
    batch_size,
    mean=None,
    std=None,
):
    n = features.shape[0]
    cluster_counts = np.zeros(n_clusters, dtype=np.int64)

    print("Predicting cluster labels for all pixels...")

    for batch_id, (start, end) in enumerate(iter_chunk_ranges(n, batch_size, shuffle_chunks=False)):
        x = get_feature_batch(features, start, end, mean, std)
        labels = kmeans.predict(x).astype(np.int16)

        batch_coords = np.asarray(coords[start:end], dtype=np.int64)
        rows = batch_coords[:, 0]
        cols = batch_coords[:, 1]

        cluster_map[rows, cols] = labels
        cluster_counts += np.bincount(labels, minlength=n_clusters)

        if batch_id % 100 == 0:
            print(f"  predicted batch {batch_id}, pixels {end}/{n}")

    cluster_map.flush()
    return cluster_counts


def main():
    parser = argparse.ArgumentParser(description="Cluster memmap TCN features using MiniBatchKMeans.partial_fit.")
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, coords.npy, metadata.npz.")
    parser.add_argument("--output-dir", required=True, help="Directory to save cluster outputs.")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--fit-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true", help="Standardize features channel-wise using all pixels before clustering.")
    parser.add_argument("--no-shuffle-chunks",  action="store_true", help="Disable chunk shuffling during MiniBatchKMeans fitting.")
    parser.add_argument("--save-tif", action="store_true", help="Save cluster map as GeoTIFF using reference_path from metadata.npz.")
    parser.add_argument("--reference-path", default=None, help="Optional reference GeoTIFF path. If not given, use metadata reference_path.")
    parser.add_argument("--max-plot-size", type=int, default=2500, help="Maximum size of cluster preview PNG. Large maps are downsampled for preview.")

    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = feature_dir / "features.npy"
    coords_path = feature_dir / "coords.npy"
    metadata_path = feature_dir / "metadata.npz"

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    features = np.load(features_path, mmap_mode="r")
    coords = np.load(coords_path, mmap_mode="r")

    meta, height, width, meta_reference_path = load_metadata(metadata_path)

    print("Loaded inputs:")
    print(f"  features: {features.shape}, dtype={features.dtype}")
    print(f"  coords:   {coords.shape}, dtype={coords.dtype}")
    print(f"  height={height}, width={width}")

    if features.shape[0] != coords.shape[0]:
        raise ValueError(f"features and coords have different N: {features.shape[0]} vs {coords.shape[0]}")

    n, c = features.shape

    kmeans, mean, std = fit_minibatch_kmeans_all(
        features=features,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        fit_epochs=args.fit_epochs,
        seed=args.seed,
        standardize=args.standardize,
        shuffle_chunks=not args.no_shuffle_chunks,
    )

    cluster_map_path = output_dir / "cluster_map.npy"
    cluster_map = initialize_cluster_map(cluster_map_path, height, width)

    cluster_counts = predict_all_to_cluster_map(
        features=features,
        coords=coords,
        kmeans=kmeans,
        cluster_map=cluster_map,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        mean=mean,
        std=std,
    )
    # Uncomment this if you want to save more metadata
    # save_cluster_counts_csv(cluster_counts, output_dir / "cluster_counts.csv")
    # np.save(output_dir / "cluster_centers.npy", kmeans.cluster_centers_)
    # np.savez(
    #     output_dir / "cluster_metadata.npz",
    #     n_clusters=np.array(args.n_clusters, dtype=np.int64),
    #     batch_size=np.array(args.batch_size, dtype=np.int64),
    #     fit_epochs=np.array(args.fit_epochs, dtype=np.int64),
    #     seed=np.array(args.seed, dtype=np.int64),
    #     standardize=np.array(args.standardize),
    #     feature_dir=np.array(str(feature_dir)),
    #     features_shape=np.array(features.shape, dtype=np.int64),
    #     height=np.array(height, dtype=np.int64),
    #     width=np.array(width, dtype=np.int64),
    #     mean=np.array([] if mean is None else mean, dtype=np.float32),
    #     std=np.array([] if std is None else std, dtype=np.float32),
    # )

    feature_dim = int(meta['feature_dim'])
    try:
        half_life_days = int(float(meta["half_life_days"]))
    except Exception:
        half_life_days = -1

    save_preview_png(
        cluster_map_path=cluster_map_path,
        output_png=output_dir / f"preview_k{args.n_clusters}_h{half_life_days}_d{feature_dim}.png",
        n_clusters=args.n_clusters,
        max_plot_size=args.max_plot_size,
    )

    if args.save_tif:
        reference_path = args.reference_path or meta_reference_path
        if reference_path is None:
            raise ValueError("No reference_path found. Please provide --reference-path or save reference_path in metadata.npz.")

        save_geotiff(
            cluster_map_path=cluster_map_path, 
            output_tif=output_dir / f"cluster_map_k{args.n_clusters}_h{half_life_days}_d{feature_dim}.tif", 
            reference_path=reference_path
        )

    cluster_map.flush()
    del cluster_map

    if cluster_map_path.exists():
        cluster_map_path.unlink()
        
    print("Done.")
    print(cluster_counts)
    print(f"cluster data saved to: {output_dir}")


if __name__ == "__main__":
    main()