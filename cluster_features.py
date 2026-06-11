import argparse
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import MiniBatchKMeans

try:
    from .utils import save_previw_png, write_cluster_geotiff
except ImportError:
    from utils import save_preview_png, write_cluster_geotiff


def predict_in_batches(kmeans, features, batch_size): 
    labels = [] 
    for start in range(0, features.shape[0], batch_size): 
        end = min(start + batch_size, features.shape[0]) 
        labels.append(kmeans.predict(features[start:end])) 
    return np.concatenate(labels, axis=0) 
    
def main(): 
    parser = argparse.ArgumentParser(description="Cluster extracted SAR TCN features.") 
    parser.add_argument("--features", required=True, help="Feature .npz file from extract_features.py.") 
    parser.add_argument("--output-tif", required=True, help="Output cluster GeoTIFF.") 
    parser.add_argument("--output-png", required=True, help="Output PNG preview.") 
    parser.add_argument("--reference-image", default=None, help="Reference GeoTIFF for georeferencing.") 
    parser.add_argument("--n-clusters", type=int, default=8) 
    parser.add_argument("--max-fit-samples", type=int, default=100000) 
    parser.add_argument("--batch-size", type=int, default=512) 
    parser.add_argument("--seed", type=int, default=42) 
    args = parser.parse_args() 
    
    data = np.load(args.features, allow_pickle=False) 
    features = data["features"].astype(np.float32) # [N, C] 
    coords = data["coords"].astype(np.int64) # [N, 2] 
    height = int(data["height"]) 
    width = int(data["width"]) 
    rng = np.random.default_rng(args.seed) 
    fit_count = min(args.max_fit_samples, features.shape[0]) 
    fit_indices = rng.choice(features.shape[0], size=fit_count, replace=False) 
    
    kmeans = MiniBatchKMeans( n_clusters=args.n_clusters, batch_size=args.batch_size, random_state=args.seed, n_init=10) 
    kmeans.fit(features[fit_indices]) 
    labels = predict_in_batches(kmeans, features, args.batch_size) 
    cluster_map = np.full((height, width), -1, dtype=np.int16) 
    cluster_map[coords[:, 0], coords[:, 1]] = labels.astype(np.int16)

    reference_image = args.reference_image 
    if reference_image is None and "reference_path" in data.files: 
        reference_image = str(data["reference_path"]) 
    if reference_image is None: 
        raise ValueError("A reference GeoTIFF is required to write the cluster map.") 
        
    with rasterio.open(reference_image) as src: 
        profile = src.profile.copy() 
    unique, counts = np.unique(cluster_map, return_counts=True) 
    print("cluster_map min:", cluster_map.min()) 
    print("cluster_map max:", cluster_map.max()) 
    print("unique labels:", unique) 
    print("counts:", counts) 
    print("valid clustered pixels:", np.sum(cluster_map >= 0)) 
    print("background pixels:", np.sum(cluster_map == -1)) 
    write_cluster_geotiff(args.output_tif, cluster_map, profile) 
    # save_cluster_png(args.output_png, cluster_map) 
    print(f"saved cluster map to {Path(args.output_tif)} and preview to {Path(args.output_png)}")

if __name__ == "__main__":
    main()

