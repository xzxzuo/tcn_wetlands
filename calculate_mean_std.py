from pathlib import Path
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import re
import numpy as np
import rasterio
from utils import collect_geotiff_paths, nan_gaussian_filter

def calculate_sequence_mean_std(
    paths,
    lower=1.0,
    upper=99.0,
    reference_path=None,
    resampling="nearest",
    gaussian_sigma=1.0,
    verbose=True,
):
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
                    print(f"[Aligning] {path} shape {image.shape} -> {reference_shape}", flush=True)
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
                if verbose:
                    print(f"Applying Gaussian blur: sigma={gaussian_sigma}", flush=True)
                aligned = nan_gaussian_filter(image=aligned, valid=valid, sigma=gaussian_sigma)
                valid = np.isfinite(aligned)
            arrays.append(aligned)
            masks.append(valid)

    valid_mask = np.logical_and.reduce(masks)
    if int(valid_mask.sum()) == 0:
        raise ValueError(
            "Common valid_mask is empty after alignment. "
            "Check whether the images actually overlap."
        )

    total_count = 0
    total_mean = 0.0
    total_m2 = 0.0

    for image, image_mask in zip(arrays, masks):
        valid = (valid_mask & image_mask & np.isfinite(image))
        values = image[valid].astype(np.float64)
        if values.size == 0:
            continue
        batch_count = values.size
        batch_mean = float(values.mean())
        batch_m2 = float(np.square(values - batch_mean).sum())
        if total_count == 0:
            total_count = batch_count
            total_mean = batch_mean
            total_m2 = batch_m2
            continue

        delta = batch_mean - total_mean
        new_count = total_count + batch_count

        total_mean = (total_mean + delta * batch_count / new_count)
        total_m2 = (total_m2 + batch_m2 + delta ** 2 * total_count * batch_count / new_count)
        total_count = new_count

    if total_count == 0:
        raise ValueError("No valid training pixels found.")

    variance = total_m2 / total_count
    train_mean = total_mean
    train_std = np.sqrt(variance)

    if train_std <= 1e-8:
        raise ValueError(f"Training std is too small: {train_std}")
    return float(train_mean), float(train_std)

if __name__ == "__main__":
    paths = collect_geotiff_paths("/nobackup/proj/flash/deep-wetlands-data-2025/personal/xuzu/Orebro_lan/2018/train_data/")
    train_mean, train_std = calculate_sequence_mean_std(paths=paths)
    print("Training SAR mean:", train_mean)
    print("Training SAR std:", train_std)
    output_dir = Path("./2018/same_kmeans")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "global_sar_mean.npy", np.asarray(train_mean, dtype=np.float32))
    np.save(output_dir / "global_sar_std.npy", np.asarray(train_std, dtype=np.float32))