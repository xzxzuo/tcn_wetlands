from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


DATA_ROOT = "/cephyr/users/xuzu/Alvis/Desktop/mimer_deep-wetlands-data-2025/xzuo"
TRAIN_DIR = Path(f"{DATA_ROOT}/Orebro_lan/2018/train_data")
# TEST_DIR = Path(f"{DATA_ROOT}/Orebro_lan/2018/train_data")
TEST_DIR = Path(f"{DATA_ROOT}/test_data/hjalstaviken/hjalstaviken_2018/0519")
TEST_DIR = Path(f"{DATA_ROOT}/test_data/hornborgasjon/hornborgasjon_2022/0501")
# TEST_DIR = Path(f"{DATA_ROOT}/test_data/svartadalen/svartadalen_2019/0406")
TRAIN_MEAN_PATH = Path("2018/march_to_june/global_sar_mean.npy")
TRAIN_STD_PATH = Path("2018/march_to_june/global_sar_std.npy")
OUTPUT_PATH = "test_zscore_comparison_horn_2022-05-01.png"

BAND = 1
BINS = 100
MAX_SAMPLES_PER_IMAGE = 200_000_000
RANDOM_SEED = 42


def collect_tif_files(directory):
    files = list(directory.rglob("*.tif"))
    files += list(directory.rglob("*.tiff"))
    files += list(directory.rglob("*.TIF"))
    files += list(directory.rglob("*.TIFF"))
    return sorted(set(files))


def read_pixels(path):
    with rasterio.open(path) as src:
        image = src.read(BAND, masked=True)

    pixels = image.compressed().astype(np.float32)
    pixels = pixels[np.isfinite(pixels)]
    # pixels = pixels[pixels != 0]

    return pixels


def sample_pixels(pixels, max_samples, rng):
    if pixels.size > max_samples:
        indices = rng.choice(
            pixels.size,
            size=max_samples,
            replace=False,
        )
        pixels = pixels[indices]

    return pixels


train_mean = np.asarray(np.load(TRAIN_MEAN_PATH)).squeeze()
train_std = np.asarray(np.load(TRAIN_STD_PATH)).squeeze()

if train_mean.ndim != 0 or train_std.ndim != 0:
    raise ValueError(
        "This script expects scalar train mean/std for a single SAR band. "
        f"Got mean shape {train_mean.shape} and std shape {train_std.shape}."
    )

train_mean = float(train_mean)
train_std = float(train_std)

if not np.isfinite(train_std) or train_std <= 0:
    raise ValueError(f"Invalid training std: {train_std}")

print(f"Training mean: {train_mean:.6f}")
print(f"Training std:  {train_std:.6f}")

train_files = collect_tif_files(TRAIN_DIR)
test_files = collect_tif_files(TEST_DIR)

if not test_files:
    raise ValueError(f"No GeoTIFF files found in: {TEST_DIR}")

rng = np.random.default_rng(RANDOM_SEED)

train_image_zscores = []
per_image_zscores = []
train_stats_zscores = []

for path in test_files:
    pixels = read_pixels(path)

    if pixels.size == 0:
        print(f"Skip empty image: {path}")
        continue

    image_mean = float(np.mean(pixels))
    image_std = float(np.std(pixels))

    if image_std <= 0:
        print(f"Skip constant image: {path}")
        continue

    z_per_image = (pixels - image_mean) / image_std
    z_train_stats = (pixels - train_mean) / train_std

    z_per_image = sample_pixels(
        z_per_image,
        MAX_SAMPLES_PER_IMAGE,
        rng,
    )
    z_train_stats = sample_pixels(
        z_train_stats,
        MAX_SAMPLES_PER_IMAGE,
        rng,
    )

    per_image_zscores.append((path, z_per_image))
    train_stats_zscores.append((path, z_train_stats))

    print(
        f"{path.name}: "
        f"mean={image_mean:.4f}, std={image_std:.4f}, "
        f"train-z mean={np.mean(z_train_stats):.4f}, "
        f"train-z std={np.std(z_train_stats):.4f}"
    )

for path in train_files:
    pixels = read_pixels(path)
    image_mean = float(np.mean(pixels))
    image_std = float(np.std(pixels))
    z_per_image = (pixels - image_mean) / image_std
    z_per_image = sample_pixels(
        z_per_image,
        MAX_SAMPLES_PER_IMAGE,
        rng,
    )
    train_image_zscores.append((path, z_per_image))

if not per_image_zscores:
    raise ValueError("No valid test images were found.")

all_zscores = np.concatenate(
    [z for _, z in per_image_zscores]
    + [z for _, z in train_stats_zscores]
    + [z for _, z in train_image_zscores]
)

lower, upper = np.percentile(all_zscores, [0.5, 99.5])
bins = np.linspace(lower, upper, BINS + 1)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.5),
    sharex=True,
    sharey=True,
)

for _, zscores in train_image_zscores:
    axes[0].hist(zscores, bins=bins, density=True, histtype="step", linewidth=1.2, alpha=0.75, label=path.stem)

axes[0].set_title("Train image per-image Z-score")
axes[0].set_xlabel("Standardized pixel intensity")
axes[0].set_ylabel("Density")
axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
axes[0].grid(alpha=0.2)

for path, zscores in per_image_zscores:
    axes[1].hist(
        zscores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        alpha=0.75,
        label=path.stem,
    )

axes[1].set_title("Per-image Z-score")
axes[1].set_xlabel("Standardized pixel intensity")
axes[1].set_ylabel("Density")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].grid(alpha=0.2)

for path, zscores in train_stats_zscores:
    axes[2].hist(
        zscores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        alpha=0.75,
        label=path.stem,
    )

axes[2].set_title("Z-score using training mean/std")
axes[2].set_xlabel("Standardized pixel intensity")
axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
axes[2].grid(alpha=0.2)

if len(per_image_zscores) <= 15:
    axes[2].legend(
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)
# plt.show()