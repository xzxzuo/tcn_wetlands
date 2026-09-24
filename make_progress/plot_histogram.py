from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

DATA_ROOT = "/cephyr/users/xuzu/Alvis/Desktop/mimer_deep-wetlands-data-2025/xzuo"
TRAIN_DIR = Path(f"{DATA_ROOT}/Orebro_lan/2018/train_data")
TEST_DIR = Path(f"{DATA_ROOT}/test_data/hjalstaviken/hjalstaviken_2018/0519")
TEST_DIR = Path(f"{DATA_ROOT}/test_data/hornborgasjon/hornborgasjon_2022/0501")
# TEST_DIR = Path(f"{DATA_ROOT}/test_data/svartadalen/svartadalen_2019/0406")
OUTPUT_PATH = "train_test_histogram_horn_2022-05-01.png"
# OUTPUT_PATH = "train_test_histogram_svar_2019-04-06.png"


BAND = 1
BINS = 100
MAX_SAMPLES_PER_IMAGE = 200_000


def collect_tif_files(directory):
    files = list(directory.rglob("*.tif"))
    files += list(directory.rglob("*.tiff"))
    return sorted(files)


def read_pixels(path, max_samples=MAX_SAMPLES_PER_IMAGE):
    with rasterio.open(path) as src:
        image = src.read(BAND, masked=True)

    pixels = image.compressed()
    pixels = pixels[np.isfinite(pixels)]
    # pixels = pixels[pixels != 0]
    if len(pixels) > max_samples:
        rng = np.random.default_rng(42)
        pixels = rng.choice(
            pixels,
            size=max_samples,
            replace=False,
        )

    return pixels


train_files = collect_tif_files(TRAIN_DIR)
test_files = collect_tif_files(TEST_DIR)

if not train_files:
    raise ValueError(f"No GeoTIFF found in: {TRAIN_DIR}")

if not test_files:
    raise ValueError(f"No GeoTIFF found in: {TEST_DIR}")

train_data = [read_pixels(path) for path in train_files]
test_data = [read_pixels(path) for path in test_files]

all_pixels = np.concatenate(train_data + test_data)
lower, upper = np.percentile(all_pixels, [0.5, 99.5])
bins = np.linspace(lower, upper, BINS + 1)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.5),
    sharex=True,
    sharey=True,
)

for path, pixels in zip(train_files, train_data):
    axes[0].hist(
        pixels,
        bins=bins,
        density=True,
        histtype="step",
        alpha=0.7,
        label=path.stem,
    )

axes[0].set_title("Training images")

for path, pixels in zip(test_files, test_data):
    axes[1].hist(
        pixels,
        bins=bins,
        density=True,
        histtype="step",
        alpha=0.7,
        label=path.stem,
    )

axes[1].set_title("Test images")

axes[2].hist(
    np.concatenate(train_data),
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    color="tab:blue",
    label="Training",
)

axes[2].hist(
    np.concatenate(test_data),
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    color="tab:red",
    label="Test",
)

axes[2].set_title("Combined distributions")
axes[2].legend()

for ax in axes:
    ax.set_xlabel("SAR pixel intensity")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Density")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)
# plt.show()