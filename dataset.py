import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .utils import collect_geotiff_paths, read_sar_stack, valid_pixel_coords
except ImportError:
    from utils import collect_geotiff_paths, read_sar_stack, valid_pixel_coords

import time
import resource

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | mem={mem_mb():.1f} MB", flush=True)

class SARPixelsDataset(Dataset):
    """
    Pixel-wise SAR time-series dataset.

    Args:
        stack: normalized SAR stack [T, H, W].
        coords: pixel coordinates [N, 2] as [row, col].

    Each item returns x with shape [T, 1]. T is read from the stack and is not
    hard-coded.
    """

    def __init__(self, stack, coords, return_coords=False):
        if stack.ndim != 3:
            raise ValueError(f"stack must have shape [T, H, W], got {stack.shape}.")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape [N, 2], got {coords.shape}.")

        self.stack = stack.astype(np.float32, copy=False)
        self.coords = coords.astype(np.int64, copy=False)
        self.return_coords = return_coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        row, col = self.coords[index]
        series = self.stack[:, row, col]  # [T]
        x = torch.from_numpy(series[:, None].copy())  # [T, 1]
        if self.return_coords:
            coord = torch.tensor([row, col], dtype=torch.long)
            return x, coord
        return x


def dataset_from_geotiffs(inputs, lower=1.0, upper=99.0, return_coords=False):
    """Convenience loader for all valid pixels in a SAR stack."""
    log("Collecting GeoTIFF paths")
    paths = collect_geotiff_paths(inputs)
    log(f"Found {len(paths)} GeoTIFF files")

    log("Reading SAR stack")
    stack, valid_mask, profile = read_sar_stack(paths, lower=lower, upper=upper)
    log(f"Finished reading stack: shape={stack.shape}, dtype={stack.dtype}")

    log("Building valid pixel coordinates")
    coords = valid_pixel_coords(valid_mask)
    log(f"Valid coords: shape={coords.shape}, dtype={coords.dtype}")

    log("Creating SARPixelsDataset")
    dataset = SARPixelsDataset(stack, coords, return_coords=return_coords)
    log(f"Dataset created: length={len(dataset)}")

    return dataset, stack, valid_mask, profile, paths

