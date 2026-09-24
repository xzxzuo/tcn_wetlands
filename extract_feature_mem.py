#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import SARPixelTCN
from utils import (
    add_recency_weighted_channel,
    collect_geotiff_paths,
    make_causal_recency_matrix,
    parse_dates,
    read_sar_stack,
    valid_pixel_coords,
)


def resolve_device(device_arg: str) -> torch.device:
    """Validate and return the requested PyTorch device."""
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def estimate_output_size_gib(
    num_pixels: int,
    num_time_steps: int,
    feature_dim: int,
    dtype: np.dtype,
) -> float:
    num_bytes = (
        int(num_pixels)
        * int(num_time_steps)
        * int(feature_dim)
        * np.dtype(dtype).itemsize
    )
    return num_bytes / (1024**3)


def select_output_dtype(name: str) -> tuple[np.dtype, torch.dtype]:
    if name == "float32":
        return np.dtype(np.float32), torch.float32
    if name == "float16":
        return np.dtype(np.float16), torch.float16
    raise ValueError(f"Unsupported output dtype: {name}")


@torch.inference_mode()
def extract_all_time_features_direct(
    *,
    model: torch.nn.Module,
    stack: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    features_mm: np.memmap,
    coords_mm: np.memmap,
    batch_size: int,
    recency_matrix: torch.Tensor | None,
    exclude_first: bool,
    output_torch_dtype: torch.dtype,
    pin_memory: bool,
    amp: bool,
    log_every: int,
    profile_every: int,
) -> int:
    """
    Extract all temporal features from an in-memory SAR stack.

    Parameters
    ----------
    stack:
        NumPy array [T, H, W].
    coords:
        Integer array [N, 2], storing [row, col].
    features_mm:
        Memmap [N, T_out, C].
    coords_mm:
        Memmap [N, 2].
    exclude_first:
        If True, save encoder outputs 1..T-1. Otherwise save 0..T-1.
    """
    model.eval()

    num_pixels = int(coords.shape[0])
    num_batches = (num_pixels + batch_size - 1) // batch_size

    use_cuda = device.type == "cuda"
    use_amp = bool(amp and use_cuda)

    if use_amp:
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext

    total_start = time.perf_counter()

    for batch_index, start in enumerate(range(0, num_pixels, batch_size)):
        end = min(start + batch_size, num_pixels)
        batch_start = time.perf_counter()

        # coords is [N, 2]: column 0 is row and column 1 is col.
        batch_coords = np.asarray(coords[start:end], dtype=np.int32)
        rows = batch_coords[:, 0]
        cols = batch_coords[:, 1]

        # Advanced indexing returns [T, B]. Transpose to [B, T], make it
        # contiguous, and add the input-channel dimension -> [B, T, 1].
        x_np = np.ascontiguousarray(stack[:, rows, cols].T, dtype=np.float32)
        prepare_done = time.perf_counter()

        x_cpu = torch.from_numpy(x_np).unsqueeze(-1)
        if pin_memory and use_cuda:
            # non_blocking=True only provides asynchronous H2D transfer from
            # pinned host memory. This is optional because pinned allocations
            # can increase host-memory pressure on very large jobs.
            x_cpu = x_cpu.pin_memory()

        x = x_cpu.to(device=device, non_blocking=pin_memory and use_cuda)
        if use_cuda and profile_every > 0 and batch_index % profile_every == 0:
            torch.cuda.synchronize(device)
        h2d_done = time.perf_counter()

        x_input = add_recency_weighted_channel(x, recency_matrix)

        with amp_context():
            encoded = model.encode(x_input)  # [B, C, T]

        if encoded.ndim != 3:
            raise RuntimeError(
                f"model.encode() must return [B,C,T], got {tuple(encoded.shape)}"
            )

        if exclude_first:
            if encoded.shape[2] < 2:
                raise RuntimeError(
                    "Cannot exclude the first feature because the encoder returned "
                    f"only T={encoded.shape[2]} time step(s)."
                )
            encoded = encoded[:, :, 1:]

        # [B, C, T_out] -> [B, T_out, C]. Convert on the GPU before D2H so
        # --output-dtype float16 also reduces transfer and output-file size.
        output = encoded.permute(0, 2, 1).contiguous().to(output_torch_dtype)

        if use_cuda and profile_every > 0 and batch_index % profile_every == 0:
            torch.cuda.synchronize(device)
        model_done = time.perf_counter()

        output_np = output.cpu().numpy()

        if use_cuda and profile_every > 0 and batch_index % profile_every == 0:
            torch.cuda.synchronize(device)
        d2h_done = time.perf_counter()

        features_mm[start:end] = output_np
        coords_mm[start:end] = batch_coords
        write_done = time.perf_counter()

        should_log = (
            batch_index == 0
            or end == num_pixels
            or (log_every > 0 and (batch_index + 1) % log_every == 0)
        )
        if should_log:
            elapsed = write_done - total_start
            rate = end / max(elapsed, 1e-9)
            print(
                f"[{batch_index + 1}/{num_batches}] "
                f"written {end}/{num_pixels} pixels "
                f"({100.0 * end / num_pixels:.1f}%), "
                f"rate={rate:,.0f} pixels/s",
                flush=True,
            )

        if profile_every > 0 and batch_index % profile_every == 0:
            print(
                "  timing: "
                f"prepare={prepare_done - batch_start:.3f}s | "
                f"H2D={h2d_done - prepare_done:.3f}s | "
                f"model={model_done - h2d_done:.3f}s | "
                f"D2H={d2h_done - model_done:.3f}s | "
                f"write={write_done - d2h_done:.3f}s",
                flush=True,
            )

        # Release large per-batch tensors promptly.
        del batch_coords, rows, cols, x_np, x_cpu, x, x_input
        del encoded, output, output_np

    features_mm.flush()
    coords_mm.flush()

    elapsed = time.perf_counter() - total_start
    print(
        f"Feature extraction finished in {elapsed:.1f}s "
        f"({num_pixels / max(elapsed, 1e-9):,.0f} pixels/s).",
        flush=True,
    )
    return num_pixels


def build_recency_inputs(
    *,
    args: argparse.Namespace,
    checkpoint: dict,
    config: dict,
    image_paths: list[str],
    num_input_times: int,
    device: torch.device,
) -> tuple[bool, list[str] | None, float | None, torch.Tensor | None]:
    """Resolve recency settings from CLI and checkpoint metadata."""
    input_channels = int(config.get("input_channels", 1))
    checkpoint_use_recency = bool(checkpoint.get("use_recency_input", False))
    use_recency_input = (
        bool(args.use_recency_input)
        or checkpoint_use_recency
        or input_channels == 2
    )

    if not use_recency_input:
        if input_channels != 1:
            raise ValueError(
                f"Checkpoint expects input_channels={input_channels}, but no supported "
                "input construction was selected."
            )
        print("Using baseline input [B, T, 1].")
        return False, None, None, None

    if input_channels != 2:
        raise ValueError(
            "Recency input is enabled, but checkpoint model_config input_channels "
            f"is {input_channels}, not 2."
        )

    if args.recency_dates is not None:
        recency_dates = [str(value) for value in args.recency_dates]
    else:
        try:
            recency_dates = [str(value) for value in parse_dates(image_paths)]
        except ValueError as exc:
            raise ValueError(
                "Dates could not be parsed from the image filenames. Provide one "
                "date per image using --recency-dates."
            ) from exc

    if len(recency_dates) != num_input_times:
        raise ValueError(
            f"Number of recency dates ({len(recency_dates)}) does not match "
            f"the number of input images ({num_input_times})."
        )

    if args.half_life_days is not None:
        half_life_days = float(args.half_life_days)
    elif checkpoint.get("half_life_days") is not None:
        half_life_days = float(checkpoint["half_life_days"])
    else:
        half_life_days = 12.0
        print(
            "Warning: half_life_days was not found in the CLI or checkpoint; "
            "using 12.0 days.",
            flush=True,
        )

    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")

    recency_matrix = make_causal_recency_matrix(
        dates=recency_dates,
        half_life_days=half_life_days,
        device=device,
    )

    print("Using recency-weighted input [B, T, 2].")
    print("Input dates:", recency_dates)
    print("Half-life days:", half_life_days)
    print("Recency matrix shape:", tuple(recency_matrix.shape))
    print(recency_matrix.detach().cpu().numpy())

    return True, recency_dates, half_life_days, recency_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all temporal TCN features from one SAR sequence in one pass "
            "and save features.npy with shape [N,T_out,C]."
        )
    )
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Accepted for compatibility with older commands but ignored. This "
            "version does not use DataLoader workers."
        ),
    )
    parser.add_argument("--p-lower", type=float, default=1.0)
    parser.add_argument("--p-upper", type=float, default=99.0)
    parser.add_argument("--normalization", default="zscore")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--include-first",
        action="store_true",
        help=(
            "Also save h0. By default h0 is excluded, so T input images produce "
            "T-1 feature slices corresponding to image dates 1..T-1."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Storage dtype. float16 halves D2H traffic and output size.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin each input batch before asynchronous CUDA transfer.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA float16 autocast for model inference.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every this many batches; 0 disables periodic logs.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help=(
            "Synchronize CUDA and print stage timings every this many batches. "
            "Use 0 normally because synchronization reduces throughput."
        ),
    )

    parser.add_argument("--use-recency-input", action="store_true")
    parser.add_argument("--half-life-days", type=float, default=None)
    parser.add_argument(
        "--recency-dates",
        nargs="+",
        default=None,
        help="One YYYY-MM-DD value per input image when dates cannot be parsed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers != 0:
        print(
            f"Note: --num-workers={args.num_workers} is ignored; direct extraction "
            "does not use a DataLoader.",
            flush=True,
        )
    if not (0.0 <= args.p_lower < args.p_upper <= 100.0):
        raise ValueError("Require 0 <= p-lower < p-upper <= 100.")

    device = resolve_device(args.device)
    output_np_dtype, output_torch_dtype = select_output_dtype(args.output_dtype)

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    image_paths = collect_geotiff_paths(args.images)
    if len(image_paths) < 2 and not args.include_first:
        raise ValueError(
            "At least two input images are required when the first feature is excluded."
        )

    print("Reading SAR stack...", flush=True)
    stack, valid_mask, _ = read_sar_stack(
        image_paths,
        lower=args.p_lower,
        upper=args.p_upper,
        normalization=args.normalization,
    )

    if stack.ndim != 3:
        raise ValueError(f"Expected SAR stack [T,H,W], got {stack.shape}")
    if stack.shape[0] != len(image_paths):
        raise ValueError(
            f"Stack has T={stack.shape[0]} but {len(image_paths)} image paths were found."
        )

    coords = valid_pixel_coords(valid_mask)
    del valid_mask
    gc.collect()

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords [N,2], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("No valid pixels were found across the SAR sequence.")

    print(f"Stack shape: {stack.shape}, dtype={stack.dtype}")
    print(f"Valid pixels: {coords.shape[0]:,}")

    # Load on CPU first to avoid a temporary duplicate checkpoint on the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain 'model_config' and 'model_state_dict'."
        )
    config = checkpoint["model_config"]

    model = SARPixelTCN(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_recency_input, recency_dates, half_life_days, recency_matrix = (
        build_recency_inputs(
            args=args,
            checkpoint=checkpoint,
            config=config,
            image_paths=image_paths,
            num_input_times=int(stack.shape[0]),
            device=device,
        )
    )

    feature_dim = int(config["num_channels"][-1])
    num_pixels = int(coords.shape[0])
    num_input_times = int(stack.shape[0])
    exclude_first = not args.include_first
    num_output_times = num_input_times - 1 if exclude_first else num_input_times

    expected_gib = estimate_output_size_gib(
        num_pixels,
        num_output_times,
        feature_dim,
        output_np_dtype,
    )
    print(
        f"Output feature shape: ({num_pixels}, {num_output_times}, {feature_dim})"
    )
    print(
        f"Estimated features.npy size: {expected_gib:.2f} GiB "
        f"({args.output_dtype})"
    )

    features_path = output_dir / "features.npy"
    coords_path = output_dir / "coords.npy"
    metadata_path = output_dir / "metadata.npz"

    # Write temporary files first so an interrupted job is not mistaken for a
    # complete extraction by downstream scripts.
    features_tmp = output_dir / "features.tmp.npy"
    coords_tmp = output_dir / "coords.tmp.npy"

    for path in (features_tmp, coords_tmp):
        if path.exists():
            path.unlink()

    features_mm = np.lib.format.open_memmap(
        features_tmp,
        mode="w+",
        dtype=output_np_dtype,
        shape=(num_pixels, num_output_times, feature_dim),
    )
    coords_mm = np.lib.format.open_memmap(
        coords_tmp,
        mode="w+",
        dtype=np.int32,
        shape=(num_pixels, 2),
    )

    written = extract_all_time_features_direct(
        model=model,
        stack=stack,
        coords=coords,
        device=device,
        features_mm=features_mm,
        coords_mm=coords_mm,
        batch_size=args.batch_size,
        recency_matrix=recency_matrix,
        exclude_first=exclude_first,
        output_torch_dtype=output_torch_dtype,
        pin_memory=args.pin_memory,
        amp=args.amp,
        log_every=args.log_every,
        profile_every=args.profile_every,
    )

    # Close memmaps before atomically replacing final paths.
    del features_mm, coords_mm
    gc.collect()

    os.replace(features_tmp, features_path)
    os.replace(coords_tmp, coords_path)

    all_dates = (
        [str(value) for value in recency_dates]
        if recency_dates is not None
        else [str(value) for value in parse_dates(image_paths)]
    )
    feature_dates = all_dates[1:] if exclude_first else all_dates

    np.savez(
        metadata_path,
        height=np.array(stack.shape[1], dtype=np.int64),
        width=np.array(stack.shape[2], dtype=np.int64),
        image_paths=np.asarray(image_paths),
        reference_path=np.array(image_paths[0]),
        input_dates=np.asarray(all_dates),
        feature_dates=np.asarray(feature_dates),
        exclude_first=np.array(exclude_first),
        num_input_times=np.array(num_input_times, dtype=np.int64),
        num_output_times=np.array(num_output_times, dtype=np.int64),
        num_features=np.array(written, dtype=np.int64),
        feature_dim=np.array(feature_dim, dtype=np.int64),
        feature_shape=np.asarray(
            [num_pixels, num_output_times, feature_dim], dtype=np.int64
        ),
        feature_dtype=np.array(args.output_dtype),
        use_recency_input=np.array(use_recency_input),
        input_channels=np.array(int(config.get("input_channels", 1)), dtype=np.int64),
        recency_dates=np.asarray([] if recency_dates is None else recency_dates),
        half_life_days=np.array(
            -1.0 if half_life_days is None else half_life_days,
            dtype=np.float64,
        ),
        recency_matrix=np.asarray(
            []
            if recency_matrix is None
            else recency_matrix.detach().cpu().numpy()
        ),
        p_lower=np.array(args.p_lower, dtype=np.float64),
        p_upper=np.array(args.p_upper, dtype=np.float64),
    )

    print(f"Saved features: {features_path}")
    print(f"Saved coords:   {coords_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"Final features shape: ({num_pixels}, {num_output_times}, {feature_dim})"
    )


if __name__ == "__main__":
    main()
