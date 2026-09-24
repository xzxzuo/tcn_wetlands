import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SARPixelTCN
from utils import collect_geotiff_paths, read_sar_stack, valid_pixel_coords, parse_dates, make_causal_recency_matrix, add_recency_weighted_channel, fit_histogram_mapping, sample_sar_sequence, apply_histogram_mapping_to_image

@torch.inference_mode()
def extract_all_time_features_direct(
    model,
    stack,
    coords,
    device,
    features_mm,
    coords_mm,
    batch_size,
    recency_matrix=None,
    exclude_first=True,
):
    """
    Run the full SAR sequence once and save temporal features for all dates.

    stack:
        [T, H, W]

    coords:
        [N, 2]

    features_mm:
        [N, T-1, C] if exclude_first=True
        [N, T, C] otherwise
    """
    model.eval()

    num_pixels = len(coords)

    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)

        batch_coords = coords[start:end]
        rows = batch_coords[:, 0]
        cols = batch_coords[:, 1]

        # [T, B] -> [B, T, 1]
        x_np = np.ascontiguousarray(
            stack[:, rows, cols].T,
            dtype=np.float32,
        )

        x = torch.from_numpy(x_np).unsqueeze(-1).to(
            device,
            non_blocking=True,
        )

        x_input = add_recency_weighted_channel(
            x,
            recency_matrix,
        )

        # [B, C, T]
        features = model.encode(x_input)

        if exclude_first:
            features = features[:, :, 1:]

        # [B, C, T_out] -> [B, T_out, C]
        output = features.permute(0, 2, 1).contiguous()

        features_mm[start:end] = output.float().cpu().numpy()
        coords_mm[start:end] = batch_coords.astype(
            np.int32,
            copy=False,
        )

        if (
            start == 0
            or end == num_pixels
            or (start // batch_size) % 10 == 0
        ):
            print(
                f"written {end}/{num_pixels} pixels "
                f"({100.0 * end / num_pixels:.1f}%)",
                flush=True,
            )

        del x, x_input, features, output

    features_mm.flush()
    coords_mm.flush()

    return num_pixels

@torch.inference_mode()
def extract_features_direct(
    model,
    stack,
    coords,
    device,
    features_mm,
    coords_mm,
    batch_size=65536,
    feature_mode="last",
    time_index=None,
    recency_matrix=None,
):
    """
    Extract features directly from an in-memory SAR stack.

    stack:
        [T, H, W], NumPy array

    coords:
        [N, 2], rows and columns
    """
    model.eval()

    n_pixels = len(coords)

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)

        batch_coords = coords[start:end]
        rows = batch_coords[:, 0]
        cols = batch_coords[:, 1]

        # stack[:, rows, cols]: [T, B]
        # transpose: [B, T]
        x_np = stack[:, rows, cols].T

        # [B, T, 1]
        x = torch.from_numpy(
            np.ascontiguousarray(x_np, dtype=np.float32)
        ).unsqueeze(-1)

        x = x.to(
            device,
            non_blocking=True,
        )

        x_input = add_recency_weighted_channel(
            x,
            recency_matrix,
        )

        features = model.encode(x_input)  # [B, C, T]

        if feature_mode == "mean":
            out_features = features.mean(dim=2)

        elif feature_mode == "last":
            out_features = features[:, :, -1]

        elif feature_mode == "time":
            if time_index is None:
                raise ValueError(
                    "feature_mode='time' requires --time-index"
                )

            t_length = features.shape[2]
            idx = time_index if time_index >= 0 else t_length + time_index

            if idx < 0 or idx >= t_length:
                raise ValueError(
                    f"time_index {time_index} out of range for T={t_length}"
                )

            out_features = features[:, :, idx]

        else:
            raise ValueError(
                f"Unknown feature_mode: {feature_mode}"
            )

        # Copy only the selected [B, C] output back to CPU.
        out_np = out_features.float().cpu().numpy()

        features_mm[start:end] = out_np
        coords_mm[start:end] = batch_coords.astype(
            np.int32,
            copy=False,
        )

        if start == 0 or end == n_pixels or start // batch_size % 20 == 0:
            print(
                f"written {end}/{n_pixels} pixels "
                f"({100.0 * end / n_pixels:.1f}%)",
                flush=True,
            )

    features_mm.flush()
    coords_mm.flush()

    return n_pixels

@torch.no_grad()
def extract_features_to_memmap(
    model,
    loader,
    device,
    features_mm,
    coords_mm,
    feature_mode="last",
    time_index=None,
    recency_matrix=None,
):
    model.eval()

    offset = 0

    for batch_idx, batch in enumerate(loader):
        x, coords = batch
        x = x.to(device, non_blocking=True)  # [B, T, 1]

        x_input = add_recency_weighted_channel(x, recency_matrix)

        features = model.encode(x_input)  # [B, C, T]

        if feature_mode == "mean":
            out_features = features.mean(dim=2)  # [B, C]

        elif feature_mode == "last":
            out_features = features[:, :, -1]  # [B, C]

        elif feature_mode == "time":
            if time_index is None:
                raise ValueError("feature_mode='time' requires --time-index")

            T = features.shape[2]
            idx = time_index
            if idx < 0:
                idx = T + idx

            if idx < 0 or idx >= T:
                raise ValueError(f"time_index {time_index} out of range for T={T}")

            out_features = features[:, :, idx]  # [B, C]

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        out_np = out_features.detach().cpu().numpy().astype(np.float32)
        coords_np = coords.numpy().astype(np.int32)

        bsz = out_np.shape[0]

        features_mm[offset:offset + bsz] = out_np
        coords_mm[offset:offset + bsz] = coords_np

        offset += bsz

        if batch_idx % 50 == 0:
            print(f"batch {batch_idx}, written {offset} pixels")

    features_mm.flush()
    coords_mm.flush()

    return offset


def main():
    parser = argparse.ArgumentParser(description="Extract TCN features using memmap to avoid RAM OOM.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument(
        "--histogram-train-images",
        nargs="+",
        default=None,
        help=(
            "Training SAR files/directories used as the "
            "reference histogram."
        ),
    )

    parser.add_argument(
        "--histogram-alpha",
        type=float,
        default=0,
        help=(
            "Histogram mapping strength. "
            "0 disables mapping; 1 performs full mapping."
        ),
    )

    parser.add_argument(
        "--histogram-quantiles",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--histogram-max-samples",
        type=int,
        default=2_000_000,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--p-lower", type=float, default=1.0)
    parser.add_argument("--p-upper", type=float, default=99.0)
    parser.add_argument("--train-mean", type=str, default=None)
    parser.add_argument("--train-std", type=str, default=None)
    parser.add_argument("--normalization", type=str, default=None)
    parser.add_argument("--feature-mode", choices=["mean", "last", "time"], default="last")
    parser.add_argument("--time-index", type=int, default=None, help="Used only when --feature-mode time. 0=h1, 1=h2, -1=hT.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-recency-input", action="store_true", help="Use additional causal recency-weighted VH channel during feature extraction.")
    parser.add_argument("--half-life-days", type=float, default=None, help="Half-life in days. If not provided, try to read from checkpoint.")
    parser.add_argument("--recency-dates", nargs="+", default=None,
        help=(
            "Dates corresponding to current input images. "
            "e.x., For Orebro 5-image test, provide exactly 5 dates if filenames cannot be parsed."
        )
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_geotiff_paths(args.images)
    train_mean = None
    train_std = None
    if args.normalization == "train_zscore":
        if args.train_mean is None or args.train_std is None:
            parser.error(
                "--normalization train_zscore requires "
                "--train-mean and --train-std."
            )
        mean_array = np.asarray(
            np.load(args.train_mean),
            dtype=np.float32,
        )
        std_array = np.asarray(
            np.load(args.train_std),
            dtype=np.float32,
        )
        if mean_array.size != 1:
            raise ValueError(
                "--train-mean must contain one scalar SAR mean, "
                f"but got shape {mean_array.shape}."
            )
        if std_array.size != 1:
            raise ValueError(
                "--train-std must contain one scalar SAR std, "
                f"but got shape {std_array.shape}."
            )
        train_mean = mean_array.item()
        train_std = std_array.item()
        if not np.isfinite(train_mean):
            raise ValueError("Training SAR mean is not finite.")
        if not np.isfinite(train_std) or train_std <= 0:
            raise ValueError(
                f"Training SAR std must be positive, got {train_std}."
            )
        print(
            "Loaded training SAR statistics: "
            f"mean={train_mean:.6f}, "
            f"std={train_std:.6f}"
        )

    histogram_mapping = None
    if args.histogram_alpha > 0:
        if args.histogram_train_images is None:
            parser.error(
                "--histogram-alpha > 0 requires "
                "--histogram-train-images."
            )
        if not 0.0 <= args.histogram_alpha <= 1.0:
            parser.error(
                "--histogram-alpha must be between 0 and 1."
            )
        train_image_paths = collect_geotiff_paths(
            args.histogram_train_images
        )
        print(
            f"Histogram reference images: "
            f"{len(train_image_paths)}"
        )
        print(
            f"Histogram target images: "
            f"{len(image_paths)}"
        )

        train_raw_stack, train_valid_mask, _ = read_sar_stack(
            paths=train_image_paths,
            lower=args.p_lower,
            upper=args.p_upper,
            normalization=None,
            verbose=True,
        )

        test_raw_stack, test_valid_mask, _ = read_sar_stack(
            paths=image_paths,
            lower=args.p_lower,
            upper=args.p_upper,
            normalization=None,
            verbose=True,
        )

        histogram_mapping = fit_histogram_mapping(
            train_stack=train_raw_stack,
            test_stack=test_raw_stack,
            train_valid_mask=train_valid_mask,
            test_valid_mask=test_valid_mask,
            n_quantiles=args.histogram_quantiles,
            max_samples=args.histogram_max_samples,
        )

        mapping_path = output_dir / "histogram_mapping.npz"

        np.savez(
            mapping_path,
            test_knots=histogram_mapping["test_knots"],
            train_knots=histogram_mapping["train_knots"],
            alpha=np.float32(args.histogram_alpha),
        )

        print(f"Saved histogram mapping to: {mapping_path}")
        print(
            "Histogram mapping strength: "
            f"alpha={args.histogram_alpha}"
        )

    stack, valid_mask, profile = read_sar_stack(
        paths=image_paths,
        lower=args.p_lower,
        upper=args.p_upper,
        normalization=args.normalization,
        train_mean=train_mean,
        train_std=train_std,
        histogram_mapping=histogram_mapping,
        histogram_alpha=args.histogram_alpha,
        verbose=True,
    )

    coords = valid_pixel_coords(valid_mask)

    print(f"stack shape: {stack.shape}")
    print(f"valid pixels: {len(coords)}")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["model_config"]

    input_channels = config.get("input_channels", 1)
    # Detect whether this checkpoint expects recency input.
    checkpoint_use_recency = checkpoint.get("use_recency_input", False)
    use_recency_input = args.use_recency_input or checkpoint_use_recency or (input_channels == 2)

    if use_recency_input:
        if input_channels != 2:
            raise ValueError(
                "Recency input is requested, but checkpoint model_config input_channels is not 2. "
                "This checkpoint may be a baseline [B,T,1] model."
            )
        T_current = stack.shape[0]
        if args.recency_dates is not None:
            recency_dates = args.recency_dates
        else:
            try:
                recency_dates = parse_dates(image_paths)
            except ValueError:
                raise ValueError(
                    "Cannot parse dates from current image filenames. "
                    "Please provide --recency-dates manually."
                )

        if len(recency_dates) != T_current:
            raise ValueError(
                f"Number of recency_dates={len(recency_dates)} does not match "
                f"current input time steps T={T_current}. "
                "For testing, recency_dates must match the test input images, "
                "not the training images."
            )

        if args.half_life_days is not None:
            half_life_days = args.half_life_days
        elif checkpoint.get("half_life_days", None) is not None:
            half_life_days = checkpoint["half_life_days"]
        else:
            half_life_days = 6.0
            print("Warning: half_life_days not found in checkpoint. Using default 6.0.")

        recency_matrix = make_causal_recency_matrix(
            dates=recency_dates,
            half_life_days=half_life_days,
            device=args.device,
        )

        print("Using recency input during feature extraction.")
        # print("Current test region: Orebro lan")
        print("Current input T:", T_current)
        print("Current recency dates:", recency_dates)
        print("Half-life days:", half_life_days)
        print("Recency matrix shape:", tuple(recency_matrix.shape))
        print(recency_matrix.detach().cpu().numpy())

    else:
        recency_dates = None
        half_life_days = None
        recency_matrix = None
        print("Using baseline input during feature extraction: [B, T, 1].")

    model = SARPixelTCN(**config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Feature dimension C
    C = int(config["num_channels"][-1])
    N = int(len(coords))

    features_path = output_dir / "features.npy"
    coords_path = output_dir / "coords.npy"
    metadata_path = output_dir / "metadata.npz"

    T = stack.shape[0]
    T_out = T - 1

    features_mm = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, T_out, C),
    )
    coords_mm = np.lib.format.open_memmap(coords_path,mode="w+",dtype=np.int32,shape=(N, 2))

    written = extract_all_time_features_direct(
        model=model,
        stack=stack,
        coords=coords,
        device=args.device,
        features_mm=features_mm,
        coords_mm=coords_mm,
        batch_size=args.batch_size,
        recency_matrix=recency_matrix,
        exclude_first=True,
    )

    np.savez(
        metadata_path,
        height=np.array(stack.shape[1], dtype=np.int64),
        width=np.array(stack.shape[2], dtype=np.int64),
        image_paths=np.array(image_paths),
        reference_path=np.array(image_paths[0]),
        feature_mode=np.array(args.feature_mode),
        time_index=np.array(-999 if args.time_index is None else args.time_index),
        num_features=np.array(written, dtype=np.int64),
        feature_dim=np.array(C, dtype=np.int64),
        use_recency_input=np.array(use_recency_input),
        input_channels=np.array(input_channels, dtype=np.int64),
        recency_dates=np.array([] if recency_dates is None else recency_dates),
        half_life_days=np.array(-1.0 if half_life_days is None else half_life_days),
        recency_matrix=np.array(
            [] if recency_matrix is None else recency_matrix.detach().cpu().numpy()
        ),
    )

    print(f"saved features to {features_path}")
    print(f"saved coords to {coords_path}")
    print(f"saved metadata to {metadata_path}")
    print(f"features shape: {(N, C)}")


if __name__ == "__main__":
    main()