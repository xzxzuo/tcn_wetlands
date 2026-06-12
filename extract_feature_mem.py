import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import SARPixelsDataset
    from .model import SARPixelTCN
    from .utils import collect_geotiff_paths, read_sar_stack, valid_pixel_coords, parse_dates, make_causal_recency_matrix, add_recency_weighted_channel
except ImportError:
    from dataset import SARPixelsDataset
    from model import SARPixelTCN
    from utils import collect_geotiff_paths, read_sar_stack, valid_pixel_coords, parse_dates, make_causal_recency_matrix, add_recency_weighted_channel


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--p-lower", type=float, default=1.0)
    parser.add_argument("--p-upper", type=float, default=99.0)
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
    stack, valid_mask, _ = read_sar_stack(image_paths, lower=args.p_lower, upper=args.p_upper)
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

    features_mm = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32, shape=(N, C))
    coords_mm = np.lib.format.open_memmap(coords_path,mode="w+",dtype=np.int32,shape=(N, 2))

    dataset = SARPixelsDataset(stack, coords, return_coords=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    written = extract_features_to_memmap(
        model=model,
        loader=loader,
        device=args.device,
        features_mm=features_mm,
        coords_mm=coords_mm,
        feature_mode=args.feature_mode,
        time_index=args.time_index,
        recency_matrix=recency_matrix,
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