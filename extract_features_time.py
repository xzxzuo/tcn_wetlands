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
def extract_features(
    model,
    loader,
    device,
    feature_mode="last",
    time_index=None,
    recency_matrix=None,
):
    """
    Extract TCN features.

    If recency_matrix is not None:
        input becomes [raw VH, recency-weighted VH], shape [B, T, 2].
    """
    model.eval()
    feature_batches = []
    coord_batches = []

    for x, coords in loader:
        x = x.to(device)  # [B, T, 1]
        # [B, T, 2] if using recency matrix
        x_input = add_recency_weighted_channel(x, recency_matrix)
        features = model.encode(x_input)  # [B, C, T]
        if feature_mode == "mean":
            out_features = features.mean(dim=2)  # [B, C]
        elif feature_mode == "last":
            out_features = features[:, :, -1]  # [B, C]
        elif feature_mode == "time":
            if time_index is None:
                raise ValueError("feature_mode='time' requires --time-index.")

            T = features.shape[2]
            idx = time_index

            if idx < 0:
                idx = T + idx

            if idx < 0 or idx >= T:
                raise ValueError(f"time_index {time_index} is out of range for T={T}.")

            out_features = features[:, :, idx]  # [B, C]

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        feature_batches.append(out_features.cpu().numpy().astype(np.float32))
        coord_batches.append(coords.numpy().astype(np.int64))

    return np.concatenate(feature_batches, axis=0), np.concatenate(coord_batches, axis=0)

def main(): 
    parser = argparse.ArgumentParser(description="Extract temporal TCN features for all valid SAR pixels.") 
    parser.add_argument("--images", nargs="+", required=True, help="GeoTIFF files, directories, or glob patterns.") 
    parser.add_argument("--checkpoint", required=True, help="Path to best_tcn_encoder.pt.") 
    parser.add_argument("--output", required=True, help="Output .npz file for features and coordinates.") 
    parser.add_argument("--batch-size", type=int, default=4096) 
    parser.add_argument("--num-workers", type=int, default=2) 
    parser.add_argument("--p-lower", type=float, default=1.0) 
    parser.add_argument("--p-upper", type=float, default=99.0) 
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument( "--feature-mode", choices=["mean", "last", "time"], default="last",
        help="Feature extraction mode: mean, last, or one selected time step."
    )
    parser.add_argument("--time-index", type=int, default=None,
        help="0-based time index used when --feature-mode time. Example: 0=h1, 1=h2, -1=hT."
    )
    parser.add_argument("--use-recency-input", action="store_true",
        help="Use additional causal recency-weighted VH channel during feature extraction."
    )
    parser.add_argument("--half-life-days", type=float, default=None,
        help="Half-life in days. If not provided, try to read from checkpoint."
    )
    parser.add_argument("--recency-dates", nargs="+", default=None,
        help=(
            "Optional date list matching input images. "
            "Example: --recency-dates 2020-08-04 2020-08-10 2020-08-16"
        )
    ) 
    args = parser.parse_args() 

    image_paths = collect_geotiff_paths(args.images) 
    stack, valid_mask, _ = read_sar_stack(image_paths, lower=args.p_lower, upper=args.p_upper) 
    coords = valid_pixel_coords(valid_mask) 
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["model_config"]
    input_channels = config.get("input_channels", 1)
    checkpoint_use_recency = checkpoint.get("use_recency_input", False)
    use_recency_input = args.use_recency_input or checkpoint_use_recency or (input_channels == 2)

    if use_recency_input:
        if input_channels != 2:
            raise ValueError(
                "You requested recency input, but checkpoint model_config input_channels is not 2. "
                "This checkpoint may be a baseline model trained with [B,T,1]."
            )

        T_current = stack.shape[0]

        if args.recency_dates is not None:
            recency_dates = args.recency_dates
        else:
            # For extraction/testing, dates should correspond to CURRENT input images,
            # not necessarily the training images stored in the checkpoint.
            try:
                recency_dates = parse_dates(image_paths)
            except ValueError:
                checkpoint_dates = checkpoint.get("recency_dates", None)
                if checkpoint_dates is not None and len(checkpoint_dates) == T_current:
                    recency_dates = checkpoint_dates
                else:
                    raise ValueError(
                        "Cannot determine recency dates for current input images. "
                        "Your checkpoint dates do not match current T. "
                        "Please provide --recency-dates manually."
                    )

        if len(recency_dates) != T_current:
            raise ValueError(
                f"Number of recency_dates={len(recency_dates)} does not match "
                f"number of current input time steps T={T_current}. "
                "Use --recency-dates matching the current input images."
            )

        if len(recency_dates) != stack.shape[0]:
            raise ValueError(
                f"Number of recency_dates={len(recency_dates)} does not match "
                f"number of time steps T={stack.shape[0]}."
            )

        if args.half_life_days is not None:
            half_life_days = args.half_life_days
        elif checkpoint.get("half_life_days", None) is not None:
            half_life_days = checkpoint["half_life_days"]
        else:
            raise ValueError(
                "half_life_days is missing. Provide --half-life-days or save it in checkpoint."
            )

        recency_matrix = make_causal_recency_matrix(
            dates=recency_dates,
            half_life_days=half_life_days,
            device=args.device,
        )

        print("Using recency input during feature extraction.")
        print("Current input T:", T_current)
        print("Recency dates used for current input:", recency_dates)
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
    dataset = SARPixelsDataset(stack, coords, return_coords=True) 
    loader = DataLoader( dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"), ) 
    features, coords = extract_features(
        model,
        loader,
        args.device,
        feature_mode=args.feature_mode,
        time_index=args.time_index,
        recency_matrix=recency_matrix,
    ) 
    output = Path(args.output) 
    output.parent.mkdir(parents=True, exist_ok=True) 
    np.savez_compressed(
        output,
        features=features,  # [N, C]
        coords=coords,      # [N, 2]
        height=np.array(stack.shape[1], dtype=np.int64),
        width=np.array(stack.shape[2], dtype=np.int64),
        image_paths=np.array(image_paths),
        reference_path=np.array(image_paths[0]),
        feature_mode=np.array(args.feature_mode),
        time_index=np.array(-999 if args.time_index is None else args.time_index, dtype=np.int64),
        use_recency_input=np.array(use_recency_input),
        recency_dates=np.array([] if recency_dates is None else recency_dates),
        half_life_days=np.array(-1.0 if half_life_days is None else half_life_days),
        recency_matrix=np.array(
            [] if recency_matrix is None else recency_matrix.detach().cpu().numpy()
        ),
    )
    print(f"saved {features.shape[0]} feature vectors with dimension {features.shape[1]} to {output}") 
    
if __name__ == "__main__": 
    main()
    
