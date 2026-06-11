import argparse
from pathlib import Path
import wandb
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import SARPixelsDataset
    from .model import SARPixelTCN, model_config
    from .utils import (
        collect_geotiff_paths,
        make_tile_split,
        parse_channels,
        read_sar_stack,
        save_json,
        set_seed,
        parse_dates,
        make_causal_recency_matrix,
        add_recency_weighted_channel,
    )
except ImportError:
    from dataset import SARPixelsDataset
    from model import SARPixelTCN, model_config
    from utils import (
        collect_geotiff_paths,
        make_tile_split,
        parse_channels,
        read_sar_stack,
        save_json,
        set_seed,
        parse_dates,
        make_causal_recency_matrix,
        add_recency_weighted_channel,
    )

import dataset
print("Dataset module file:", dataset.__file__, flush=True)

def train_one_epoch(model, loader, criterion, optimizer, device, recency_matrix=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x in loader:
        x = x.to(device, non_blocking=True)  # [B, T, 1]/[B, T, 2] if using recency weighted
        target = x.permute(0, 2, 1).contiguous()  # [B, 1, T]
        x_input = add_recency_weighted_channel(x, recency_matrix)
        pred, _ = model(x_input)  # pred: [B, 1, T]
        loss = criterion(pred[:, :, :-1], target[:, :, 1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, recency_matrix=None):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x in loader:
        x = x.to(device)  # [B, T, 1]
        # target remains raw VH
        target = x.permute(0, 2, 1).contiguous()  # [B, 1, T]
        # model input can be [raw VH, recency-weighted VH]
        x_input = add_recency_weighted_channel(x, recency_matrix)
        pred, _ = model(x_input)  # pred: [B, 1, T]
        # original next-step prediction loss
        loss = criterion(pred[:, :, :-1], target[:, :, 1:])

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_loss,
    image_paths,
    stack_shape,
    use_recency_input=False,
    recency_dates=None,
    half_life_days=None,
):
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_config": model_config(model),
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "image_paths": image_paths,
        "stack_shape": stack_shape,
        "use_recency_input": use_recency_input,
        "recency_dates": recency_dates,
        "half_life_days": half_life_days,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Self-supervised causal TCN training for SAR pixel time series.")
    parser.add_argument("--images", nargs="+", required=True, help="GeoTIFF files, directories, or glob patterns.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and split metadata.")
    parser.add_argument("--channels", default="32,32,32", help="Comma-separated TCN hidden channels.")
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["l1", "smooth_l1"], default="l1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--p-lower", type=float, default=1.0)
    parser.add_argument("--p-upper", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-recency-input", action="store_true", help="Use additional causal recency-weighted VH channel as model input.")
    parser.add_argument("--half-life-days", type=float, default=6.0, help="Half-life in days for causal recency weighting.")
    parser.add_argument("--recency-dates", nargs="+", default=None,
        help=(
            "Optional list of dates corresponding to input images, e.g. "
            "2020-08-04 2020-08-10 2020-08-16. "
            "If not provided, dates are parsed from image filenames."
        )
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="deep-wetlands", help="W&B project name.")
    parser.add_argument("--wandb-entity", default="xzuo-kth-kth-royal-institute-of-technology", help="W&B entity/team name.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name.")
    
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_geotiff_paths(args.images)
    stack, valid_mask, _ = read_sar_stack(image_paths, lower=args.p_lower, upper=args.p_upper)
    if stack.shape[0] < 2:
        raise ValueError("Need at least two time steps for next-step prediction.")

    if args.use_recency_input:
        if args.recency_dates is not None:
            recency_dates = args.recency_dates
        else:
            recency_dates = parse_dates(image_paths)

        if len(recency_dates) != stack.shape[0]:
            raise ValueError(
                f"Number of recency dates ({len(recency_dates)}) "
                f"must match number of time steps ({stack.shape[0]})."
            )

        recency_matrix = make_causal_recency_matrix(
            dates=recency_dates,
            half_life_days=args.half_life_days,
            device=args.device
        )

        input_channels = 2

        print("Using recency-weighted input channel.")
        print("Recency dates:", recency_dates)
        print("Half-life days:", args.half_life_days)
        print("Recency matrix:")
        print(recency_matrix.detach().cpu().numpy())

    else:
        recency_dates = None
        recency_matrix = None
        input_channels = 1

    split_coords = make_tile_split(
        valid_mask,
        tile_size=args.tile_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    np.savez_compressed(
        output_dir / "tile_split_coords.npz",
        train=split_coords["train"],
        val=split_coords["val"],
    )

    train_dataset = SARPixelsDataset(stack, split_coords["train"])
    val_dataset = SARPixelsDataset(stack, split_coords["val"])
    print("Loading data...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    print("Finish loading")
    model = SARPixelTCN(
        num_channels=parse_channels(args.channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        input_channels=input_channels,
    ).to(args.device)

    criterion = torch.nn.L1Loss() if args.loss == "l1" else torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    best_path = output_dir / "best_tcn_encoder.pt"
    metadata = {
        "image_paths": image_paths,
        "stack_shape": list(stack.shape),
        "valid_pixels": int(valid_mask.sum()),
        "split_counts": {key: int(value.shape[0]) for key, value in split_coords.items()},
        "args": vars(args),
        "use_recency_input": args.use_recency_input,
        "input_channels": input_channels,
        "recency_dates": recency_dates,
        "half_life_days": args.half_life_days if args.use_recency_input else None,
    }
    save_json(output_dir / "training_metadata.json", metadata)
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "stack_shape": list(stack.shape),
                "valid_pixels": int(valid_mask.sum()),
                "split_counts": {key: int(value.shape[0]) for key, value in split_coords.items()},
                "num_channels": parse_channels(args.channels),
                "model_config": model_config(model),
            },
        )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device, recency_matrix=recency_matrix)
        val_loss = evaluate(model, val_loader, criterion, args.device, recency_matrix=recency_matrix)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_json(output_dir / "training_history.json", history)

        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_loss,
                image_paths,
                list(stack.shape),
                use_recency_input=args.use_recency_input,
                recency_dates=recency_dates,
                half_life_days=args.half_life_days if args.use_recency_input else None,
            )

    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.use_wandb:
        wandb.log({
            "best_val_loss": best_val_loss,
        })
        wandb.finish()
    save_json(output_dir / "test_metrics.json", {"best_val_loss": best_val_loss})
    print(f"best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

