#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class FeatureBlock:
    time_index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def choose_sample_blocks(
    n_rows: int,
    n_times: int,
    samples_per_time: int,
    block_size: int,
    seed: int,
) -> list[FeatureBlock]:
    """Choose reproducible, non-overlapping contiguous blocks for each date."""
    if n_rows <= 0 or n_times <= 0:
        raise ValueError("n_rows and n_times must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    rng = np.random.default_rng(seed)
    all_blocks: list[FeatureBlock] = []
    n_available_blocks = int(np.ceil(n_rows / block_size))

    for time_index in range(n_times):
        if samples_per_time <= 0 or samples_per_time >= n_rows:
            selected_block_ids = np.arange(n_available_blocks, dtype=np.int64)
        else:
            n_selected_blocks = min(
                n_available_blocks,
                int(np.ceil(samples_per_time / block_size)),
            )
            selected_block_ids = rng.choice(
                n_available_blocks,
                size=n_selected_blocks,
                replace=False,
            )
            selected_block_ids.sort()

        remaining = n_rows if samples_per_time <= 0 else min(samples_per_time, n_rows)

        for block_id in selected_block_ids:
            if remaining <= 0:
                break

            start = int(block_id) * block_size
            end = min(start + block_size, n_rows)

            if samples_per_time > 0:
                end = min(end, start + remaining)

            if end > start:
                all_blocks.append(
                    FeatureBlock(
                        time_index=time_index,
                        start=start,
                        end=end,
                    )
                )
                remaining -= end - start

    return all_blocks


def load_block(
    features: np.ndarray,
    block: FeatureBlock,
) -> np.ndarray:
    """Load one [B, C] block as float32."""
    return np.asarray(
        features[block.start:block.end, block.time_index, :],
        dtype=np.float32,
    )


def compute_sample_mean_std(
    features: np.ndarray,
    blocks: list[FeatureBlock],
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute feature-wise mean/std over the selected training blocks."""
    total_count = 0
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None

    for block_idx, block in enumerate(blocks):
        x = load_block(features, block).astype(np.float64, copy=False)

        if total_sum is None:
            feature_dim = x.shape[1]
            total_sum = np.zeros(feature_dim, dtype=np.float64)
            total_sumsq = np.zeros(feature_dim, dtype=np.float64)

        total_sum += x.sum(axis=0)
        total_sumsq += np.square(x).sum(axis=0)
        total_count += x.shape[0]

        if block_idx == 0 or (block_idx + 1) % 50 == 0 or block_idx + 1 == len(blocks):
            print(
                f"[Mean/std] block {block_idx + 1}/{len(blocks)}, "
                f"vectors={total_count:,}",
                flush=True,
            )

    if total_sum is None or total_sumsq is None or total_count == 0:
        raise RuntimeError("No feature vectors were read.")

    mean = total_sum / total_count
    variance = total_sumsq / total_count - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    std[~np.isfinite(std)] = 1.0
    std[std < eps] = 1.0

    return mean.astype(np.float32), std.astype(np.float32), int(total_count)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def train_minibatch_kmeans(
    features: np.ndarray,
    blocks: list[FeatureBlock],
    mean: np.ndarray,
    std: np.ndarray,
    k: int,
    batch_size: int,
    epochs: int,
    seed: int,
    n_init: int,
) -> MiniBatchKMeans:
    """Train one MiniBatchKMeans using the selected feature blocks."""
    if k < 2:
        raise ValueError("k must be at least 2.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        n_init=n_init,
        max_no_improvement=50,
        reassignment_ratio=0.01,
        verbose=0,
    )

    rng = np.random.default_rng(seed)
    initialized = False

    for epoch in range(epochs):
        order = np.arange(len(blocks))
        rng.shuffle(order)

        seen = 0
        print(f"[KMeans] epoch {epoch + 1}/{epochs}", flush=True)

        for order_idx, block_idx in enumerate(order):
            block = blocks[int(block_idx)]
            x = load_block(features, block)
            x = standardize(x, mean, std)

            if x.shape[0] < k:
                continue

            model.partial_fit(x)
            initialized = True
            seen += x.shape[0]

            if order_idx == 0 or (order_idx + 1) % 50 == 0 or order_idx + 1 == len(order):
                print(
                    f"  block {order_idx + 1}/{len(order)}, "
                    f"vectors this epoch={seen:,}",
                    flush=True,
                )

    if not initialized:
        raise RuntimeError("MiniBatchKMeans was not initialized.")

    return model


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and save a global MiniBatchKMeans from one [N,T,C] "
            "TCN feature memmap."
        )
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=2)

    parser.add_argument(
        "--samples-per-time",
        type=int,
        default=10_000_000,
        help=(
            "Number of feature vectors sampled from each date. "
            "Use 0 to scan all rows from every date."
        ),
    )
    parser.add_argument(
        "--sample-block-size",
        type=int,
        default=65_536,
        help="Contiguous row-block size used for efficient memmap reads.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="MiniBatchKMeans internal batch size.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=3)
    args = parser.parse_args()

    features_path = args.features.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.is_file():
        raise FileNotFoundError(features_path)

    features = np.load(features_path, mmap_mode="r")
    if features.ndim != 3:
        raise ValueError(
            f"Expected features.npy with shape [N,T,C], got {features.shape}"
        )

    n_rows, n_times, feature_dim = map(int, features.shape)
    print(f"Feature file: {features_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"K: {args.k}")

    blocks = choose_sample_blocks(
        n_rows=n_rows,
        n_times=n_times,
        samples_per_time=args.samples_per_time,
        block_size=args.sample_block_size,
        seed=args.seed,
    )

    sampled_per_time = [0 for _ in range(n_times)]
    for block in blocks:
        sampled_per_time[block.time_index] += block.size

    total_sampled = int(sum(sampled_per_time))
    print(f"Selected blocks: {len(blocks)}")
    print(f"Sampled vectors per date: {sampled_per_time}")
    print(f"Total sampled vectors: {total_sampled:,}")

    print("[1/3] Computing global feature mean/std from sampled training vectors...")
    mean, std, stats_count = compute_sample_mean_std(features, blocks)
    np.save(output_dir / "global_feature_mean.npy", mean)
    np.save(output_dir / "global_feature_std.npy", std)

    print("[2/3] Training global MiniBatchKMeans...")
    kmeans = train_minibatch_kmeans(
        features=features,
        blocks=blocks,
        mean=mean,
        std=std,
        k=args.k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        n_init=args.n_init,
    )

    with (output_dir / "global_kmeans.pkl").open("wb") as file:
        pickle.dump(kmeans, file)

    centers_standardized = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    centers_original = centers_standardized * std[None, :] + mean[None, :]

    np.save(
        output_dir / "global_cluster_centers_standardized.npy",
        centers_standardized,
    )
    np.save(
        output_dir / "global_cluster_centers_original.npy",
        centers_original.astype(np.float32),
    )

    print("[3/3] Saving training summary...")
    summary = {
        "method": "sampled global MiniBatchKMeans over all dates",
        "features_path": str(features_path),
        "feature_shape": [n_rows, n_times, feature_dim],
        "feature_dtype": str(features.dtype),
        "k": int(args.k),
        "samples_per_time_requested": int(args.samples_per_time),
        "samples_per_time_actual": [int(value) for value in sampled_per_time],
        "total_sampled_vectors": total_sampled,
        "statistics_vector_count": int(stats_count),
        "sample_block_size": int(args.sample_block_size),
        "minibatch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "n_init": int(args.n_init),
        "sample_blocks": [asdict(block) for block in blocks],
        "outputs": {
            "kmeans": str(output_dir / "global_kmeans.pkl"),
            "mean": str(output_dir / "global_feature_mean.npy"),
            "std": str(output_dir / "global_feature_std.npy"),
            "centers_standardized": str(
                output_dir / "global_cluster_centers_standardized.npy"
            ),
            "centers_original": str(
                output_dir / "global_cluster_centers_original.npy"
            ),
        },
        "note": (
            "This model has no water semantics yet. Use training Otsu masks to "
            "identify which global cluster id corresponds to water, then save "
            "that id separately. Test data must use this saved mean/std and "
            "kmeans.predict(), not fit a new KMeans."
        ),
    }
    save_json(output_dir / "global_kmeans_training_summary.json", summary)

    print("Done.")
    print(f"Saved model: {output_dir / 'global_kmeans.pkl'}")
    print(f"Saved mean:  {output_dir / 'global_feature_mean.npy'}")
    print(f"Saved std:   {output_dir / 'global_feature_std.npy'}")
    print(f"Saved summary: {output_dir / 'global_kmeans_training_summary.json'}")


if __name__ == "__main__":
    main()
