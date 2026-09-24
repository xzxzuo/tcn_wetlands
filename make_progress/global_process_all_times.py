#!/usr/bin/env python3
import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date_from_name(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(
            f"Cannot parse YYYY-MM-DD from filename: {path.name}. "
            "Expected a name such as area_2018-05-19.tif."
        )
    return match.group(1)


def find_sorted_tifs(sequence_dir: Path) -> list[Path]:
    tifs = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    tifs.sort(key=parse_date_from_name)

    if len(tifs) < 2:
        raise ValueError(
            f"{sequence_dir} contains {len(tifs)} TIFF file(s); at least 2 are required."
        )

    dates = [parse_date_from_name(path) for path in tifs]
    if len(dates) != len(set(dates)):
        raise ValueError(
            "Duplicate dates were found in the input sequence:\n"
            + "\n".join(f"  {date}: {path}" for date, path in zip(dates, tifs))
        )

    return tifs


def all_time_features_done(feature_dir: Path, expected_time_steps: int | None = None) -> bool:
    required = [
        feature_dir / "features.npy",
        feature_dir / "coords.npy",
        feature_dir / "metadata.npz",
    ]
    if not all(path.exists() for path in required):
        return False

    try:
        features = np.load(feature_dir / "features.npy", mmap_mode="r")
    except Exception:
        return False

    if features.ndim != 3:
        return False

    if expected_time_steps is not None and features.shape[1] != expected_time_steps:
        return False

    return True


def global_cluster_done(global_output_dir: Path, final_date: str) -> bool:
    model_exists = (global_output_dir / "global_kmeans.pkl").exists()
    final_map_exists = any(
        (global_output_dir / final_date / filename).exists()
        for filename in ("global_cluster_map.tif", "global_cluster_map.npy")
    )
    return model_exists and final_map_exists


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n[CMD]", flush=True)
    print(shlex.join([str(item) for item in cmd]), flush=True)

    if not dry_run:
        subprocess.run(cmd, check=True)


def build_extract_cmd(
    python_bin: str,
    extract_script: Path,
    sar_paths: list[Path],
    output_dir: Path,
    checkpoint: Path,
    extract_extra_args: str,
    use_recency_input: bool,
    half_life_days: float | None,
    auto_recency_dates: bool,
) -> list[str]:
    """Build one command that extracts all temporal features in one pass."""
    cmd = [
        python_bin,
        str(extract_script),
        "--images",
        *[str(path) for path in sar_paths],
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]

    if use_recency_input:
        cmd.append("--use-recency-input")

        if half_life_days is not None:
            cmd.extend(["--half-life-days", str(half_life_days)])

        if auto_recency_dates:
            dates = [parse_date_from_name(path) for path in sar_paths]
            cmd.extend(["--recency-dates", *dates])

    if extract_extra_args:
        cmd.extend(shlex.split(extract_extra_args))

    return cmd


def build_global_cluster_cmd(
    python_bin: str,
    global_script: Path,
    feature_dir: Path,
    output_dir: Path,
    k: int,
    reference_tif: Path,
    global_extra_args: str,
) -> list[str]:
    """Build a command for global_cluster_all_times.py."""
    cmd = [
        python_bin,
        str(global_script),
        "--feature-dir",
        str(feature_dir),
        "--output-dir",
        str(output_dir),
        "--k",
        str(k),
        "--reference-tif",
        str(reference_tif),
    ]

    if global_extra_args:
        cmd.extend(shlex.split(global_extra_args))

    return cmd


def copy_final_cluster_outputs(
    global_output_dir: Path,
    final_date: str,
    final_output_dir: Path,
) -> None:
    source_dir = global_output_dir / final_date
    if not source_dir.exists():
        candidates = sorted(global_output_dir.rglob("global_cluster_map.*"))
        raise FileNotFoundError(
            f"Expected final-date output directory not found: {source_dir}\n"
            f"Cluster-map candidates found: {[str(path) for path in candidates]}"
        )

    final_output_dir.mkdir(parents=True, exist_ok=True)

    filenames = (
        "global_cluster_map.npy",
        "global_cluster_labels.npy",
        "global_cluster_map.tif",
        "global_cluster_map_preview.png",
        "global_cluster_counts.csv",
    )

    copied = 0
    for filename in filenames:
        source = source_dir / filename
        if source.exists():
            destination = final_output_dir / filename
            shutil.copy2(source, destination)
            copied += 1
            print(f"[Copied] {source} -> {destination}", flush=True)

    if copied == 0:
        raise FileNotFoundError(f"No final-date cluster outputs found in {source_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all temporal TCN features from one SAR sequence in one pass, "
            "run global clustering, and copy the final-date cluster map."
        )
    )

    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--k", type=int, default=2)

    parser.add_argument(
        "--extract-script",
        type=Path,
        default=Path("./extract_feature_all_times.py"),
        help="Extractor that saves features.npy with shape [N, T-1, C].",
    )
    parser.add_argument(
        "--global-script",
        type=Path,
        default=Path("./global_cluster_all_times.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument(
        "--extract-extra-args",
        default="--batch-size 524288 --device cuda --num-workers 0 --p-lower 1 --p-upper 99",
        help=(
            "Additional arguments passed verbatim to the all-time feature extractor. "
            "Do not repeat --images, --checkpoint, or --output-dir."
        ),
    )
    parser.add_argument(
        "--global-extra-args",
        default="--save-tif --save-preview",
        help="Additional arguments passed verbatim to global_cluster_all_times.py.",
    )

    parser.add_argument("--use-recency-input", action="store_true")
    parser.add_argument("--half-life-days", type=float, default=None)
    parser.add_argument("--auto-recency-dates", action="store_true")

    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-global", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    sequence_dir = args.sequence_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    extract_script = args.extract_script.expanduser().resolve()
    global_script = args.global_script.expanduser().resolve()

    if not sequence_dir.is_dir():
        raise NotADirectoryError(sequence_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not extract_script.is_file():
        raise FileNotFoundError(extract_script)
    if not global_script.is_file():
        raise FileNotFoundError(global_script)
    if args.k < 2:
        raise ValueError("--k must be at least 2.")
    if args.half_life_days is not None and args.half_life_days <= 0:
        raise ValueError("--half-life-days must be positive.")
    if (args.half_life_days is not None or args.auto_recency_dates) and not args.use_recency_input:
        raise ValueError(
            "--half-life-days and --auto-recency-dates require --use-recency-input."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tifs = find_sorted_tifs(sequence_dir)
    dates = [parse_date_from_name(path) for path in tifs]
    feature_dates = dates[1:]
    final_date = dates[-1]
    final_tif = tifs[-1]

    print("=" * 100)
    print(f"[Sequence directory] {sequence_dir}")
    print(f"[Output directory]   {output_dir}")
    print(f"[Number of images]   {len(tifs)}")
    print(f"[Feature dates]      {len(feature_dates)}")
    print(f"[Final date]         {final_date}")
    print("=" * 100)
    for date, path in zip(dates, tifs):
        print(f"  {date}: {path.name}")

    feature_dir = output_dir / "all_time_features"
    global_output_dir = output_dir / f"global_cluster_k{args.k}"
    final_output_dir = output_dir / f"final_date_cluster_k{args.k}"

    if args.overwrite_features and feature_dir.exists():
        shutil.rmtree(feature_dir)

    if all_time_features_done(feature_dir, expected_time_steps=len(feature_dates)) and not args.overwrite_features:
        print(f"[Skip feature extraction] Complete all-time features already exist: {feature_dir}")
    else:
        feature_dir.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(
            f"[Extract all temporal features] input images={len(tifs)}, "
            f"expected output shape=[N, {len(feature_dates)}, C]"
        )

        command = build_extract_cmd(
            python_bin=args.python_bin,
            extract_script=extract_script,
            sar_paths=tifs,
            output_dir=feature_dir,
            checkpoint=checkpoint,
            extract_extra_args=args.extract_extra_args,
            use_recency_input=args.use_recency_input,
            half_life_days=args.half_life_days,
            auto_recency_dates=args.auto_recency_dates,
        )
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run and not all_time_features_done(
        feature_dir,
        expected_time_steps=len(feature_dates),
    ):
        try:
            features = np.load(feature_dir / "features.npy", mmap_mode="r")
            actual = str(features.shape)
        except Exception:
            actual = "unavailable"
        raise RuntimeError(
            f"All-time feature extraction is incomplete. Expected features.npy shape "
            f"[N, {len(feature_dates)}, C], got {actual}."
        )

    if args.overwrite_global and global_output_dir.exists():
        shutil.rmtree(global_output_dir)

    if (
        not args.dry_run
        and global_cluster_done(global_output_dir, final_date)
        and not args.overwrite_global
    ):
        print(f"[Skip global clustering] Complete outputs already exist: {global_output_dir}")
    else:
        global_output_dir.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"[Global clustering] temporal slices={len(feature_dates)}, k={args.k}")

        command = build_global_cluster_cmd(
            python_bin=args.python_bin,
            global_script=global_script,
            feature_dir=feature_dir,
            output_dir=global_output_dir,
            k=args.k,
            reference_tif=final_tif,
            global_extra_args=args.global_extra_args,
        )
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        copy_final_cluster_outputs(
            global_output_dir=global_output_dir,
            final_date=final_date,
            final_output_dir=final_output_dir,
        )

    print("=" * 100)
    print("Done.")
    print(f"All-time features:  {feature_dir}")
    print(f"Global outputs:     {global_output_dir}")
    print(f"Final-date outputs: {final_output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
