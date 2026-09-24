#!/usr/bin/env python3
import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date_from_name(path: Path) -> str:
    """Extract YYYY-MM-DD from a filename."""
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(
            f"Cannot parse YYYY-MM-DD from filename: {path.name}. "
            "Expected a name such as area_2018-05-19.tif."
        )
    return match.group(1)


def find_sorted_tifs(sequence_dir: Path) -> list[Path]:
    """Find direct child GeoTIFFs and sort them by date in the filename."""
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


def feature_done(feature_dir: Path) -> bool:
    """Return True only when all expected extraction outputs exist."""
    return all(
        (feature_dir / filename).exists()
        for filename in ("features.npy", "coords.npy", "metadata.npz")
    )


def global_cluster_done(global_output_dir: Path, final_date: str) -> bool:
    """Check whether global clustering and the final-date prediction are complete."""
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
    """
    Build one extract_feature_mem.py command.

    Each prefix uses --feature-mode last, so the saved feature corresponds to
    the final SAR date in that prefix.
    """
    cmd = [
        python_bin,
        str(extract_script),
        "--images",
        *[str(path) for path in sar_paths],
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--feature-mode",
        "last",
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
    feature_dirs: list[Path],
    output_dir: Path,
    k: int,
    reference_tif: Path,
    global_extra_args: str,
) -> list[str]:
    """Build one global_cluster.py command."""
    cmd = [
        python_bin,
        str(global_script),
        "--feature-dirs",
        *[str(path) for path in feature_dirs],
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
    """Copy the final date's cluster products into one convenient directory."""
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
            "Extract prefix-date TCN features from one SAR sequence directory, "
            "run global clustering, and copy the final-date cluster map."
        )
    )

    parser.add_argument(
        "--sequence-dir",
        required=True,
        type=Path,
        help=(
            "One directory containing the SAR sequence directly, for example "
            "/path/to/hjalstaviken_2018/0519."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for this single sequence.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="TCN checkpoint passed to extract_feature_mem.py.",
    )
    parser.add_argument("--k", type=int, default=2)

    parser.add_argument(
        "--extract-script",
        type=Path,
        default=Path("./extract_feature_mem.py"),
    )
    parser.add_argument(
        "--global-script",
        type=Path,
        default=Path("./global_cluster_all_times.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument(
        "--extract-extra-args",
        default="--batch-size 131072 --device cuda --num-workers 0 --p-lower 1 --p-upper 99",
        help=(
            "Additional arguments passed verbatim to extract_feature_mem.py. "
            "Do not repeat --images, --checkpoint, --output-dir, or --feature-mode."
        ),
    )
    parser.add_argument(
        "--global-extra-args",
        default="--save-tif --save-preview",
        help="Additional arguments passed verbatim to global_cluster.py.",
    )

    parser.add_argument(
        "--use-recency-input",
        action="store_true",
        help="Pass --use-recency-input to extract_feature_mem.py.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=None,
        help="Pass --half-life-days when recency input is enabled.",
    )
    parser.add_argument(
        "--auto-recency-dates",
        action="store_true",
        help="Parse dates from filenames and pass --recency-dates for every prefix.",
    )

    parser.add_argument("--overwrite-features", default=False, action="store_true")
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
    final_date = dates[-1]
    final_tif = tifs[-1]

    print("=" * 100)
    print(f"[Sequence directory] {sequence_dir}")
    print(f"[Output directory]   {output_dir}")
    print(f"[Number of images]   {len(tifs)}")
    print(f"[Final date]         {final_date}")
    print("=" * 100)
    for date, path in zip(dates, tifs):
        print(f"  {date}: {path.name}")

    feature_output_root = output_dir / "per_date_features"
    global_output_dir = output_dir / f"global_cluster_k{args.k}"
    final_output_dir = output_dir / f"final_date_cluster_k{args.k}"
    feature_output_root.mkdir(parents=True, exist_ok=True)

    date_feature_dirs: list[Path] = []

    # Exclude the first date. Each feature is extracted from a growing prefix.
    for index in range(1, len(tifs)):
        current_date = dates[index]
        prefix_tifs = tifs[: index + 1]
        date_feature_dir = feature_output_root / current_date
        date_feature_dirs.append(date_feature_dir)

        if feature_done(date_feature_dir) and not args.overwrite_features:
            print(f"[Skip feature extraction] {current_date}: outputs already exist.")
            continue

        if args.overwrite_features and date_feature_dir.exists():
            shutil.rmtree(date_feature_dir)
        date_feature_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 100)
        print(f"[Extract feature] date={current_date}, prefix length={len(prefix_tifs)}")

        command = build_extract_cmd(
            python_bin=args.python_bin,
            extract_script=extract_script,
            sar_paths=prefix_tifs,
            output_dir=date_feature_dir,
            checkpoint=checkpoint,
            extract_extra_args=args.extract_extra_args,
            use_recency_input=args.use_recency_input,
            half_life_days=args.half_life_days,
            auto_recency_dates=args.auto_recency_dates,
        )
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        incomplete = [path for path in date_feature_dirs if not feature_done(path)]
        if incomplete:
            raise RuntimeError(
                "Some feature directories are incomplete:\n"
                + "\n".join(f"  {path}" for path in incomplete)
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
        print(f"[Global clustering] feature dates={len(date_feature_dirs)}, k={args.k}")

        command = build_global_cluster_cmd(
            python_bin=args.python_bin,
            global_script=global_script,
            feature_dirs=date_feature_dirs,
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
    print(f"Final-date outputs: {final_output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
