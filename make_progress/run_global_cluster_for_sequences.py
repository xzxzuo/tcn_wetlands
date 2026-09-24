#!/usr/bin/env python3
import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date_from_name(path: Path):
    """
    Extract YYYY-MM-DD from filename.
    Example:
        hjalstaviken_2018-05-19.tif -> 2018-05-19
    """
    m = DATE_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    return m.group(1)


def mmdd_to_date(mmdd: str, year: int):
    """
    Convert folder name like 0519 to 2018-05-19.
    """
    if len(mmdd) != 4 or not mmdd.isdigit():
        raise ValueError(f"Sequence folder name should be MMDD, got: {mmdd}")

    month = int(mmdd[:2])
    day = int(mmdd[2:])
    return f"{year:04d}-{month:02d}-{day:02d}"


def find_sequence_dirs(sequence_root: Path):
    """
    Find sequence folders such as 0519, 0605, 0711.
    """
    dirs = [
        p for p in sequence_root.iterdir()
        if p.is_dir() and len(p.name) == 4 and p.name.isdigit()
    ]
    return sorted(dirs, key=lambda p: p.name)


def find_sorted_tifs(sequence_dir: Path):
    """
    Find and sort tif files by date parsed from filename.
    """
    tifs = sorted(
        list(sequence_dir.glob("*.tif")) + list(sequence_dir.glob("*.tiff")),
        key=lambda p: parse_date_from_name(p),
    )

    if len(tifs) < 2:
        raise ValueError(f"{sequence_dir} has fewer than 2 tif files.")

    return tifs


def feature_done(feature_dir: Path):
    """
    Check whether feature extraction result already exists.
    """
    return (
        (feature_dir / "features.npy").exists()
        and (feature_dir / "coords.npy").exists()
        and (feature_dir / "metadata.npz").exists()
    )


def run_command(cmd, dry_run=False):
    print("\n[CMD]")
    print(" ".join(str(x) for x in cmd), flush=True)

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def build_extract_cmd(
    python_bin,
    extract_script,
    sar_paths,
    output_dir,
    checkpoint,
    extract_extra_args,
    use_recency_input=False,
    half_life_days=None,
    auto_recency_dates=False,
):
    """
    Call extract_feature_mem.py.

    Your extractor args are:

        --images img1 img2 ... imgT
        --checkpoint checkpoint.pt
        --output-dir output_dir
        --batch-size
        --num-workers
        --p-lower
        --p-upper
        --feature-mode
        --time-index
        --device
        --use-recency-input
        --half-life-days
        --recency-dates

    For this global-clustering experiment, we use prefix sequences and
    --feature-mode last, so each output corresponds to the current date.
    """

    cmd = [
        python_bin,
        str(extract_script),
        "--images",
        *[str(p) for p in sar_paths],
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
            dates = [parse_date_from_name(Path(p)) for p in sar_paths]
            cmd.extend(["--recency-dates", *dates])

    if extract_extra_args:
        cmd.extend(shlex.split(extract_extra_args))

    return cmd


def build_global_cluster_cmd(
    python_bin,
    global_script,
    feature_dirs,
    output_dir,
    k,
    reference_tif,
    global_feature_dirs_arg,
    global_output_arg,
    global_k_arg,
    global_reference_arg,
    global_extra_args,
):
    """
    Default assumed global cluster command:

    python ./global_cluster.py \
      --feature-dirs date_feature_dir_1 date_feature_dir_2 ... \
      --output-dir global_cluster_output \
      --k 4 \
      --reference-tif final_sar.tif \
      --save-tif \
      --save-preview

    This matches the global_cluster.py structure we discussed earlier.
    """
    cmd = [
        python_bin,
        str(global_script),
        global_feature_dirs_arg,
        *[str(d) for d in feature_dirs],
        global_output_arg,
        str(output_dir),
        global_k_arg,
        str(k),
    ]

    if reference_tif is not None:
        cmd.extend([global_reference_arg, str(reference_tif)])

    if global_extra_args:
        cmd.extend(shlex.split(global_extra_args))

    return cmd


def copy_final_cluster_outputs(global_output_dir: Path, final_date: str, final_output_dir: Path):
    """
    Copy the final-date cluster outputs to a convenient folder.

    Expected global_cluster.py output structure:
        global_output_dir/
            2018-05-19/
                global_cluster_map.npy
                global_cluster_map.tif
                global_cluster_map_preview.png
    """
    final_output_dir.mkdir(parents=True, exist_ok=True)

    candidate_dir = global_output_dir / final_date

    if not candidate_dir.exists():
        print(
            f"[Warning] Expected final date output folder not found: {candidate_dir}",
            flush=True,
        )
        print("[Warning] Will try recursive search.", flush=True)

        matches = list(global_output_dir.rglob("global_cluster_map.npy"))
        if not matches:
            print("[Warning] No global_cluster_map.npy found under global output.", flush=True)
            return

        print("[Warning] Found cluster maps:")
        for m in matches:
            print("  ", m)

        return

    names = [
        "global_cluster_map.npy",
        "global_cluster_labels.npy",
        "global_cluster_map.tif",
        "global_cluster_map_preview.png",
        "global_cluster_counts.csv",
    ]

    for name in names:
        src = candidate_dir / name
        if src.exists():
            dst = final_output_dir / name
            shutil.copy2(src, dst)
            print(f"[Copied] {src} -> {dst}", flush=True)


def process_one_sequence(args, sequence_dir: Path):
    sequence_name = sequence_dir.name
    expected_final_date = mmdd_to_date(sequence_name, args.year)

    print("=" * 100)
    print(f"[Sequence] {sequence_name}")
    print(f"[Expected final date] {expected_final_date}")
    print("=" * 100)

    tifs = find_sorted_tifs(sequence_dir)
    tif_dates = [parse_date_from_name(p) for p in tifs]

    print("[TIF sequence]")
    for d, p in zip(tif_dates, tifs):
        print(f"  {d}: {p}")

    actual_final_date = tif_dates[-1]
    final_tif = tifs[-1]

    if actual_final_date != expected_final_date:
        msg = (
            f"Folder name {sequence_name} means {expected_final_date}, "
            f"but last tif date is {actual_final_date}: {final_tif}"
        )
        if args.strict_final_date:
            raise ValueError(msg)
        else:
            print(f"[Warning] {msg}", flush=True)

    sequence_output_dir = args.output_root / sequence_name
    feature_output_root = sequence_output_dir / "per_date_features"
    global_output_dir = sequence_output_dir / f"global_cluster_k{args.k}"
    final_output_dir = sequence_output_dir / f"final_date_cluster_k{args.k}"

    feature_output_root.mkdir(parents=True, exist_ok=True)

    date_feature_dirs = []

    # Exclude the first date.
    # For date i, use prefix sequence tifs[:i+1].
    for i in range(1, len(tifs)):
        current_date = tif_dates[i]
        prefix_tifs = tifs[: i + 1]

        date_feature_dir = feature_output_root / current_date
        date_feature_dirs.append(date_feature_dir)

        if feature_done(date_feature_dir) and not args.overwrite_features:
            print(f"[Skip feature extraction] {current_date}: already exists.")
            continue

        date_feature_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f"[Extract feature] sequence={sequence_name}, date={current_date}")
        print(f"[Prefix length] {len(prefix_tifs)}")
        print("-" * 80)

        cmd = build_extract_cmd(
            python_bin=args.python_bin,
            extract_script=args.extract_script,
            sar_paths=prefix_tifs,
            output_dir=date_feature_dir,
            checkpoint=args.checkpoint,
            extract_extra_args=args.extract_extra_args,
            use_recency_input=args.use_recency_input,
            half_life_days=args.half_life_days,
            auto_recency_dates=args.auto_recency_dates,
        )

        run_command(cmd, dry_run=args.dry_run)

    # Check all required feature dirs exist.
    if not args.dry_run:
        missing = [d for d in date_feature_dirs if not feature_done(d)]
        if missing:
            raise RuntimeError(
                "Some date feature directories are incomplete:\n"
                + "\n".join(str(d) for d in missing)
            )

    if global_output_dir.exists() and args.overwrite_global:
        shutil.rmtree(global_output_dir)

    if global_output_dir.exists() and not args.overwrite_global:
        print(f"[Skip global clustering] output already exists: {global_output_dir}")
    else:
        global_output_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f"[Global clustering] sequence={sequence_name}")
        print(f"[Feature dirs] {len(date_feature_dirs)} dates")
        print(f"[Final reference tif] {final_tif}")
        print("-" * 80)

        cmd = build_global_cluster_cmd(
            python_bin=args.python_bin,
            global_script=args.global_script,
            feature_dirs=date_feature_dirs,
            output_dir=global_output_dir,
            k=args.k,
            reference_tif=final_tif,
            global_feature_dirs_arg=args.global_feature_dirs_arg,
            global_output_arg=args.global_output_arg,
            global_k_arg=args.global_k_arg,
            global_reference_arg=args.global_reference_arg,
            global_extra_args=args.global_extra_args,
        )

        run_command(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        copy_final_cluster_outputs(
            global_output_dir=global_output_dir,
            final_date=actual_final_date,
            final_output_dir=final_output_dir,
        )

    print(f"[Done sequence] {sequence_name}")
    print(f"[Final outputs] {final_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each SAR input sequence folder, extract per-date TCN features "
            "for all dates except the first one, then run global clustering and "
            "copy the final-date cluster map."
        )
    )

    parser.add_argument(
        "--sequence-root",
        required=True,
        type=Path,
        help="Root folder such as /path/to/hjalstaviken_2018 containing 0519, 0605, ...",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Year used to interpret folders like 0519 as YYYY-05-19.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root directory.",
    )

    parser.add_argument("--extract-script", default="./extract_feature_mem.py", type=Path)
    parser.add_argument("--global-script", default="./global_cluster.py", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="TCN checkpoint passed to extract_feature_mem.py.",
    )

    parser.add_argument("--k", type=int, required=True)

    # Argument names for extract_feature_mem.py
    parser.add_argument("--extract-path-arg", default="--sar-paths")
    parser.add_argument("--extract-output-arg", default="--output-dir")
    parser.add_argument("--extract-checkpoint-arg", default="--checkpoint")
    parser.add_argument(
        "--extract-extra-args",
        default="--batch-size 4096 --use-recency-input --half-life-days 6 --p-lower 1 --p-upper 99 --device cuda",
        help=(
            "Extra args passed to extract_feature_mem.py, e.g. "
            "'--batch-size 4096 --num-workers 2 --device cuda --normalization zscore'"
        ),
    )

    # Argument names for global_cluster.py
    parser.add_argument("--global-feature-dirs-arg", default="--feature-dirs")
    parser.add_argument("--global-output-arg", default="--output-dir")
    parser.add_argument("--global-k-arg", default="--k")
    parser.add_argument("--global-reference-arg", default="--reference-tif")
    parser.add_argument(
        "--global-extra-args",
        default="--save-tif --save-preview",
        help="Extra args passed to global_cluster.py.",
    )

    parser.add_argument(
        "--only-sequences",
        nargs="*",
        default=None,
        help="Optional list of sequence folder names to process, e.g. 0519 0605.",
    )

    parser.add_argument("--overwrite-features", default=False, action="store_true")
    parser.add_argument("--overwrite-global", action="store_true")
    parser.add_argument("--strict-final-date", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
   

    parser.add_argument(
        "--use-recency-input",
        action="store_true",
        help="Pass --use-recency-input to extract_feature_mem.py.",
    )

    parser.add_argument(
        "--half-life-days",
        type=float,
        default=None,
        help="Pass --half-life-days to extract_feature_mem.py.",
    )

    parser.add_argument(
        "--auto-recency-dates",
        action="store_true",
        help="Automatically pass --recency-dates parsed from SAR filenames for each prefix sequence.",
    )

    args = parser.parse_args()

    args.sequence_root = args.sequence_root.resolve()
    args.output_root = args.output_root.resolve()
    args.extract_script = args.extract_script.resolve()
    args.global_script = args.global_script.resolve()

    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()

    if not args.sequence_root.exists():
        raise FileNotFoundError(args.sequence_root)

    if not args.extract_script.exists():
        raise FileNotFoundError(args.extract_script)

    if not args.global_script.exists():
        raise FileNotFoundError(args.global_script)

    args.output_root.mkdir(parents=True, exist_ok=True)

    sequence_dirs = find_sequence_dirs(args.sequence_root)

    if args.only_sequences is not None and len(args.only_sequences) > 0:
        keep = set(args.only_sequences)
        sequence_dirs = [d for d in sequence_dirs if d.name in keep]

    if len(sequence_dirs) == 0:
        raise ValueError("No sequence folders found.")

    print("[Sequence root]", args.sequence_root)
    print("[Output root]", args.output_root)
    print("[Sequences]")
    for d in sequence_dirs:
        print("  ", d.name)

    for sequence_dir in sequence_dirs:
        process_one_sequence(args, sequence_dir)

    print("=" * 100)
    print("All done.")
    print("=" * 100)


if __name__ == "__main__":
    main()

# python3 run_global_cluster_for_sequences.py \
#   --sequence-root ./hjalstaviken_2018 \
#   --year 2018 \
#   --output-root ./hjalstaviken_2018_global_cluster_outputs \
#   --checkpoint ./tcn_checkpoint.pt \
#   --k 2 \
#   --auto-recency-dates
#   --extract-extra-args "--batch-size 4096 --num-workers 2" \
#   --dry-run