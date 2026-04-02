#!/usr/bin/env python3
"""Extract packaged datasets with a simple root-folder remap.

Usage goal:
- Archive stores paths like: data/DeepMIMO/...
- You can remap them to:    data-tmp/DeepMIMO/...
  by changing EXTRACT_ROOT_NAME below.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


# ===== Easy-to-edit settings =====
# Root folder name used inside archive paths.
ARCHIVE_ROOT_NAME = "data"
# Extraction root folder name to create locally.
# Change to "data" (default restore) or "data-tmp", etc.
EXTRACT_ROOT_NAME = "data-tmp"

DATASETS = {
    "DeepMIMO-O1": [
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/mat"),
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/x_trace_smooth.npy"),
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/y_trace_smooth.npy"),
    ],
}


def remap_root(rel_path: Path) -> Path:
    """Replace leading ARCHIVE_ROOT_NAME with EXTRACT_ROOT_NAME."""
    parts = list(rel_path.parts)
    if parts and parts[0] == ARCHIVE_ROOT_NAME:
        parts[0] = EXTRACT_ROOT_NAME
    return Path(*parts)


def extract_one(repo_root: Path, zip_dir: Path, dataset_name: str, force: bool) -> None:
    archive_path = zip_dir / f"{dataset_name}.tar.gz"
    archive_rel_list = DATASETS[dataset_name]
    target_rel_list = [remap_root(p) for p in archive_rel_list]
    target_abs_list = [(repo_root / rel).resolve() for rel in target_rel_list]

    if not archive_path.exists():
        raise FileNotFoundError(f"[{dataset_name}] archive not found: {archive_path}")

    if not force:
        existing_targets = [p for p in target_abs_list if p.exists()]
        if existing_targets:
            existing_str = "\n".join([f"  - {p}" for p in existing_targets])
            raise FileExistsError(
                f"[{dataset_name}] target path(s) already exist:\n{existing_str}\n"
                f"Refusing to overwrite. Re-run with --force if you really want to overwrite."
            )

    print(f"[{dataset_name}] Archive: {archive_path}")
    print(f"[{dataset_name}] Root remap: {ARCHIVE_ROOT_NAME}/ -> {EXTRACT_ROOT_NAME}/")

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            mapped_path = remap_root(member_path)
            member.name = str(mapped_path)  # tell tarfile where to extract this member
            tar.extract(member, path=repo_root)

    print(f"[{dataset_name}] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract dataset archives with optional root remap.")
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("./data-exp-zip"),
        help="Directory containing *.tar.gz archives (default: ./data-exp-zip).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to extract. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow extraction when target path already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    zip_dir = (repo_root / args.zip_dir).resolve()

    print(f"Repository root: {repo_root}")
    print(f"Archive dir: {zip_dir}")
    for name in args.datasets:
        extract_one(repo_root, zip_dir, name, force=args.force)

    print("All requested datasets have been extracted.")


if __name__ == "__main__":
    main()

