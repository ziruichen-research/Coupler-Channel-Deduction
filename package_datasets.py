#!/usr/bin/env python3
"""Package DeepMIMO raw dataset into ./data-exp-zip/.

Output archives:
- ./data-exp-zip/DeepMIMO-O1.tar.gz

The archived paths keep their repository-relative layout, so extraction can
restore them back to the same `./data/...` locations.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


DATASETS = {
    # DeepMIMO raw source + trace path files used by trace_smooth.
    "DeepMIMO-O1": [
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/mat"),
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/x_trace_smooth.npy"),
        Path("data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/y_trace_smooth.npy"),
    ],
}


def package_one(repo_root: Path, out_dir: Path, dataset_name: str) -> None:
    src_list = DATASETS[dataset_name]
    for src_rel in src_list:
        src_abs = (repo_root / src_rel).resolve()
        if not src_abs.exists():
            raise FileNotFoundError(f"[{dataset_name}] source path not found: {repo_root / src_rel}")

    out_path = out_dir / f"{dataset_name}.tar.gz"
    print(f"[{dataset_name}] Packing {len(src_list)} path(s):")
    for src_rel in src_list:
        print(f"  - {repo_root / src_rel}")
    print(f"[{dataset_name}] Output archive: {out_path}")

    with tarfile.open(out_path, "w:gz") as tar:
        for src_rel in src_list:
            src_abs = (repo_root / src_rel).resolve()
            # Use repo-relative arcname so extraction recreates the same path layout.
            tar.add(src_abs, arcname=str(src_rel), recursive=True)

    print(f"[{dataset_name}] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package datasets into tar.gz archives.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data-exp-zip"),
        help="Output directory for archives (default: ./data-exp-zip).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to package. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {repo_root}")
    print(f"Archive output dir: {out_dir}")
    for name in args.datasets:
        package_one(repo_root, out_dir, name)

    print("All requested datasets have been packaged.")


if __name__ == "__main__":
    main()
