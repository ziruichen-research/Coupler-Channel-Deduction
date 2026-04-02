#!/usr/bin/env python3
"""Package the whole exp directory for pretrained-model distribution.

The archive keeps repository-relative layout (default arcname: `exp`), so users
can extract it to project root and directly run test scripts.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package exp directory into a tar.gz archive.")
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("./exp"),
        help="Path to experiment directory (default: ./exp).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data-exp-zip"),
        help="Output directory for archive (default: ./data-exp-zip).",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default="exp-pretrained.tar.gz",
        help="Archive file name (default: exp-pretrained.tar.gz).",
    )
    parser.add_argument(
        "--arcname",
        type=str,
        default="exp",
        help="Top-level folder name inside archive (default: exp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    exp_root = (repo_root / args.exp_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_path = out_dir / args.archive_name

    if not exp_root.exists():
        raise FileNotFoundError(f"exp root not found: {exp_root}")
    if not exp_root.is_dir():
        raise NotADirectoryError(f"exp root is not a directory: {exp_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {repo_root}")
    print(f"Exp source: {exp_root}")
    print(f"Output archive: {out_path}")
    print(f"Archive top-level folder: {args.arcname}")

    # dereference=True follows symlinks (important when ./exp is a symlink).
    with tarfile.open(out_path, mode="w:gz", dereference=True) as tar:
        tar.add(exp_root, arcname=args.arcname, recursive=True)

    print("Done.")


if __name__ == "__main__":
    main()
