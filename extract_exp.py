#!/usr/bin/env python3
"""Extract pretrained exp archive with optional root-folder remap.

Typical usage:
- Archive stores paths like: exp/...
- You can restore to:        exp/...      (default)
- Or remap to:               exp-tmp/...  (by changing EXTRACT_ROOT_NAME)
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


# ===== Easy-to-edit settings =====
# Root folder name inside archive members.
ARCHIVE_ROOT_NAME = "exp"
# Local root folder name after extraction.
# Change to "exp-tmp" if you do not want to overwrite existing exp/.
EXTRACT_ROOT_NAME = "exp"


def remap_root(rel_path: Path) -> Path:
    """Replace leading ARCHIVE_ROOT_NAME with EXTRACT_ROOT_NAME."""
    parts = list(rel_path.parts)
    if parts and parts[0] == ARCHIVE_ROOT_NAME:
        parts[0] = EXTRACT_ROOT_NAME
    return Path(*parts)


def extract_archive(repo_root: Path, archive_path: Path, force: bool) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    target_root = (repo_root / EXTRACT_ROOT_NAME).resolve()
    if target_root.exists() and not force:
        raise FileExistsError(
            f"Target root already exists: {target_root}\n"
            f"Refusing to overwrite. Re-run with --force if you really want to overwrite."
        )

    print(f"Repository root: {repo_root}")
    print(f"Archive path: {archive_path}")
    print(f"Root remap: {ARCHIVE_ROOT_NAME}/ -> {EXTRACT_ROOT_NAME}/")

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            mapped_path = remap_root(member_path)
            member.name = str(mapped_path)
            tar.extract(member, path=repo_root)

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pretrained exp archive.")
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("./data-exp-zip/exp-pretrained.tar.gz"),
        help="Path to exp archive (default: ./data-exp-zip/exp-pretrained.tar.gz).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow extraction when target root already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    archive_path = (repo_root / args.archive_path).resolve()
    extract_archive(repo_root=repo_root, archive_path=archive_path, force=args.force)


if __name__ == "__main__":
    main()
