#!/usr/bin/env python3
"""Run pinned OGBench reference training with an explicit dataset directory."""

from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    dataset_dir_value = os.environ.get("OGBENCH_DATASET_DIR")
    impls_root_value = os.environ.get("OGBENCH_IMPLS_ROOT")
    if not dataset_dir_value or not impls_root_value:
        raise RuntimeError("OGBENCH_DATASET_DIR and OGBENCH_IMPLS_ROOT are required")
    dataset_dir = Path(dataset_dir_value).resolve()
    impls_root = Path(impls_root_value).resolve()
    upstream_main = impls_root / "main.py"
    if not upstream_main.is_file():
        raise FileNotFoundError(upstream_main)

    import ogbench

    original = ogbench.make_env_and_datasets

    @functools.wraps(original)
    def with_dataset_dir(*args, **kwargs):
        kwargs.setdefault("dataset_dir", str(dataset_dir))
        return original(*args, **kwargs)

    ogbench.make_env_and_datasets = with_dataset_dir
    if str(impls_root) not in sys.path:
        sys.path.insert(0, str(impls_root))
    runpy.run_path(str(upstream_main), run_name="__main__")


if __name__ == "__main__":
    main()
