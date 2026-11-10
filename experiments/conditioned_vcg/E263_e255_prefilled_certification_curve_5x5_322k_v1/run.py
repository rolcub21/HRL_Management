#!/usr/bin/env python3
"""Rerun E243's certification curve with corrected E255 seed 0."""

from unittest.mock import patch

from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1 import run as core

from . import protocol as p


def main() -> None:
    with patch.object(core, "p", p):
        core.main()


if __name__ == "__main__":
    main()
