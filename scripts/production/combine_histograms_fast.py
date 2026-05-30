#!/usr/bin/env python3
"""Compatibility CLI for the bounded-memory streaming histogram combiner.

Historically this script loaded every decompressed histogram through worker
processes before reduction.  Streaming is faster for sparse compressed products
on the target filesystems and, more importantly, bounds peak memory.
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_histograms import combine_files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True, help="Glob pattern for histogram files")
    parser.add_argument("--output", default=None, help="Output NPZ filename")
    parser.add_argument("--mode", choices=("node", "slice"), default="node")
    parser.add_argument("--workers", type=int, default=None,
                        help="Accepted for compatibility; streaming reduction is intentionally bounded-memory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.workers is not None and args.verbose:
        print("--workers is ignored: streaming reduction keeps one decompressed input resident")
    input_files = sorted(glob(args.pattern))
    if not input_files:
        parser.error(f"No files found matching pattern {args.pattern!r}")
    return combine_files(input_files, mode=args.mode, output=args.output, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
