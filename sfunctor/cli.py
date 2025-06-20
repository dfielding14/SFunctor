"""Command line interface entry points for sfunctor package."""

import sys
from pathlib import Path


def main():
    """Main entry point for sfunctor command."""
    # Add the parent directory to sys.path to allow imports of run_sf
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from run_sf import main as run_sf_main

    run_sf_main()


def extract_main():
    """Entry point for sfunctor-extract command."""
    # Add the parent directory to sys.path to allow imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from run_extract_slice import main as extract_slice_main

    extract_slice_main()


if __name__ == "__main__":
    main()
