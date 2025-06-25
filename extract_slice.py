#!/usr/bin/env python3
"""Compatibility wrapper for extract_2d_slice functionality.

This script maintains backward compatibility with the old interface.
"""

from sfunctor.io.extract import main

if __name__ == "__main__":
    main()