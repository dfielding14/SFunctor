#!/usr/bin/env python3
"""Create a template configuration file for SFunctor."""

import sys
from pathlib import Path
from sfunctor.utils.config import create_config_template

def main():
    """Create config template with optional output path."""
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = Path("sfunctor.yaml")
    
    # Check if file exists
    if output_path.exists():
        response = input(f"{output_path} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create template
    template = create_config_template()
    
    # Write to file
    output_path.write_text(template)
    print(f"Created configuration template: {output_path}")
    print("\nTo use this configuration:")
    print(f"  python run_analysis.py --config {output_path} --file_name <slice.npz>")
    print("\nTo use a specific profile:")
    print(f"  python run_analysis.py --config {output_path} --profile production --file_name <slice.npz>")

if __name__ == "__main__":
    main()