"""Setup script for SFunctor package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()

setup(
    name="sfunctor",
    version="0.1.0",
    author="SFunctor Development Team",
    description="Structure Function Analysis for MHD Turbulence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sfunctor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "sfunctor=sfunctor.analysis.batch:main",
            "sfunctor-simple=sfunctor.analysis.simple:main",
            "sfunctor-viz=sfunctor.visualization.plots:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sfunctor": ["*.md", "*.txt"],
    },
)