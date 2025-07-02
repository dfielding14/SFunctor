"""Setup script for SFunctor package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def parse_requirements(filename):
    """Parse requirements file, ignoring comments, -r directives, and empty lines."""
    requirements = []
    for line in (this_directory / filename).read_text().splitlines():
        line = line.strip()
        # skip blanks, comments, recursive includes, editable installs, and VCS URLs
        if (
            not line
            or line.startswith("#")
            or line.startswith(("-r", "-e", "git+", "hg+", "svn+", "bzr+"))
        ):
            continue
        requirements.append(line)
    return requirements

requirements = parse_requirements("requirements.txt")
dev_requirements = parse_requirements("requirements-dev.txt")

setup(
    name="sfunctor",
    version="1.0.0",
    author="SFunctor Development Team",
    description="Structure Function Analysis for MHD Turbulence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sfunctor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "dev": dev_requirements
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