#!/usr/bin/env python
from setuptools import setup, find_packages
import os

# Read the version from version.py without importing the package
version_file = os.path.join(os.path.dirname(__file__), 'src', 'enzywizard_substrate', 'version.py')
with open(version_file) as f:
    exec(f.read())  # defines __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enzywizard-substrate",
    version=__version__,                      # currently "1.0.1"
    author="bioinfbrad",
    description=(
        "Process small-molecule substrates from names or SMILES strings, generate "
        "a detailed JSON report, and produce 3D substrate structure files in SDF format."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioinfbrad/enzywizard-substrate",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "rdkit>=2026.03.1",          # core cheminformatics
        "numpy>=1.23.5",            # numerical operations
        "biopython>=1.86",          # sequence handling (used in some utilities)
        "requests>=2.33.0",         # API calls to ChEBI and PubChem
        "packaging",                # internal version handling
    ],
    entry_points={
        "console_scripts": [
            "enzywizard-substrate = enzywizard_substrate.cli:main",
        ],
    },
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
