from setuptools import setup, find_packages

setup(
    name="foldtree2",
    version="0.1.0",
    description="Maximum likelihood phylogenetic tree inference from protein structures",
    long_description="FoldTree2 is a Python package for maximum likelihood phylogenetic tree inference from protein structures. It provides tools for analyzing and visualizing protein structures, as well as methods for inferring phylogenetic trees based on structural data.",
    long_description_content_type="text/markdown",
    author="Dave Moi",
    author_email="dmoi@unil.ch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "matplotlib",
        "torch",
        "scipy",
        "h5py",
        "pebble",
        "networkx",
        "einops",
        "pytorch-lightning",
        "torch-geometric",
        "biopython",
        "datasketch",
        "wget",
        "toytree",
        "toyplot",
        "statsmodels"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# The above code is a setup script for a Python package named "foldtree2".