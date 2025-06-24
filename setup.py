from setuptools import setup, find_packages

setup(
    name="foldtree2",
    version="0.1.0",
    description="Maximum likelihood phylogenetic tree inference from protein structures",
    long_description="FoldTree2 is a Python package for maximum likelihood phylogenetic tree inference from protein structures. It provides tools for analyzing and visualizing protein structures, as well as methods for inferring phylogenetic trees based on structural data.",
    long_description_content_type="text/markdown",
    author="Dave Moi",
    author_email="dmoi@unil.ch",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)

# The above code is a setup script for a Python package named "foldtree2".