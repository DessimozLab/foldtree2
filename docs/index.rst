FoldTree2 Documentation
=======================

FoldTree2 is a toolkit for protein structure phylogenetic analysis using neural network encoders.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   api

Overview
========

FoldTree2 combines structural biology and phylogenetics by:

* **Structure-based phylogenetics**: Uses 3D protein structures instead of just sequences
* **Neural network encoding**: Converts protein structures to discrete representations
* **Custom substitution matrices**: Generates matrices based on structural similarities
* **Phylogenetic inference**: Supports both maximum likelihood and distance-based methods

Key Features
------------

* **makesubmat.py**: Generate structure-based substitution matrices from trained models
* **Neural network training**: Train encoder-decoder models on protein structures
* **Integration with phylogenetic tools**: Compatible with RAxML, MAFFT, and other tools
* **AlphaFold integration**: Works with AlphaFold Database structures

Quick Start
===========

To generate a substitution matrix from a trained model:

.. code-block:: bash

   python makesubmat.py --modelname my_model --download_structs --encode_alns

API Reference
=============

.. toctree::
   :maxdepth: 4

   foldtree2

Modules
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   foldtree2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`