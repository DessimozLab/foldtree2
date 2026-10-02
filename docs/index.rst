FoldTree2 Documentation
=======================

FoldTree2 is a toolkit for protein structure phylogenetic analysis using neural network encoders.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   user_guide
   training
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

* **Structure-based phylogenetics**: Infer phylogenetic trees directly from protein 3D structures
* **Neural network encoding**: Encode structures using trained encoder-decoder models
* **Custom substitution matrices**: Generate MAFFT/RAxML-compatible matrices from structural alignments
* **Phylogenetic inference**: Integration with RAxML-NG, MAFFT, and MAD for tree inference and rooting

Quick Start
===========

Building a Phylogenetic Tree
----------------------------

To build a phylogenetic tree from protein structures using a trained model:

.. code-block:: bash

   ft2treebuilder \
     --encoder models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt \
     --decoder models/production/30char_minimal_decoder/final_30char_contacts_aa_decoder_full_epoch_52.pt \
     --mafftmat models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_mafftmat.mtx \
     --submat models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_submat.txt \
     --charmaps models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_pair_counts.pkl \
     --structures "/path/to/structures/*.pdb" \
     --device cpu --ncores 4 --outdir results/new_run

The final inferred tree is `results/new_run/*_.raxml.bestTree` in Newick format.

For a complete guide to tree inference, see the :doc:`user_guide`.

Preparing Training Data
-----------------------

To convert PDB files to graph representations for model training:

.. code-block:: bash

   pdbs-to-graphs /path/to/pdb_directory /path/to/training_graphs.h5 --verbose

This creates an HDF5 dataset suitable for training with the FoldTree2 neural network architecture.

For more details on data preparation and training, see the :doc:`training` guide.

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