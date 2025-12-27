Overview
========

FoldTree2 is a computational toolkit that revolutionizes phylogenetic analysis by incorporating 3D protein structural information. Unlike traditional sequence-based phylogenetic methods, FoldTree2 uses neural networks to encode protein structures into discrete representations that capture evolutionary relationships at the structural level.

Core Concepts
-------------

Structure-Based Phylogenetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional phylogenetic methods rely on sequence alignments to infer evolutionary relationships. FoldTree2 extends this approach by:

1. **Structural encoding**: Converting 3D protein structures into discrete "structural alphabets"
2. **Structure-aware alignments**: Using tools like FoldSeek to align structures rather than sequences
3. **Custom substitution matrices**: Computing substitution probabilities based on structural similarities

Neural Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FoldTree2 employs encoder-decoder neural networks that:

* **Encode** protein structures as graph neural networks (PyTorch Geometric)
* **Learn** discrete structural representations through vector quantization
* **Decode** structural features back to validate the learned representations

Workflow Components
-------------------

1. **Data Preparation**
   
   * Download protein structures from AlphaFold Database
   * Convert PDB files to graph representations
   * Organize structures by evolutionary clusters

2. **Model Training**
   
   * Train encoder-decoder models on diverse protein structures
   * Learn discrete structural vocabularies
   * Validate models on held-out test sets

3. **Matrix Generation**
   
   * Encode aligned structures using trained models
   * Compute substitution frequencies from structural alignments
   * Generate matrices compatible with phylogenetic software

4. **Phylogenetic Analysis**
   
   * Use generated matrices with RAxML, MAFFT, or other tools
   * Perform maximum likelihood or distance-based inference
   * Analyze evolutionary patterns at the structural level

Applications
------------

* **Protein family analysis**: Understanding evolutionary relationships within protein families
* **Functional prediction**: Inferring function from structural phylogenies
* **Comparative genomics**: Large-scale structural comparisons across species
* **Drug discovery**: Identifying structurally similar proteins for drug targeting

Key Tools
---------

makesubmat.py
~~~~~~~~~~~~~

The primary tool for generating structure-based substitution matrices:

.. code-block:: bash

   python makesubmat.py --modelname trained_model \
                       --download_structs \
                       --align_structs \
                       --encode_alns

Training Scripts
~~~~~~~~~~~~~~~~

* ``learn_lightning.py``: PyTorch Lightning-based model training
* ``learn_monodecoder.py``: Standard PyTorch training pipeline

Analysis Tools
~~~~~~~~~~~~~~

* Structure visualization and analysis
* Matrix comparison and validation
* Phylogenetic tree evaluation