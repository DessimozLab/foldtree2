Installation
============

Requirements
------------

FoldTree2 requires Python 3.8+ and the following dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

* PyTorch >= 1.9.0
* PyTorch Geometric
* NumPy
* Pandas
* Matplotlib
* tqdm
* h5py

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* scikit-learn (for additional analysis tools)
* Biopython (for sequence handling)
* FoldSeek (for structural alignments)

Installation from Source
------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/DessimozLab/foldtree2.git
   cd foldtree2

2. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

3. Install the package:

.. code-block:: bash

   pip install -e .

Docker Installation
------------------

For reproducible environments, use the provided Docker configuration:

.. code-block:: bash

   docker build -t foldtree2 .
   docker run -it foldtree2

GPU Support
-----------

For GPU acceleration, install PyTorch with CUDA support:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric