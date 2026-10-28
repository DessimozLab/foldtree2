Source and API map
==================

The maintained `source module map
<https://github.com/DessimozLab/foldtree2/blob/main/foldtree2/src/README.md>`_
groups modules by responsibility and links their notebook origins.
Public import and class names remain stable for existing notebooks, scripts
and serialized checkpoints.

Core inference
--------------

* ``foldtree2.ft2treebuilder`` orchestrates the structure-to-tree pipeline.
* ``foldtree2.src.pdbgraphmk2`` converts structures and loads graph datasets.
* ``foldtree2.src.encoder`` and ``foldtree2.src.quantizers`` encode learned states.
* ``foldtree2.makesubmat`` builds model-specific matrices from references.
* ``foldtree2.src.ancestral`` shares native ancestral reconstruction.

Training and research
---------------------

See the `decoder/geometry/loss module catalogue
<https://github.com/DessimozLab/foldtree2/blob/main/foldtree2/src/README.md#model-and-geometry-modules>`_
for input/output contracts, optional dependencies and research limitations.
See :doc:`cli/README` for executable interfaces and :doc:`training` for protocols.

Documentation builds deliberately do not import training CLIs or optional
research modules. Historical generated RST pages remain in the repository for
reference but are excluded from the default site: some import modules that parse
arguments or require unavailable legacy components. Source docstrings remain the
implementation-level reference; importing every module is not a documentation
validation strategy.
