Crop row Detector
============================================

This tool, *crop-row-detector* makes it possible to detect rows of crops in
georeferenced orthomosaic.

*crop-row-detector* uses the output from `CDC - Color Distance Calculator <https://github.com/henrikmidtiby/CDC>`_ or similar segmented orthomosaic and extract the crop from the background and uses Hough transform to detect crop rows.

Installation
------------

*crop-row-detector* is a python package and can be installed with pip:

.. code-block:: shell

   pip install .

See :doc:`Installation <installation>` for more advanced methods of installation.

Acknowledgement
---------------

the *crop-row-detector* tool was developed by SDU UAS Center as part of the project *Præcisionsfrøavl*, that was supported by the `Green Development and Demonstration Programme (GUDP) <https://gudp.lbst.dk/>`_ and `Frøafgiftsfonden <https://froeafgiftsfonden.dk/>`_ both from Denmark.

Index
-----

.. toctree::
   :maxdepth: 2

   installation
   reference
   CLI
   contributing
