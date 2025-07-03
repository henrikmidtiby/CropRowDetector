Installation
============

*crop-row-detector* is a python package and can be installed with pip or another package manager such as uv:

.. note::
   Python 3.11 or newer is required!

.. code-block:: shell

   pip install .

It have the following main dependencies:

* matplotlib
* numpy
* rasterio
* scikit-learn
* scipy
* pandas
* opencv
* pybaselines
* tqdm

If you want changes in the code to be reflected install it as an editable with:

.. code-block:: shell

   pip install -e .

All of these methods can also be used inside a virtual environment:

.. code-block:: shell

   python -m venv venv
   source venv/bin/activate
   pip install .
