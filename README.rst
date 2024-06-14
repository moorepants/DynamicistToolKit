=================
DynamicistToolKit
=================

.. list-table::

   * - PyPi
     - .. image:: https://img.shields.io/pypi/v/DynamicistToolKit.svg
          :target: https://pypi.org/project/DynamicistToolKit
       .. image:: https://pepy.tech/badge/DynamicistToolKit
          :target: https://pypi.org/project/DynamicistToolKit
   * - Anaconda
     - .. image:: https://anaconda.org/conda-forge/dynamicisttoolkit/badges/version.svg
          :target: https://anaconda.org/conda-forge/dynamicisttoolkit
       .. image:: https://anaconda.org/conda-forge/dynamicisttoolkit/badges/downloads.svg
          :target: https://anaconda.org/conda-forge/dynamicisttoolkit
   * - Documentation
     - .. image:: https://readthedocs.org/projects/dynamicisttoolkit/badge/?version=stable
          :target: http://dynamicisttoolkit.readthedocs.io
   * - Continous Integration
     - .. image:: https://github.com/moorepants/DynamicistToolKit/actions/workflows/test.yml/badge.svg

Introduction
============

This is a collection of Python modules which contain tools that are helpful for
a dynamics. We use it at the TU Delft Bicycle Lab as an initial shared location
to house reusable tools for the group. These tools may eventually graduate to
packages of their own or be incorporated into other existing specialized
packages.

Modules
=======

**bicycle**
   Generic tools for basic bicycle dynamics analysis.
**control**
  Functions helpful in control systems analysis.
**inertia**
   Various functions for calculating and manipulating inertial quantities.
**process**
   Various tools for common signal processing tasks.

Installation
============

You can install DynamicistToolKit with conda::

   $ conda install -c conda-forge dynamicisttoolkit

or pip::

   $ python -m pip install DynamicistToolKit

Or download the source with your preferred method and install manually.

Using Git::

   $ git clone git@github.com:moorepants/DynamicistToolKit.git
   $ cd DynamicistToolKit

Or wget::

   $ wget https://github.com/moorepants/DynamicistToolKit/archive/master.zip
   $ unzip master.zip
   $ cd DynamicistToolKit-master

Then for basic installation::

   $ python setup.py install

Or install for development purposes::

   $ python setup.py develop

Tests
=====

Run the unit tests with pytest::

   $ pytest dtk

and doctests with::

   $ pytest --doctest-modules dtk

Documentation
=============

The documentation is hosted at ReadTheDocs:

http://dynamicisttoolkit.readthedocs.org

You can build the documentation if you have Sphinx and numpydoc::

   $ cd docs
   $ make html
   $ firefox _build/html/index.html

To locally build on Windows, open an Anaconda prompt (the base environment has
sphinx and numpydoc installed), navigate to the DynamicistToolKit directory and
execute::

   $ cd docs
   $ make.bat html
   $ start "" ".\_build\html\index.html"
