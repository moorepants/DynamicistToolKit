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
a dynamicist. Right now it is basically a place I place general tools that
don't necessarily need a distribution of their own.

Modules
=======

**bicycle**
   Generic tools for basic bicycle dynamics analysis.
**inertia**
   Various functions for calculating and manipulating inertial quantities.
**process**
   Various tools for common signal processing tasks.

Installation
============

We recommend installing with conda so that dependency installation is not an
issue::

   $ conda install -c conda-forge dynamicisttoolkit

You will need Python 3.8+ and setuptools to install the packages. You can
install using pip. Pip will theoretically [#]_ get the dependencies for you (or
at least check if you have them)::

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

.. [#] You will need all build dependencies and also note that matplotlib
       doesn't play nice with pip.

Tests
=====

Run the tests with nose::

   $ nosetests

Documentation
=============

The documentation is hosted at ReadTheDocs:

http://dynamicisttoolkit.readthedocs.org

You can build the documentation (currently sparse) if you have Sphinx and
numpydoc::

   $ cd docs
   $ make html
   $ firefox _build/html/index.html
