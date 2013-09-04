=================
DynamicistToolKit
=================

This is a collection of Python modules and tools that are helpful for a
dynamicist. Right not is basically a place I throw general tools that I need
use of in the other software I write.

Modules
=======

process
-------

Various tools for common signal processing tasks.

inertia
-------

Various functions for calculating and manipulating inertial quantities.

bicycle
-------

Generic tools for basic bicycle dynamics analysis.

walk
----

Tools for working with gait data and walking models.

Installation
============

Get the dependencies: SciPy Stack http://www.scipy.org/stackspec.html

Now download the source.

Then for basic system installation::

   $ python setup.py install

And to install for development purposes::

   $ python setup.py develop

Tests
=====

Run the tests with nose::

   $ nosetests

Documentation
=============

You can build the documentation (currently sparse) if you have Sphinx::

   $ pip install sphinx numpydoc
   $ cd docs
   $ make html
   $ firefox _build/html/index.html
