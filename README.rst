=================
DynamicistToolKit
=================

This is a collection of Python modules which contain tools that are helpful for
a dynamicist. Right now it is basically a place I place general tools that
don't necessarily need a distribution of their own.

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

Install the dependencies first (NumPy, SciPy, matplotlib, Pandas). The SciPy
Stack instructions are helpful for this: http://www.scipy.org/stackspec.html

Pip will theoretically [#]_ get the dependencies for you (or at least check)::

   $ pip install -r requirements.txt

And the development requirements::

   $ pip install -r dev-requirements.txt

I'm only testing with the versions in the requirements files for now, but the
software may work on older versions.

Now download the source with your preferred method.

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

Run the tests with nose::

   $ nosetests

Documentation
=============

You can build the documentation (currently sparse) if you have Sphinx and
numpydoc::

   $ cd docs
   $ make html
   $ firefox _build/html/index.html

.. [#] You will need all build dependencies and also note that matplotlib
       doesn't play nice with pip.

Release Notes
=============

0.2.0
-----

- Addition of walking dynamics module.

0.1.0
-----

- Original code base that was used for the computations in this dissertation:
  https://github.com/moorepants/dissertation
