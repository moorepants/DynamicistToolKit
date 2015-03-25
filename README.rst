.. image:: https://pypip.in/version/DynamicistToolKit/badge.svg
   :target: https://pypi.python.org/pypi/DynamicistToolKit/
   :alt: Latest Version

.. image:: https://binstar.org/moorepants/dynamicisttoolkit/badges/version.svg
   :target: https://binstar.org/moorepants/dynamicisttoolkit

.. image:: https://travis-ci.org/moorepants/DynamicistToolKit.png?branch=master
   :target: http://travis-ci.org/moorepants/DynamicistToolKit

Introduction
============
My name is Robbie HAHAHAHA
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

You will need Python 2.7 and setuptools to install the packages. Its best to
install the dependencies first (NumPy, SciPy, matplotlib, Pandas).  The SciPy
Stack instructions are helpful for this: http://www.scipy.org/stackspec.html.

You can install using pip (or easy_install). Pip will theoretically [#]_ get
the dependencies for you (or at least check if you have them)::

   $ pip install DynamicistToolKit

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

Vagrant
=======

A vagrant file and provisioning script are included to test the code on an
Ubuntu 13.10 box. To load the box and run the tests simply type::

  $ vagrant up

See ``bootstrap.sh`` and ``VagrantFile`` to see what's going on.

Documentation
=============

The documentation is hosted at ReadTheDocs:

http://dynamicisttoolkit.readthedocs.org

You can build the documentation (currently sparse) if you have Sphinx and
numpydoc::

   $ cd docs
   $ make html
   $ firefox _build/html/index.html

Release Notes
=============

0.3.5
-----

- Fixed bug in coefficient_of_determination. [PR #23]

0.3.4
-----

- Fixed bug in normalized cutoff frequency calculation. [PR #21]

0.3.2
-----

- Fixed bug in butterworth function and added tests.

0.3.1
-----

- Fixed butterworth to work with SciPy 0.9.0. [PR #18]

0.3.0
-----

- Removed pandas dependency.
- Improved time vector function.
- Removed gait analysis code (walk.py), now at
  http://github.com/csu-hmc/Gait-Analysis-Toolkit.
- TravisCI tests now run, added image to readme.
- Added documentation at ReadTheDocs.

0.2.0
-----

- Addition of walking dynamics module.

0.1.0
-----

- Original code base that was used for the computations in this dissertation:
  https://github.com/moorepants/dissertation
