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

- Removed gait analysis code (walk.py), now at
  http://github.com/csu-hmc/Gait-Analysis-Toolkit.
- TravisCI tests now run.

0.2.0
-----

- Addition of walking dynamics module.

0.1.0
-----

- Original code base that was used for the computations in this dissertation:
  https://github.com/moorepants/dissertation
