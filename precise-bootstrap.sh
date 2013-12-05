#!/usr/bin/env bash

# This script installs all of the dependencies necessary to run the software on
# Ubuntu 12.04, Precise Pangolin.

apt-get update
# installation
apt-get install -y python-setuptools python-pip
# main dependencies
apt-get install -y python-numpy python-scipy python-matplotlib
# testing
apt-get install -y python-nose python-coverage
# documentation
apt-get install -y python-sphinx
pip install numpydoc
# other
apt-get install -y ipython

# Test and install current branch stored on local machine with the VM
cd /vagrant
nosetests -v --with-coverage --cover-package=dtk
python setup.py install
cd docs
make html

# Test and install HEAD of master branch pulled from Github
apt-get install -y git
cd $HOME
git clone https://github.com/moorepants/DynamicistToolKit.git
cd DynamicistToolKit
nosetests -v --with-coverage --cover-package=dtk
python setup.py install
cd docs
make html
