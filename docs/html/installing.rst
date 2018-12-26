Installation
============


Prerequisite
-------------

Anncolvar requires one of three machine learning beckends, either
Tensorflow, Theano or CNTK (all tested by continuous integration
services). Chose one of beckends and install by following these
sites:

`TensorFlow`_

`Theano`_

`CNTK`_


Installing with pip
-------------------

To install with pip, run the following::

 pip install anncolvar


Installing with pip from GitHub
-------------------------------

To install the master version from GitHub with pip, run the following::

 git clone https://github.com/spiwokv/anncolvar.git

 cd anncolvar

 pip install .


Upgrading
---------

To upgrade, type::

 pip install -U pip


Compatibility
-------------

Anncolvar requires Python libraries sys, datetime, argparse, numpy, mdtraj and keras.
Keras must run on one of three backends: Tensorflow, Theano or CNTK.

.. _TensorFlow https://www.tensorflow.org/install/

.. _Theano http://deeplearning.net/software/theano/install.html

.. _CNTK https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine

