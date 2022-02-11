|PyPI| |Anaconda| |BuildStatus| |WeeklyBuildStatus|  |codecov| |lgtm| |lgtmpy|
|DOI| |nest|

Read more in 
D. Trapl, I. Horvaćanin, V. Mareška, F. Özçelik, G. Unal and V. Spiwok: anncolvar: Approximation of Complex Collective Variables by Artificial Neural Networks for Analysis and Biasing of Molecular Simulations <https://www.frontiersin.org/articles/10.3389/fmolb.2019.00025/> *Front. Mol. Biosci.*  2019, **6**, 25 (doi: 10.3389/fmolb.2019.00025)

*********
anncolvar
*********

News
====

August 2020: Support for Python 2.7 terminated, use Python 3.

Current master vsersion makes it possible to use ANN module of recent master version of Plumed.

Syntax
======

Collective variables by artificial neural networks::

  usage: anncolvar [-h] [-i INFILE] [-p INTOP] [-c COLVAR] [-col COL]
                   [-boxx BOXX] [-boxy BOXY] [-boxz BOXZ] [-nofit NOFIT]
                   [-testset TESTSET] [-shuffle SHUFFLE] [-layers LAYERS]
                   [-layer1 LAYER1] [-layer2 LAYER2] [-layer3 LAYER3]
                   [-actfun1 ACTFUN1] [-actfun2 ACTFUN2] [-actfun3 ACTFUN3]
                   [-optim OPTIM] [-loss LOSS] [-epochs EPOCHS] [-batch BATCH]
                   [-o OFILE] [-model MODELFILE] [-plumed PLUMEDFILE]
                   [-plumed2 PLUMEDFILE2]
  
  Artificial neural network learning of collective variables of molecular
  systems, requires numpy, keras and mdtraj
  
  optional arguments:
    -h, --help           show this help message and exit
    -i INFILE            Input trajectory in pdb, xtc, trr, dcd, netcdf or mdcrd,
                         WARNING: the trajectory must be 1. must contain only atoms
                         to be analyzed, 2. must not contain any periodic boundary
                         condition issues!
    -p INTOP             Input topology in pdb, WARNING: the structure must be 1.
                         centered in the PBC box and 2. must contain only atoms
                         to be analyzed!
    -c COLVAR            Input collective variable file in text format, must
                         contain the same number of lines as frames in the
                         trajectory
    -col COL             The index of the column containing collective variables
                         in the input collective variable file
    -boxx BOXX           Size of x coordinate of PBC box (from 0 to set value in
                         nm)
    -boxy BOXY           Size of y coordinate of PBC box (from 0 to set value in
                         nm)
    -boxz BOXZ           Size of z coordinate of PBC box (from 0 to set value in
                         nm)
    -nofit NOFIT         Disable fitting, the trajectory must be properly fited
                         (default False)
    -testset TESTSET     Size of test set (fraction of the trajectory, default =
                         0.1)
    -shuffle SHUFFLE     Shuffle trajectory frames to obtain training and test
                         set (default True)
    -layers LAYERS       Number of hidden layers (allowed values 1-3, default =
                         1)
    -layer1 LAYER1       Number of neurons in the first encoding layer (default =
                         256)
    -layer2 LAYER2       Number of neurons in the second encoding layer (default
                         = 256)
    -layer3 LAYER3       Number of neurons in the third encoding layer (default =
                         256)
    -actfun1 ACTFUN1     Activation function of the first layer (default =
                         sigmoid, for options see keras documentation)
    -actfun2 ACTFUN2     Activation function of the second layer (default =
                         linear, for options see keras documentation)
    -actfun3 ACTFUN3     Activation function of the third layer (default =
                         linear, for options see keras documentation)
    -optim OPTIM         Optimizer (default = adam, for options see keras
                         documentation)
    -loss LOSS           Loss function (default = mean_squared_error, for options
                         see keras documentation)
    -epochs EPOCHS       Number of epochs (default = 100, >1000 may be necessary
                         for real life applications)
    -batch BATCH         Batch size (0 = no batches, default = 256)
    -o OFILE             Output file with original and approximated collective
                         variables (txt, default = no output)
    -model MODELFILE     Prefix for output model files (experimental, default =
                         no output)
    -plumed PLUMEDFILE   Output file for Plumed (default = plumed.dat)
    -plumed2 PLUMEDFILE2 Output file for Plumed with ANN module (default =
                         plumed2.dat)

Introduction
============

Biased simulations, such as metadynamics, use a predefined set of parameters known
as collective variables. An artificial bias force is applied on collective variables
to enhance sampling. There are two conditions for a parameter to be applied as
a collective variable. First, the value of the collective variables can be calculated
solely from atomic coordinates. Second, the force acting on collective variables
can be converted to the force acting on individual atoms. In the other words, it
is possible to calculate the first derivative of the collective variables with
respect to atomic coordinates. Both calculations must be fast enough, because
they must be evaluated in every step of the simulation.

There are many potential collective variables that cannot be easily calculated.
It is possible to calculate the collective variable for hundreds or thousands of
structures, but not for millions of structures (which is necessary for nanosecond
long simulations). *anncolvar* can approximate such collective variables using
a neural network.

Installation
============

You have to chose and install one of keras backends, such as Tensorflow, Theano or
CNTK. For this follow one of these links:

- TensorFlow_
- Theano_
- CNTK_ (CNTK 2.7 is the last release since 2019)

Install numpy and cython by PIP::

  pip install numpy cython

Next, install anncolvar by PIP::

  pip install anncolvar

If you use Anaconda type::

  conda install -c spiwokv anncolvar

Usage
=====

A series of representative structures (hundreds or more) with pre-calculated values
of the collective variable is used to train the neural network. The user can specify
the input set of reference structures (*-i*) in the form of a trajectory in pdb, xtc,
trr, dcd, netcdf or mdcrd. The trajectory must contain only atoms to be analyzed
(for example only non-hydrogen atoms). The trajectory must not contain any periodic
boundary condition issues. Both conversions can be made by molecular dynamics
simulation packages, for example by *gmx trjconv*. It is not necessary to fit
frames to a reference structure. It is possible to switch fitting off by
*-nofit True*.

It is necessary to supply an input topology in PDB. This is a structure used
as a template for fitting. It is also used to define a box. This box must be large
enough to fit the molecule in all frames of the trajectory. It should not be too
large because this suppresses non-linearity in the neural network. When the user
decides to use a 3x3x3 nm box it is necessary to place the molecule to be centered
at coordinates (1.5,1.5,1.5) nm. In Gromacs it is possible to use::

  gmx editconf -f mol.pdb -o reference.pdb -c -box 3 3 3

It must also contain only atoms to be analyzed. Size of the box can be specified
by parameters *-boxx*, *-boxy* and *-boxz* (in nm).

Last input file is the collective variable file. It is a space-separated text
file with the same number of lines as the number of frames in the input trajectory.
The index of the column can be specified by *-col* (e.g. *-col 2* for the second
column of the file.

The option *-testset* can control the fraction of the trajectory used as
the test set. For example *-testset 0.1* means that 10 % of input data is used
as the test set and 90 % as the training set. The option *-shuffle True* causes
that first 90 % is used as the training set and remaining 10 % as the test set.
Otherwise frames are shuffled before separation to the training and test set.

The architecture of the neural network is controlled by multiple parameters.
The input layer contains 3N neurons (where N is the number of atoms). The number
of hidden layers is controlled by *-layers*. This can be 1, 2 or 3. For higher
number of layers contact the authors. Number of neurons in the first, second and
third layer is controlled by *-layer1*, *-layer2* and *-layer3*. It is useful
to use the number of layers equal to powers of 2 (32, 64, 128 etc.). Huge numbers
of neurons can cause that the program is slow or run out of memory. Activation
functions of neurons can be controlled by *-actfun1*, *-actfun2* and *-actfun3*.
Any activation function supported by keras can be used.

The optimizer used in the training process can be controlled by *-optim*. The
default ADAM optimizer (*-optim adam*) works well. The loss function can be
controlled by *-loss*. The default *-loss mean_squared_error* works well. The
number of epochs can be controlled by *-epochs*. The default value (100) is
quite little, usually >1000 is necessary for real life applications. The batch
size can be controlled by *-batch* (*-batch 0* for no batches, default is 256).

Output is written into the text file *-o*. It contains the approximated and
the original values of collective variable. The model can be stored in the set
of text files (try *-model*). The input file is printed into the file controlled
by *-plumed* (by default plumed.dat). This file can be directly used to calculate
the evolution of the collective variable by *plumed driver* or by Plumed-patched
molecular dynamics engine. To use the collective variable in enhances sampling
(for example metadynamics) it is necessary to add a suitable keyword (for example
METAD).

.. |PyPI| image:: https://img.shields.io/pypi/v/anncolvar.svg
    :target: https://pypi.org/project/anncolvar/
    :alt: Latest version released on PyPI

.. |Anaconda| image:: https://anaconda.org/spiwokv/anncolvar/badges/version.svg
    :target: https://anaconda.org/spiwokv/anncolvar
    :alt: Latest version released on Anaconda Cloud

.. |BuildStatus| image:: https://github.com/spiwokv/anncolvar/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/spiwokv/anncolvar/actions
    :alt: Build status of the master branch on Mac/Linux at Github Actions

.. |WeeklyBuildStatus| image:: https://github.com/spiwokv/anncolvar/actions/workflows/weekly.yml/badge.svg
    :target: https://github.com/spiwokv/anncolvar/actions
    :alt: Weekly Monday 10 AM build status of the master branch on Mac/Linux at Github Actions

.. |codecov| image:: https://codecov.io/gh/spiwokv/anncolvar/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spiwokv/anncolvar/
    :alt: Code coverage

.. |lgtm| image:: https://img.shields.io/lgtm/alerts/g/spiwokv/anncolvar.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/spiwokv/anncolvar/alerts/
    :alt: LGTM code alerts

.. |lgtmpy| image:: https://img.shields.io/lgtm/grade/python/g/spiwokv/anncolvar.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/spiwokv/anncolvar/context:python
    :alt: LGTM python quality
    
.. |nest| image:: https://www.plumed-nest.org/eggs/19/008/badge.svg
    :target: https://www.plumed-nest.org/eggs/19/008/
    :alt: Plumed Nest ID: 008 

.. |DOI| image:: https://zenodo.org/badge/DOI/10.3389/fmolb.2019.00025.svg
    :target: https://doi.org/10.3389/fmolb.2019.00025 
    :alt: DOI: 10.3389/fmolb.2019.00025 

.. _TensorFlow: https://www.tensorflow.org/install/

.. _Theano: http://deeplearning.net/software/theano/install.html

.. _CNTK: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine

