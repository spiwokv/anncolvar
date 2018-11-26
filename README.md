[![Build Status](https://travis-ci.org/spiwokv/anncolvar.svg?branch=master)](https://travis-ci.org/spiwokv/anncolvar)

# anncolvar
Collective variables by artificial neural networks

```
usage: anncolvar [-h] [-i INFILE] [-p INTOP] [-c COLVAR] [-col COL]
                 [-boxx BOXX] [-boxy BOXY] [-boxz BOXZ] [-nofit NOFIT]
                 [-testset TESTSET] [-shuffle SHUFFLE] [-layers LAYERS]
                 [-layer1 LAYER1] [-layer2 LAYER2] [-layer3 LAYER3]
                 [-actfun1 ACTFUN1] [-actfun2 ACTFUN2] [-actfun3 ACTFUN3]
                 [-optim OPTIM] [-loss LOSS] [-epochs EPOCHS] [-batch BATCH]
                 [-o OFILE] [-model MODELFILE] [-plumed PLUMEDFILE]

Artificial neural network learning of collective variables of molecular
systems, requires numpy, keras and mdtraj

optional arguments:
  -h, --help          show this help message and exit
  -i INFILE           Input trajectory in pdb, xtc, trr, dcd, netcdf or mdcrd,
                      WARNING: the trajectory must be 1. centered in the PBC
                      box, 2. fitted to a reference structure and 3. must
                      contain only atoms to be analysed!
  -p INTOP            Input topology in pdb, WARNING: the structure must be 1.
                      centered in the PBC box and 2. must contain only atoms
                      to be analysed!
  -c COLVAR           Input collective variable file in text formate, must
                      contain the same number of lines as frames in the
                      trajectory
  -col COL            The index of the column containing collective variables
                      in the input collective variable file
  -boxx BOXX          Size of x coordinate of PBC box (from 0 to set value in
                      nm)
  -boxy BOXY          Size of y coordinate of PBC box (from 0 to set value in
                      nm)
  -boxz BOXZ          Size of z coordinate of PBC box (from 0 to set value in
                      nm)
  -nofit NOFIT        Disable fitting, the trajectory must be properly fited
                      (default False)
  -testset TESTSET    Size of test set (fraction of the trajectory, default =
                      0.1)
  -shuffle SHUFFLE    Shuffle trajectory frames to obtain training and test
                      set (default True)
  -layers LAYERS      Number of hidden layers (allowed values 1-3, default =
                      1)
  -layer1 LAYER1      Number of neurons in the first encoding layer (default =
                      256)
  -layer2 LAYER2      Number of neurons in the second encoding layer (default
                      = 256)
  -layer3 LAYER3      Number of neurons in the third encoding layer (default =
                      256)
  -actfun1 ACTFUN1    Activation function of the first layer (default =
                      sigmoid, for options see keras documentation)
  -actfun2 ACTFUN2    Activation function of the second layer (default =
                      linear, for options see keras documentation)
  -actfun3 ACTFUN3    Activation function of the third layer (default =
                      linear, for options see keras documentation)
  -optim OPTIM        Optimizer (default = adam, for options see keras
                      documentation)
  -loss LOSS          Loss function (default = mean_squared_error, for options
                      see keras documentation)
  -epochs EPOCHS      Number of epochs (default = 100, >1000 may be necessary
                      for real life applications)
  -batch BATCH        Batch size (0 = no batches, default = 256)
  -o OFILE            Output file with original and approximated collective
                      variables (txt, default = no output)
  -model MODELFILE    Prefix for output model files (experimental, default =
                      no output)
  -plumed PLUMEDFILE  Output file for Plumed (default = plumed.dat)

```
 
