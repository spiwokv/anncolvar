import pytest
import mdtraj as md
import numpy as np
import keras as krs
import argparse as arg
import datetime as dt
import sys
import os

import anncolvar

def test_it():
  myinfilename = os.path.join(os.path.dirname(__file__), 'traj_fit.xtc')
  myintopname = os.path.join(os.path.dirname(__file__), 'reference.pdb')
  mycolvarname = os.path.join(os.path.dirname(__file__), 'results_isomap')
  myplumedname = os.path.join(os.path.dirname(__file__), 'test.dat')
  ae, cor = anncolvar.anncollectivevariable(infilename=myinfilename,
                                            intopname=myintopname,
                                            colvarname=mycolvarname,
                                            column=2, boxx=1.0, boxy=1.0, boxz=1.0,
                                            atestset=0.1, shuffle=1, nofit=0, layers=3, layer1=16, layer2=8, layer3=4,
                                            actfun1='sigmoid', actfun2='sigmoid', actfun3='sigmoid',
                                            optim='adam', loss='mean_squared_error', epochs=1000, batch=256,
                                            ofilename='', modelfile='', plumedfile=myplumedname)
  
  command = "plumed driver --mf_pdb "+myintopname+" --plumed "+myplumedname
  os.system(command)
  ifile = open("COLVAR", "r").readlines()
  print(ifile)
  assert(cor > 0.99)

if __name__ == '__main__':
  pytest.main([__file__])


