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
  myplumedname2 = os.path.join(os.path.dirname(__file__), 'test2.dat')
  ae, cor = anncolvar.anncollectivevariable(infilename=myinfilename,
                                            intopname=myintopname,
                                            colvarname=mycolvarname,
                                            column=2, boxx=1.0, boxy=1.0, boxz=1.0,
                                            atestset=0.1, shuffle=1, nofit=0, layers=2, layer1=16, layer2=8, layer3=4,
                                            actfun1='sigmoid', actfun2='sigmoid', actfun3='linear',
                                            optim='adam', loss='mean_squared_error', epochs=1000, batch=256,
                                            ofilename='', modelfile='', plumedfile=myplumedname, plumedfile2=myplumedname2)
  
  command = "plumed driver --mf_pdb "+myintopname+" --plumed "+myplumedname2
  os.system(command)
  ifile = open("COLVAR", "r").readlines()
  sline = str.split(ifile[1])
  x = float(sline[1])
  assert((x > 0.29) and (x < 0.33))

if __name__ == '__main__':
  pytest.main([__file__])

