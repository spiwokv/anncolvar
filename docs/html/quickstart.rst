Quickstart
==========

For help type::

 anncolvar -h


In a local directory place an mdtraj-compatible trajectory (without periodic boundary issues,
analysed atoms only) into the file traj_fit.xtc, its structure in PDB format in reference.pdb
(same atoms as in traj_fit.xtc and values of collective variables in the file results_isomap
(space separated file with structure number in the first column and collective variables in
second, third, fourth and fifth column). Next type::

 anncolvar -p reference.pdb -c results_isomap -col 2 -boxx 1 -boxy 1 -boxz 1 \
           -layers 3 -layer1 16 -layer2 8 -layer3 4 -actfun1 sigmoid -actfun2 sigmoid -actfun3 sigmoid \
           -optim adam -loss mean_squared_error -epochs 1000 -batch 256 \
           -o low.txt -model model -plumed plumed.dat

