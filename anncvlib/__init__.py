# Loading necessary libraries
libnames = [('mdtraj', 'md'), ('numpy', 'np'), ('keras', 'krs'), ('argparse', 'arg'), ('datetime', 'dt'), ('sys', 'sys')]

for (name, short) in libnames:
  try:
    lib = __import__(name)
  except:
    print("Library %s is not installed, exiting" % name)
    exit(0)
  else:
    globals()[short] = lib

def anncollectivevariable(infilename='', intopname='', colvarname='', column=2,
                          boxx=0.0, boxy=0.0, boxz=0.0, atestset=0.1,
                          shuffle=1, layers=2, layer1=256, layer2=256, layer3=256,
                          actfun1='sigmoid', actfun2='sigmoid', actfun3='sigmoid',
                          optim='adam', loss='mean_squared_error', epochs=100, batch=0,
                          ofilename='', modelfile='', plumedfile=''):
  try:
    traj = md.load(infilename, top=intopname)
  except:
    print("Cannot load %s or %s, exiting." % (infilename, intopname))
    exit(0)
  else:
    print("%s succesfully loaded" % traj)
  print("")
  
  # Conversion of the trajectory from Nframes x Natoms x 3 to Nframes x (Natoms x 3)
  trajsize = traj.xyz.shape
  traj2 = np.zeros((trajsize[0], trajsize[1]*3))
  for i in range(trajsize[1]):
    traj2[:,3*i]   = traj.xyz[:,i,0]
    traj2[:,3*i+1] = traj.xyz[:,i,1]
    traj2[:,3*i+2] = traj.xyz[:,i,2]
  
  # Checking whether all atoms fit the box
  if (np.amin(traj2)) < 0.0:
    print("ERROR: Some of atom has negative coordinate (i.e. it is outside the box)")
    exit(0)

  if boxx == 0.0 or boxy == 0.0 or boxz == 0.0:
    print("WARNING: box size not set, it will be determined automatically")
    if boxx == 0.0:
      boxx = 1.2*np.amax(traj.xyz[:,:,0])
    if boxy == 0.0:
      boxy = 1.2*np.amax(traj.xyz[:,:,1])
    if boxz == 0.0:
      boxz = 1.2*np.amax(traj.xyz[:,:,2])
    print("box size set to %6.3f x %6.3f x %6.3f nm" % (boxx, boxy, boxz))
    print("")
  
  if np.amax(traj.xyz[:,:,0]) > boxx or np.amax(traj.xyz[:,:,1]) > boxy or np.amax(traj.xyz[:,:,2]) > boxz:
    print("ERROR: Some of atom has coordinate higher than box size (i.e. it is outside the box)")
    exit(0)
  
  if boxx > 2.0*np.amax(traj.xyz[:,:,0]) or boxy > 2.0*np.amax(traj.xyz[:,:,0]) or boxz > 2.0*np.amax(traj.xyz[:,:,0]):
    print("WARNING: Box size is bigger than 2x of highest coordinate,")
    print("maybe the box is too big or the molecule is not centered")
  
  maxbox = max([boxx, boxy, boxz])
  
  # Checking colvar file
  try:
    cvfile = open(colvarname, 'r').readlines()
  except:
    print("Cannot load %s, exiting." % colvarname)
    exit(0)
  cvs = []
  for line in cvfile:
    sline = str.split(line)
    if len(sline) > 1:
      if sline[0][0]!="#":
        if len(sline) >= column:
          try:
            cvs.append(float(sline[column-1]))
          except:
            print("Cannot read %s." % colvarname)
            exit(0)
  if len(cvs) != trajsize[0]:
    print("File %s contains %i values, but %s contains %i frames, exiting." % (colvarname, len(cvs), infilename, trajsize[0]))
    exit(0)
  cvs = np.array(cvs)
  
  # Splitting the trajectory into training and testing sets
  testsize = int(atestset * trajsize[0])
  if testsize < 1:
    print("ERROR: testset empty, increase testsize")
    exit(0)
  print("Training and test sets consist of %i and %i trajectory frames, respectively" % (trajsize[0]-testsize, testsize))
  print("")
  
  # Shuffling the trajectory before splitting
  if shuffle == 1:
    print("Trajectory will be shuffled before splitting into training and test set")
  elif shuffle == 0:
    print("Trajectory will NOT be shuffled before splitting into training and test set")
    print("(first %i frames will be used for trainintg, next %i for testing)" % (trajsize[0]-testsize, testsize))
  indexes = list(range(trajsize[0]))
  if shuffle == 1:
    np.random.shuffle(indexes)
  training_set, testing_set = traj2[indexes[:-testsize],:]/maxbox, traj2[indexes[-testsize:],:]/maxbox
  training_cvs, testing_cvs = cvs[indexes[:-testsize]], cvs[indexes[-testsize:]]
  
  # (Deep) learning  
  input_coord = krs.layers.Input(shape=(trajsize[1]*3,))
  encoded = krs.layers.Dense(layer1, activation=actfun1, use_bias=True)(input_coord)
  if layers == 4:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=True)(encoded)
    encoded = krs.layers.Dense(layer3, activation=actfun3, use_bias=True)(encoded)
  if layers == 3:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=True)(encoded)
  encoded = krs.layers.Dense(1, activation='linear', use_bias=True)(encoded)
  codecvs = krs.models.Model(input_coord, encoded)
  codecvs.compile(optimizer=optim, loss=loss)
  
  if batch>0:
    codecvs.fit(training_set, training_cvs,
                epochs=epochs,
                batch_size=batch,
                validation_data=(testing_set, testing_cvs))
  else:
    codecvs.fit(training_set, training_cvs,
                epochs=epochs,
                validation_data=(testing_set, testing_cvs))
  
  # Encoding and decoding the trajectory
  coded_cvs = codecvs.predict(traj2/maxbox)
  
  # Calculating Pearson correlation coefficient
  print("")
  print("Pearson correlation coefficient for original and coded cvs is %f" % np.corrcoef(cvs,coded_cvs[:,0])[0,1])
  print("")
  
  print("Pearson correlation coefficient for original and coded cvs in training set is %f" % np.corrcoef(training_cvs,coded_cvs[indexes[:-testsize],0])[0,1])
  print("")
  
  print("Pearson correlation coefficient for original and coded cvs in testing set is %f" % np.corrcoef(testing_cvs,coded_cvs[indexes[-testsize:],0])[0,1])
  print("")
  
  # Generating low-dimensional output
  if len(ofilename) > 0:
    print("Writing collective variables into %s" % ofilename)
    print("")
    ofile = open(ofilename, "w")
    for i in range(trajsize[0]):
      ofile.write("%f %f " % (coded_cvs[i],cvs[i]))
      typeofset = 'TE'
      if i in indexes[:-testsize]:
        typeofset = 'TR'
      ofile.write("%s \n" % typeofset)
    ofile.close()
  
  # Saving the model
  if modelfile != '':
    print("Writing model into %s.txt" % modelfile)
    print("")
    ofile = open(modelfile+'.txt', "w")
    ofile.write("maxbox = %f\n" % maxbox)
    ofile.write("input_coord = krs.layers.Input(shape=(trajsize[1]*3,))\n")
    ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(input_coord)\n" % (layer1, actfun1))
    if layers == 4:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer2, actfun2))
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer3, actfun3))
    if layers == 3:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer2, actfun2))
    ofile.write("encoded = krs.layers.Dense(%i, activation='linear', use_bias=True)(encoded)\n" % encdim)
    ofile.write("codecvs = krs.models.Model(input_coord, encoded)\n")
    ofile.close()
    print("Writing model weights and biases into %s_*.npy NumPy arrays" % modelfile)
    print("")
    if layers == 2:
      np.save(file=modelfile+"_1.npy", arr=codecvs.layers[1].get_weights())
      np.save(file=modelfile+"_2.npy", arr=codecvs.layers[2].get_weights())
    if layers == 3:
      np.save(file=modelfile+"_1.npy", arr=codecvs.layers[1].get_weights())
      np.save(file=modelfile+"_2.npy", arr=codecvs.layers[2].get_weights())
      np.save(file=modelfile+"_3.npy", arr=codecvs.layers[3].get_weights())
    else:
      np.save(file=modelfile+"_1.npy", arr=codecvs.layers[1].get_weights())
      np.save(file=modelfile+"_2.npy", arr=codecvs.layers[2].get_weights())
      np.save(file=modelfile+"_3.npy", arr=codecvs.layers[3].get_weights())
      np.save(file=modelfile+"_4.npy", arr=codecvs.layers[4].get_weights())
  
  if plumedfile != '':
    print("Writing Plumed input into %s" % plumedfile)
    print("")
    traj = md.load(infilename, top=intopname)
    table, bonds = traj.topology.to_dataframe()
    atoms = table['serial'][:]
    ofile = open(plumedfile, "w")
    ofile.write("WHOLEMOLECULES ENTITY0=1-%i\n" % np.max(atoms))
    ofile.write("FIT_TO_TEMPLATE STRIDE=1 REFERENCE=%s TYPE=OPTIMAL\n" % intopname)
    for i in range(trajsize[1]):
      ofile.write("p%i: POSITION ATOM=%i\n" % (i+1,atoms[i]))
    for i in range(trajsize[1]):
      ofile.write("p%ix: COMBINE ARG=p%i.x COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iy: COMBINE ARG=p%i.y COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iz: COMBINE ARG=p%i.z COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
    if layers==2:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        onebias = codecvs.layers[1].get_weights()[1][i]
        if onebias>0.0:
          if actfun1 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun1 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun1 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun1 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun1 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      toprint = "l2: COMBINE ARG="
      for j in range(layer1):
        toprint = toprint + "l1r_%i," % (j+1)
      toprint = toprint[:-1] + " COEFFICIENTS="
      for j in range(layer1):
        toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j])
      toprint = toprint[:-1] + " PERIODIC=NO\n"
      ofile.write(toprint)
      if codecvs.layers[2].get_weights()[1][0]>0.0:
        ofile.write("l2r: MATHEVAL ARG=l2 FUNC=x+%0.6f PERIODIC=NO\n" % (codecvs.layers[2].get_weights()[1][0]))
      else:
        ofile.write("l2r: MATHEVAL ARG=l2 FUNC=x-%0.6f PERIODIC=NO\n" % (-codecvs.layers[2].get_weights()[1][0]))
      toprint = "PRINT ARG=l2r STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==3:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        onebias = codecvs.layers[1].get_weights()[1][i]
        if onebias>0.0:
          if actfun1 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun1 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun1 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun1 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun1 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer2):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer2):
        onebias = codecvs.layers[2].get_weights()[1][i]
        if onebias>0.0:
          if actfun2 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun2 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun2 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun2 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun2 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l2r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      toprint = "l3: COMBINE ARG="
      for j in range(layer2):
        toprint = toprint + "l2r_%i," % (j+1)
      toprint = toprint[:-1] + " COEFFICIENTS="
      for j in range(layer2):
        toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j])
      toprint = toprint[:-1] + " PERIODIC=NO\n"
      ofile.write(toprint)
      if codecvs.layers[3].get_weights()[1][0]>0.0:
        ofile.write("l3r: MATHEVAL ARG=l3 FUNC=x+%0.6f PERIODIC=NO\n" % (codecvs.layers[3].get_weights()[1][0]))
      else:
        ofile.write("l3r: MATHEVAL ARG=l3 FUNC=x-%0.6f PERIODIC=NO\n" % (-codecvs.layers[3].get_weights()[1][0]))
      toprint = "PRINT ARG=l3r STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==4:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        onebias = codecvs.layers[1].get_weights()[1][i]
        if onebias>0.0:
          if actfun1 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun1 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun1 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun1 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun1 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer2):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer2):
        onebias = codecvs.layers[2].get_weights()[1][i]
        if onebias>0.0:
          if actfun2 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun2 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun2 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun2 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun2 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l2r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer3):
        toprint = "l3_%i: COMBINE ARG=" % (i+1)
        for j in range(layer2):
          toprint = toprint + "l2r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer2):
          toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer3):
        onebias = codecvs.layers[3].get_weights()[1][i]
        if onebias>0.0:
          if actfun3 == 'elu': printfun = "(exp(x+%0.6f)-1.0)*step(-x-%0.6f)+(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun3 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.6f)-1.67326)*step(-x-%0.6f)+1.0507*(x+%0.6f)*step(x+%0.6f)" % (onebias,onebias,onebias,onebias)
          elif actfun3 == 'softplus': printfun = "log(1.0+exp(x+%0.6f))" % (onebias)
          elif actfun3 == 'softsign': printfun = "(x+%0.6f)/(1.0+step(x+%0.6f)*(x+%0.6f)+step(-x-%0.6f)*(-x-%0.6f))" % (onebias,onebias,onebias,onebias,onebias)
          elif actfun3 == 'relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (onebias,onebias)
          elif actfun3 == 'tanh': printfun = "(exp(x+%0.6f)-exp(-x-%0.6f))/(exp(x+%0.6f)+exp(-x-%0.6f))" % (onebias,onebias,onebias,onebias)
          elif actfun3 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (onebias)
          elif actfun3 == 'hard_sigmoid': printfun = "step(x+2.5+%0.6f)*((0.2*(x+%0.6f)+0.5)-step(x-2.5+%0.6f)*(0.2*(x+%0.6f)-0.5))" % (onebias,onebias,onebias,onebias)
          elif actfun3 == 'linear': printfun = "(x-%0.6f)" % (onebias)
        else:
          if actfun3 == 'elu': printfun = "(exp(x-%0.6f)-1.0)*step(-x+%0.6f)+(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun3 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.6f)-1.67326)*step(-x+%0.6f)+1.0507*(x-%0.6f)*step(x-%0.6f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun3 == 'softplus': printfun = "log(1.0+exp(x-%0.6f))" % (-onebias)
          elif actfun3 == 'softsign': printfun = "(x-%0.6f)/(1.0+step(x-%0.6f)*(x-%0.6f)+step(-x+%0.6f)*(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias,-onebias)
          elif actfun3 == 'relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-onebias,-onebias)
          elif actfun3 == 'tanh': printfun = "(exp(x-%0.6f)-exp(-x+%0.6f))/(exp(x-%0.6f)+exp(-x+%0.6f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun3 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-onebias)
          elif actfun3 == 'hard_sigmoid': printfun = "step(x+2.5-%0.6f)*((0.2*(x-%0.6f)+0.5)-step(x-2.5-%0.6f)*(0.2*(x-%0.6f)-0.5))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun3 == 'linear': printfun = "(x+%0.6f)" % (-onebias)
        ofile.write("l3r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      #for i in range(encdim):
      toprint = "l4: COMBINE ARG="
      for j in range(layer3):
        toprint = toprint + "l3r_%i," % (j+1)
      toprint = toprint[:-1] + " COEFFICIENTS="
      for j in range(layer3):
        toprint = toprint + "%0.6f," % (codecvs.layers[4].get_weights()[0][j])
      toprint = toprint[:-1] + " PERIODIC=NO\n"
      ofile.write(toprint)
      #for i in range(encdim):
      if codecvs.layers[4].get_weights()[1][0]>0.0:
        ofile.write("l4r: MATHEVAL ARG=l4 FUNC=x+%0.6f PERIODIC=NO\n" % (codecvs.layers[4].get_weights()[1][0]))
      else:
        ofile.write("l4r: MATHEVAL ARG=l4 FUNC=x-%0.6f PERIODIC=NO\n" % (-codecvs.layers[4].get_weights()[1][0]))
      toprint = "PRINT ARG=l4r STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    ofile.close()
  return codecvs, np.corrcoef(cvs,coded_cvs[:,0])[0,1]

