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

def deepcollectivevariable(infilename='', intopname='', colvarname='', column=2,
                           boxx=0.0, boxy=0.0, boxz=0.0, atestset=0.1,
                           shuffle=1, layers=2, layer1=256, layer2=256, layer3=256,
                           actfun1='sigmoid', actfun2='sigmoid', actfun2='sigmoid',
                           optim='adam', loss='mean_squared_error', epochs=100, batch=0,
                           lowfilename='', lowfiletype=0, highfilename='', highfiletype=0,
                           ofilename='', modelfile='', plumedfile=''):
  try:
    traj = md.load(infilename, top=intopname)
  except:
    print("Cannot load %s or %s, exiting." % (infilename, intopname))
    exit(0)
  else:
    print("%s succesfully loaded" % traj)
  print()
  
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
    print()
  
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
  print()
  
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
  
  # (Deep) learning  
  input_coord = krs.layers.Input(shape=(trajsize[1]*3,))
  encoded = krs.layers.Dense(layer1, activation=actfun1, use_bias=True)(input_coord)
  if layers == 4:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=True)(encoded)
    encoded = krs.layers.Dense(layer3, activation=actfun3, use_bias=True)(encoded)
  if layers == 3:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=True)(encoded)
  encoded = krs.layers.Dense(encdim, activation='linear', use_bias=True)(encoded)
  if layers == 3:
    decoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=True)(encoded)
    decoded = krs.layers.Dense(layer1, activation=actfun1, use_bias=True)(decoded)
  else:
    decoded = krs.layers.Dense(layer1, activation=actfun1, use_bias=True)(encoded)
  decoded = krs.layers.Dense(trajsize[1]*3, activation='linear', use_bias=True)(decoded)
  autoencoder = krs.models.Model(input_coord, decoded)
  
  encoder = krs.models.Model(input_coord, encoded)
  
  encoded_input = krs.layers.Input(shape=(encdim,))
  if layers == 2:
    decoder_layer1 = autoencoder.layers[-2]
    decoder_layer2 = autoencoder.layers[-1]
    decoder = krs.models.Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
  if layers == 3:
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    decoder = krs.models.Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
  
  autoencoder.compile(optimizer=optim, loss=loss)
  
  if batch>0:
    autoencoder.fit(training_set, training_set,
                    epochs=epochs,
                    batch_size=batch,
                    validation_data=(testing_set, testing_set))
  else:
    autoencoder.fit(training_set, training_set,
                    epochs=epochs,
                    validation_data=(testing_set, testing_set))
  
  # Encoding and decoding the trajectory
  encoded_coords = encoder.predict(traj2/maxbox)
  decoded_coords = decoder.predict(encoded_coords)
  
  # Calculating Pearson correlation coefficient
  vec1 = traj2.reshape((trajsize[0]*trajsize[1]*3))
  vec2 = decoded_coords.reshape((trajsize[0]*trajsize[1]*3))*maxbox
  print()
  print("Pearson correlation coefficient for encoded-decoded trajectory is %f" % np.corrcoef(vec1,vec2)[0,1])
  print()
  
  #training_set, testing_set = traj2[indexes[:-testsize],:]/maxbox, traj2[indexes[-testsize:],:]/maxbox
  vec1 = traj2[indexes[:-testsize],:].reshape(((trajsize[0]-testsize)*trajsize[1]*3))
  vec2 = decoded_coords[indexes[:-testsize],:].reshape(((trajsize[0]-testsize)*trajsize[1]*3))*maxbox
  print("Pearson correlation coefficient for encoded-decoded training set is %f" % np.corrcoef(vec1,vec2)[0,1])
  print()
  
  vec1 = traj2[indexes[-testsize:],:].reshape((testsize*trajsize[1]*3))
  vec2 = decoded_coords[indexes[-testsize:],:].reshape((testsize*trajsize[1]*3))*maxbox
  print("Pearson correlation coefficient for encoded-decoded testing set is %f" % np.corrcoef(vec1,vec2)[0,1])
  print() 
  
  # Generating low-dimensional output
  if lowfiletype > 0:
    print("Writing low-dimensional embeddings into %s" % lowfilename)
    print()
    if lowfiletype == 1:
      ofile = open(lowfilename, "w")
      ofile.write("# This file was created on %s\n" % dt.datetime.now().isoformat())
      ofile.write("# Created by: encodetraj.py V 0.1\n")
      sysargv = ""
      for item in sys.argv:
        sysargv = sysargv+item+" "
      ofile.write("# Command line: %s\n" % sysargv)
      ofile.write("@TYPE xy\n")
      ofile.write("@ title \"Autoencoded trajectory\"\n")
      ofile.write("@ xaxis  label \"low-dimensional embedding 1\"\n")
      ofile.write("@ yaxis  label \"low-dimensional embedding 2\"\n")
      for i in range(trajsize[0]):
        for j in range(encdim):
          ofile.write("%f " % encoded_coords[i][j])
        typeofset = 'TE'
        if i in indexes[:-testsize]:
          typeofset = 'TR'
        ofile.write("%s \n" % typeofset)
      ofile.close()
    if lowfiletype == 2:
      ofile = open(lowfilename, "w")
      for i in range(trajsize[0]):
        for j in range(encdim):
          ofile.write("%f " % encoded_coords[i][j])
        typeofset = 'TE'
        if i in indexes[:-testsize]:
          typeofset = 'TR'
        ofile.write("%s \n" % typeofset)
      ofile.close()
  
  # Generating high-dimensional output
  if highfiletype > 0:
    print("Writing original and encoded-decoded coordinates into %s" % highfilename)
    print()
    if highfiletype == 1:
      ofile = open(highfilename, "w")
      ofile.write("# This file was created on %s\n" % dt.datetime.now().isoformat())
      ofile.write("# Created by: encodetraj.py V 0.1\n")
      sysargv = ''
      for item in sys.argv:
        sysargv = sysargv+item+' '
      ofile.write("# Command line: %s\n" % sysargv)
      ofile.write("@TYPE xy\n")
      ofile.write("@ title \"Autoencoded and decoded trajectory\"\n")
      ofile.write("@ xaxis  label \"original coordinate\"\n")
      ofile.write("@ yaxis  label \"encoded and decoded coordinate\"\n")
      for i in range(trajsize[0]):
        for j in range(trajsize[1]*3):
          ofile.write("%f %f " % (traj2[i][j], decoded_coords[i][j]*maxbox))
          typeofset = 'TE'
          if i in indexes[:-testsize]:
            typeofset = 'TR'
          ofile.write("%s \n" % typeofset)
      ofile.close()
    if highfiletype == 2:
      ofile = open(highfilename, "w")
      for i in range(trajsize[0]):
        for j in range(trajsize[1]*3):
          ofile.write("%f %f " % (traj2[i][j], decoded_coords[i][j]*maxbox))
          typeofset = 'TE'
          if i in indexes[:-testsize]:
            typeofset = 'TR'
          ofile.write("%s \n" % typeofset)
      ofile.close()
  
  # Generating filtered trajectory
  if filterfilename != '':
    print("Writing encoded-decoded trajectory into %s" % filterfilename)
    print()
    decoded_coords2 = np.zeros((trajsize[0], trajsize[1], 3))
    for i in range(trajsize[1]):
      decoded_coords2[:,i,0] = decoded_coords[:,3*i]*maxbox
      decoded_coords2[:,i,1] = decoded_coords[:,3*i+1]*maxbox
      decoded_coords2[:,i,2] = decoded_coords[:,3*i+2]*maxbox
    traj.xyz = decoded_coords2
    traj.save_xtc(filterfilename)
  
  # Saving a plot of the model
  if plotfiletype == 1:
    print("Printing model into %s" % plotfilename)
    print()
    krs.utils.plot_model(autoencoder, show_shapes=True, to_file=plotfilename)
  
  # Saving the model
  if modelfile != '':
    print("Writing model into %s.txt" % modelfile)
    print()
    ofile = open(modelfile+'.txt', "w")
    ofile.write("maxbox = %f\n" % maxbox)
    ofile.write("input_coord = krs.layers.Input(shape=(trajsize[1]*3,))\n")
    ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(input_coord)\n" % (layer1, actfun1))
    if layers == 3:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer2, actfun2))
    ofile.write("encoded = krs.layers.Dense(%i, activation='linear', use_bias=True)(encoded)\n" % encdim)
    if layers == 3:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer2, actfun2))
    ofile.write("decoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (layer1, actfun1))
    ofile.write("decoded = krs.layers.Dense(trajsize[1]*3, activation='linear', use_bias=True)(decoded)\n")
    ofile.write("autoencoder = krs.models.Model(input_coord, decoded)\n")
    ofile.close()
    print("Writing model weights and biases into %s_*.npy NumPy arrays" % modelfile)
    print()
    if layers == 2:
      np.save(file=modelfile+"_1.npy", arr=autoencoder.layers[1].get_weights())
      np.save(file=modelfile+"_2.npy", arr=autoencoder.layers[2].get_weights())
      np.save(file=modelfile+"_3.npy", arr=autoencoder.layers[3].get_weights())
      np.save(file=modelfile+"_4.npy", arr=autoencoder.layers[4].get_weights())
    else:
      np.save(file=modelfile+"_1.npy", arr=autoencoder.layers[1].get_weights())
      np.save(file=modelfile+"_2.npy", arr=autoencoder.layers[2].get_weights())
      np.save(file=modelfile+"_3.npy", arr=autoencoder.layers[3].get_weights())
      np.save(file=modelfile+"_4.npy", arr=autoencoder.layers[4].get_weights())
      np.save(file=modelfile+"_5.npy", arr=autoencoder.layers[5].get_weights())
      np.save(file=modelfile+"_6.npy", arr=autoencoder.layers[6].get_weights())
  
  # Saving collective motions trajectories
  if collectivefile != '':
    if collectivefile[-4:] == '.xtc':
      collectivefile = collectivefile[:-4]
    traj = traj[:ncollective]
    print("Writing collective motion into %s_1.xtc" % collectivefile)
    print()
    collective = np.zeros((ncollective, 3))
    cvmin = np.amin(encoded_coords[:,0])
    cvmax = np.amax(encoded_coords[:,0])
    for i in range(ncollective):
      collective[i,0] = cvmin+(cvmax-cvmin)*float(i)/float(ncollective-1)
      collective[i,1] = np.mean(encoded_coords[:,1])
      collective[i,2] = np.mean(encoded_coords[:,2])
    collective2 = decoder.predict(collective)
    collective3 = np.zeros((ncollective, trajsize[1], 3))
    for i in range(trajsize[1]):
      collective3[:,i,0] = collective2[:,3*i]*maxbox
      collective3[:,i,1] = collective2[:,3*i+1]*maxbox
      collective3[:,i,2] = collective2[:,3*i+2]*maxbox
    traj.xyz = collective3
    traj.save_xtc(collectivefile+"_1.xtc")
    print("Writing collective motion into %s_2.xtc" % collectivefile)
    print()
    collective = np.zeros((ncollective, 3))
    cvmin = np.amin(encoded_coords[:,1])
    cvmax = np.amax(encoded_coords[:,1])
    for i in range(ncollective):
      collective[i,0] = np.mean(encoded_coords[:,0])
      collective[i,1] = cvmin+(cvmax-cvmin)*float(i)/float(ncollective-1)
      collective[i,2] = np.mean(encoded_coords[:,2])
    collective2 = decoder.predict(collective)
    collective3 = np.zeros((ncollective, trajsize[1], 3))
    for i in range(trajsize[1]):
      collective3[:,i,0] = collective2[:,3*i]*maxbox
      collective3[:,i,1] = collective2[:,3*i+1]*maxbox
      collective3[:,i,2] = collective2[:,3*i+2]*maxbox
    traj.xyz = collective3
    traj.save_xtc(collectivefile+"_2.xtc")
    print("Writing collective motion into %s_3.xtc" % collectivefile)
    print()
    collective = np.zeros((ncollective, 3))
    cvmin = np.amin(encoded_coords[:,2])
    cvmax = np.amax(encoded_coords[:,2])
    for i in range(args.ncollective):
      collective[i,0] = np.mean(encoded_coords[:,0])
      collective[i,1] = np.mean(encoded_coords[:,1])
      collective[i,2] = cvmin+(cvmax-cvmin)*float(i)/float(ncollective-1)
    collective2 = decoder.predict(collective)
    collective3 = np.zeros((ncollective, trajsize[1], 3))
    for i in range(trajsize[1]):
      collective3[:,i,0] = collective2[:,3*i]*maxbox
      collective3[:,i,1] = collective2[:,3*i+1]*maxbox
      collective3[:,i,2] = collective2[:,3*i+2]*maxbox
    traj.xyz = collective3
    traj.save_xtc(collectivefile+"_3.xtc")
  
  if plumedfile != '':
    print("Writing Plumed input into %s" % plumedfile)
    print()
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
          toprint = toprint + "%0.5f," % (autoencoder.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        onebias = autoencoder.layers[1].get_weights()[1][i]
        if onebias>0.0:
          if actfun1 == 'elu': printfun = "(exp(x+%0.5f)-1.0)*step(-x-%0.5f)+(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.5f)-1.67326)*step(-x-%0.5f)+1.0507*(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x+%0.5f))" % (onebias)
          elif actfun1 == 'softsign': printfun = "(x+%0.5f)/(1.0+step(x+%0.5f)*(x+%0.5f))" % (onebias,onebias,onebias)
          elif actfun1 == 'relu': printfun = "step(x+%0.5f)*(x+%0.5f)" % (onebias,onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x+%0.5f)-exp(-x-%0.5f))/(exp(x+%0.5f)+exp(-x-%0.5f))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.5f))" % (onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5+%0.5f)*step(x-2.5+%0.5f)*(0.2*(x+%0.5f)+0.5) + step(x-2.5+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'linear': printfun = "(x-%0.5f)" % (onebias)
        else:
          if actfun1 == 'elu': printfun = "(exp(x-%0.5f)-1.0)*step(-x+%0.5f)+(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.5f)-1.67326)*step(-x+%0.5f)+1.0507*(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x-%0.5f))" % (-onebias)
          elif actfun1 == 'softsign': printfun = "(x-%0.5f)/(1.0+step(x-%0.5f)*(x-%0.5f))" % (-onebias,-onebias,-onebias)
          elif actfun1 == 'relu': printfun = "step(x-%0.5f)*(x-%0.5f)" % (-onebias,-onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x-%0.5f)-exp(-x+%0.5f))/(exp(x-%0.5f)+exp(-x+%0.5f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.5f))" % (-onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5-%0.5f)*step(x-2.5-%0.5f)*(0.2*(x-%0.5f)+0.5) + step(x-2.5-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'linear': printfun = "(x+%0.5f)" % (-onebias)
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(encdim):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer2):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer2):
          toprint = toprint + "%0.5f," % (autoencoder.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(encdim):
        if autoencoder.layers[2].get_weights()[1][i]>0.0:
          ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=x+%0.5f PERIODIC=NO\n" % (i+1,i+1,autoencoder.layers[2].get_weights()[1][i]))
        else:
          ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=x-%0.5f PERIODIC=NO\n" % (i+1,i+1,-autoencoder.layers[2].get_weights()[1][i]))
      toprint = "PRINT ARG="
      for i in range(encdim):
        toprint = toprint + "l2r_%i," % (i+1)
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==3:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.5f," % (autoencoder.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        onebias = autoencoder.layers[1].get_weights()[1][i]
        if onebias>0.0:
          if actfun1 == 'elu': printfun = "(exp(x+%0.5f)-1.0)*step(-x-%0.5f)+(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.5f)-1.67326)*step(-x-%0.5f)+1.0507*(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x+%0.5f))" % (onebias)
          elif actfun1 == 'softsign': printfun = "(x+%0.5f)/(1.0+step(x+%0.5f)*(x+%0.5f))" % (onebias,onebias,onebias)
          elif actfun1 == 'relu': printfun = "step(x+%0.5f)*(x+%0.5f)" % (onebias,onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x+%0.5f)-exp(-x-%0.5f))/(exp(x+%0.5f)+exp(-x-%0.5f))" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.5f))" % (onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5+%0.5f)*step(x-2.5+%0.5f)*(0.2*(x+%0.5f)+0.5) + step(x-2.5+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun1 == 'linear': printfun = "(x-%0.5f)" % (onebias)
        else:
          if actfun1 == 'elu': printfun = "(exp(x-%0.5f)-1.0)*step(-x+%0.5f)+(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.5f)-1.67326)*step(-x+%0.5f)+1.0507*(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'softplus': printfun = "log(1.0+exp(x-%0.5f))" % (-onebias)
          elif actfun1 == 'softsign': printfun = "(x-%0.5f)/(1.0+step(x-%0.5f)*(x-%0.5f))" % (-onebias,-onebias,-onebias)
          elif actfun1 == 'relu': printfun = "step(x-%0.5f)*(x-%0.5f)" % (-onebias,-onebias)
          elif actfun1 == 'tanh': printfun = "(exp(x-%0.5f)-exp(-x+%0.5f))/(exp(x-%0.5f)+exp(-x+%0.5f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.5f))" % (-onebias)
          elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5-%0.5f)*step(x-2.5-%0.5f)*(0.2*(x-%0.5f)+0.5) + step(x-2.5-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun1 == 'linear': printfun = "(x+%0.5f)" % (-onebias)
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer2):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "1lr_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.5f," % (autoencoder.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer2):
        onebias = autoencoder.layers[2].get_weights()[1][i]
        if onebias>0.0:
          if actfun2 == 'elu': printfun = "(exp(x+%0.5f)-1.0)*step(-x-%0.5f)+(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x+%0.5f)-1.67326)*step(-x-%0.5f)+1.0507*(x+%0.5f)*step(x+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x+%0.5f))" % (onebias)
          elif actfun2 == 'softsign': printfun = "(x+%0.5f)/(1.0+step(x+%0.5f)*(x+%0.5f))" % (onebias,onebias,onebias)
          elif actfun2 == 'relu': printfun = "step(x+%0.5f)*(x+%0.5f)" % (onebias,onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x+%0.5f)-exp(-x-%0.5f))/(exp(x+%0.5f)+exp(-x-%0.5f))" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x-%0.5f))" % (onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5+%0.5f)*step(x-2.5+%0.5f)*(0.2*(x+%0.5f)+0.5) + step(x-2.5+%0.5f)" % (onebias,onebias,onebias,onebias)
          elif actfun2 == 'linear': printfun = "(x-%0.5f)" % (onebias)
        else:
          if actfun2 == 'elu': printfun = "(exp(x-%0.5f)-1.0)*step(-x+%0.5f)+(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x-%0.5f)-1.67326)*step(-x+%0.5f)+1.0507*(x-%0.5f)*step(x-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'softplus': printfun = "log(1.0+exp(x-%0.5f))" % (-onebias)
          elif actfun2 == 'softsign': printfun = "(x-%0.5f)/(1.0+step(x-%0.5f)*(x-%0.5f))" % (-onebias,-onebias,-onebias)
          elif actfun2 == 'relu': printfun = "step(x-%0.5f)*(x-%0.5f)" % (-onebias,-onebias)
          elif actfun2 == 'tanh': printfun = "(exp(x-%0.5f)-exp(-x+%0.5f))/(exp(x-%0.5f)+exp(-x+%0.5f))" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x+%0.5f))" % (-onebias)
          elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5-%0.5f)*step(x-2.5-%0.5f)*(0.2*(x-%0.5f)+0.5) + step(x-2.5-%0.5f)" % (-onebias,-onebias,-onebias,-onebias)
          elif actfun2 == 'linear': printfun = "(x+%0.5f)" % (-onebias)
        ofile.write("l2r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(encdim):
        toprint = "l3_%i: COMBINE ARG=" % (i+1)
        for j in range(layer2):
          toprint = toprint + "l2r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer2):
          toprint = toprint + "%0.5f," % (autoencoder.layers[3].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(encdim):
        if autoencoder.layers[2].get_weights()[1][i]>0.0:
          ofile.write("l3r_%i: MATHEVAL ARG=l3_%i FUNC=x+%0.5f PERIODIC=NO\n" % (i+1,i+1,autoencoder.layers[3].get_weights()[1][i]))
        else:
          ofile.write("l3r_%i: MATHEVAL ARG=l3_%i FUNC=x-%0.5f PERIODIC=NO\n" % (i+1,i+1,-autoencoder.layers[3].get_weights()[1][i]))
      toprint = "PRINT ARG="
      for i in range(encdim):
        toprint = toprint + "l3r_%i," % (i+1)
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    ofile.close()
  return autoencoder, np.corrcoef(vec1,vec2)[0,1]

