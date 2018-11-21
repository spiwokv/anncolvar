from setuptools import setup

def readme():
  with open('README.md') as f:
    return f.read()

def readme():
  with open('README.md') as f:
    return f.read()

setup(name='anncolvar',
      version='0.1',
      description='Coding collective variables by artificial neural networks',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
      ],
      keywords='artificial neural networks molecular dynamics simulation',
      url='https://github.com/spiwokv/anncolvar',
      author='Vojtech Spiwok, ',
      author_email='spiwokv@vscht.cz',
      license='MIT',
      packages=['anncolvar'],
      scripts=['bin/anncolvar'],
      install_requires=[
	  'absl-py',
       	  'astor',
	  'backports.weakref',
	  'Cython',
	  'DateTime',
	  'enum',
	  'enum34',
	  'funcsigs',
	  'futures',
	  'gast',
	  'grpcio',
	  'h5py',
	  'Keras',
	  'Keras-Applications',
	  'Keras-Preprocessing',
	  'Markdown',
	  'mdtraj',
	  'mock',
	  'numpy',
	  'pbr',
	  'protobuf',
	  'pytz',
	  'PyYAML',
	  'scipy',
	  'six',
	  'tensorboard',
	  'tensorflow',
	  'termcolor',
	  'Werkzeug',
	  'zope.interface',
      ],
      include_package_data=True,
      zip_safe=False)

