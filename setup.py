from setuptools import setup

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='anncolvar',
      version='0.8',
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
          'numpy',
          'cython',
          'mdtraj',
          'keras',
          'argparse',
          'datetime',
          'codecov',
      ],
      include_package_data=True,
      zip_safe=False)

