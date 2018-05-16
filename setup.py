from setuptools import setup

def readme():
  with open('README.md') as f:
    return f.read()


setup(name='encodetraj',
      version='0.1',
      description='(Deep learning) autoencoders for molecular trajectory analysis',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
      ],
      keywords='autoencoder deep learning molecular dynamics simulation',
      url='https://github.com/spiwokv/encodetraj',
      author='Vojtech Spiwok, ',
      author_email='spiwokv@vscht.cz',
      license='MIT',
      packages=['encodetrajlib'],
      scripts=['bin/encodetraj'],
      install_requires=[
          'numpy',
          'cython',
          'mdtraj',
          'keras',
          'argparse',
          'datetime',
      ],
      include_package_data=True,
      zip_safe=False)

