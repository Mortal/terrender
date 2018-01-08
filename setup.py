from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from terrender import DESCRIPTION


headline = DESCRIPTION.split('\n', 1)[0].rstrip('.')


sourcefiles = ['terrender/_predicates.pyx', 'terrender/rectangle_sweep.cpp']

extensions = [Extension("terrender._predicates", sourcefiles)]

setup(name='terrender',
      version='0.1',
      description=headline,
      long_description=DESCRIPTION,
      author='https://github.com/Mortal',
      url='https://github.com/Mortal/terrender',
      packages=['terrender'],
      ext_modules=cythonize(extensions),
)
