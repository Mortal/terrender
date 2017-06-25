from distutils.core import setup
from Cython.Build import cythonize
from terrender import DESCRIPTION


headline = DESCRIPTION.split('\n', 1)[0].rstrip('.')


setup(name='terrender',
      version='0.1',
      description=headline,
      long_description=DESCRIPTION,
      author='https://github.com/Mortal',
      url='https://github.com/Mortal/terrastream-scripts',
      packages=['terrender'],
      ext_modules=cythonize('terrender/*.pyx'),
)
