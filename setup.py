# setup.py based on https://github.com/getsentry/milksnake
import os
import re
from terrender import DESCRIPTION
from setuptools import setup, Extension, find_packages


NAME = 'terrender'
headline = DESCRIPTION.split('\n', 1)[0].rstrip('.')


def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'Cargo.toml')) as fp:
        version_line = next(line for line in fp if line.startswith('version'))
    mo = re.match(r'version = "(.*)"', version_line)
    return mo.group(1)


def build_native(spec):
    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
    )

    spec.add_cffi_module(
        module_path='%s._native' % NAME,
        dylib=lambda: build.find_dylib(NAME, in_path='target/release'),
        header_filename=lambda: build.find_header('%s.h' % NAME, in_path='include'),
    )


if 'MANYLINUX' in os.environ:
    # Disable Cython so that the wheel is 'py2.py3'
    extensions = []
else:
    import numpy
    from Cython.Build import cythonize

    sourcefiles = ['terrender/_predicates.pyx', 'terrender/rectangle_sweep.cpp']
    extensions = cythonize(
        [Extension("terrender._predicates", sourcefiles,
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=['-static-libstdc++'],
                   extra_link_args=['-static-libstdc++'])])


setup(name=NAME,
      version=get_version(),
      description=headline,
      long_description=DESCRIPTION,
      author='https://github.com/Mortal',
      url='https://github.com/Mortal/%s' % NAME,
      packages=find_packages(include=[NAME, NAME + '.*']),
      include_package_data=True,
      zip_safe=False,
      platforms='any',
      install_requires=['numpy', 'milksnake'],
      setup_requires=['numpy', 'milksnake'],
      milksnake_tasks=[
          build_native,
      ],
      ext_modules=extensions,
)
