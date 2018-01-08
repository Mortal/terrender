# setup.py based on https://github.com/getsentry/milksnake
from setuptools import setup
from terrender import DESCRIPTION


headline = DESCRIPTION.split('\n', 1)[0].rstrip('.')


def build_native(spec):
    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
        path='./native',
    )

    spec.add_cffi_module(
        module_path='terrender._native',
        dylib=lambda: build.find_dylib('terrender', in_path='target/release'),
        header_filename=lambda: build.find_header('terrender.h', in_path='include'),
    )


setup(name='terrender',
      version='0.2.0',
      description=headline,
      long_description=DESCRIPTION,
      author='https://github.com/Mortal',
      url='https://github.com/Mortal/terrender',
      packages=['terrender'],
      include_package_data=True,
      zip_safe=False,
      platforms='any',
      install_requires=['milksnake'],
      setup_requires=['milksnake'],
      milksnake_tasks=[
          build_native,
      ],
)
