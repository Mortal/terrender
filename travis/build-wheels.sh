#!/bin/sh
# Based on https://github.com/getsentry/symbolic
# and https://github.com/pypa/python-manylinux-demo

set -e -x

# Install dependencies needed by our wheel
yum -y install libffi-devel
/opt/python/cp27-cp27mu/bin/pip install cython numpy

# Install GCC 7.1 (https://github.com/Noctem/pogeo/blob/develop/travis/manylinux-build.sh)
if [[ "$(uname -m)" = i686 ]]; then
	TOOLCHAIN_URL='https://github.com/Noctem/pogeo-toolchain/releases/download/v1.3/gcc-7.1-binutils-2.28-centos5-i686.tar.bz2'
	export LD_LIBRARY_PATH="/toolchain/lib:${LD_LIBRARY_PATH}"
	MFLAG="-m32"
else
	TOOLCHAIN_URL='https://github.com/Noctem/pogeo-toolchain/releases/download/v1.3/gcc-7.1-binutils-2.28-centos5-x86-64.tar.bz2'
	export LD_LIBRARY_PATH="/toolchain/lib64:/toolchain/lib:${LD_LIBRARY_PATH}"
	MFLAG="-m64"
fi

curl -L "$TOOLCHAIN_URL" -o toolchain.tar.bz2
tar -C / -xf toolchain.tar.bz2

export MANYLINUX=1
export PATH="/toolchain/bin:${PATH}"
export CFLAGS="-I/toolchain/include ${MFLAG}"
export CXXFLAGS="-I/toolchain/include ${MFLAG} -static-libstdc++"

# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH=~/.cargo/bin:$PATH

# Build wheels
cd /work
/opt/python/cp27-cp27mu/bin/python setup.py bdist_wheel --verbose

# Audit wheels
for wheel in dist/*-linux_*.whl; do
  auditwheel repair "$wheel" -w wheelhouse/ || auditwheel -v show "$wheel"
done
