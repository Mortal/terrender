# Makefile based on https://github.com/getsentry/symbolic
NAME=terrender
PYTHON=python3

all: inplace

inplace:
	python setup.py build_ext --inplace

version.mk: Cargo.toml
	sed -ne 's/^version = "\(.*\)"$$/VERSION=\1/p' $< > $@ && [ -s $@ ] || ($(RM) $@; false)

include version.mk

wheel-install: wheel
	pip install --user -U dist/$(NAME)-$(VERSION)-py2.py3-none-linux_x86_64.whl

wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: all inplace wheel-install wheel
