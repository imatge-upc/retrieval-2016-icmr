#!/usr/bin/env python
#
# To build the C++ extension modules, run::
#
#   $ python setup.py build_ext --inplace
#
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
from numpy.distutils.misc_util import get_numpy_include_dirs

invidx_extension = Extension(
    name = "invidx._invidx", 
    sources = ["invidx/_invidx.pyx"] + glob("invidx/cpp/*.cpp"),
    language = "c++", 
    include_dirs = ['invidx', "invidx/cpp"] + get_numpy_include_dirs(),
    extra_compile_args=["-std=c++11"]
)

setup(
    packages = ['invidx'],
    name = 'invidx',
    version = '1.0',
    description = 'Inverted index implementation',
    cmdclass = { 'build_ext': build_ext },
    ext_modules = [invidx_extension],
)