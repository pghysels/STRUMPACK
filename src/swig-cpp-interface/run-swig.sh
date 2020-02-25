#!/bin/sh

if [ "$HOSTNAME" == "vostok" ]; then
  export PATH=/rnsdhpc/code/build/swig-debug:$PATH
  export SWIG_LIB=/rnsdhpc/code/src/swig/Lib
fi

exec swig -fortran -c++ -I../ \
  -fext f03 \
  -outdir generated -o generated/strumpackFORTRAN_wrap.cxx \
  strumpack.i
