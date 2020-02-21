#!/bin/sh

if [ "$HOSTNAME" == "vostok" ]; then
  export PATH=/rnsdhpc/code/build/swig-debug:$PATH
  export SWIG_LIB=/rnsdhpc/code/src/swig/Lib
fi

# Since this "flat" interface simply provides fortran interface function and no
# wrapper code, we can ignore the generated wrap.c file .
exec swig -fortran \
  -outdir generated -o /dev/null strumpack_flat.i
