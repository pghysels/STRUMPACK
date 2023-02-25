#!/bin/sh

if [ "$HOSTNAME" == "cs-it-7098760" ]; then
    export PATH=/home/pieterg/local/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.0.1/swig-fortran-syvd3hmyvd5rghktoxqjgkairf74d4jk/bin/:$PATH
fi
which swigfortran

# Since this "flat" interface simply provides fortran interface function and no
# wrapper code, we can ignore the generated wrap.c file .
swigfortran -fortran -outdir . -o /dev/null strumpack_dense.i
swigfortran -fortran -outdir . -o strumpack_dense_mpi.c strumpack_dense_mpi.i
