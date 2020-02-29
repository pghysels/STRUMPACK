#!/usr/bin/env bash

rm -rf build
rm -rf install
mkdir build
mkdir install
cd build

# METIS is required
export METIS_DIR=$HOME/local/metis-5.1.0/install

# the following are optional
export SCOTCH_DIR=$HOME/local/scotch_6.0.4
export ParMETIS_DIR=$HOME/local/parmetis-4.0.3/install
export ZFP_DIR=$HOME/local/zfp-0.5.5/install
export ButterflyPACK_DIR=$HOME/LBL/STRUMPACK/ButterflyPACK_export/install/lib/cmake/ButterflyPACK

cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install

## if not found automatically, you can specify BLAS/LAPACK/SCALAPACK as:
#  -DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libopenblas.a"
#  -DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.a"
#  -DTPL_SCALAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so"


make install -j4
make examples -j4
make tests -j4
# make test
