#!/usr/bin/env bash

rm -rf build
rm -rf install
mkdir build
mkdir install

cd build

SCOTCHDIR=$HOME/local/scotch_6.0.4
BPACKDIR=$HOME/LBL/STRUMPACK/ButterflyPACK/
PARMETISDIR=$HOME/local/parmetis-4.0.3/install/
METISDIR=$HOME/local/metis-5.1.0/install/
ZFPDIR=$HOME/local/zfp-0.5.5/install

cmake ../ \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=../install \
      -DSTRUMPACK_USE_MPI=ON \
      -DTPL_METIS_PREFIX=$METISDIR \
      -DTPL_SCOTCH_PREFIX=$SCOTCHDIR \
      -DTPL_PTSCOTCH_PREFIX=$SCOTCHDIR \
      -DTPL_PARMETIS_PREFIX=$PARMETISDIR \
      -DTPL_ZFP_PREFIX=$ZFPDIR \
      -DTPL_ENABLE_BPACK=ON \
      -DTPL_BPACK_INCLUDE_DIRS="$BPACKDIR/SRC_DOUBLE/;$BPACKDIR/SRC_DOUBLECOMPLEX" \
      -DTPL_BPACK_LIBRARIES="-L$BPACKDIR/build/SRC_DOUBLE/ -ldbutterflypack -L$BPACKDIR/build/SRC_DOUBLECOMPLEX/ -lzbutterflypack" \

make install
