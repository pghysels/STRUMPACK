#!/bin/bash

cd ..
rm -rf STRUMPACK-build
rm -rf STRUMPACK-install
mkdir STRUMPACK-build
mkdir STRUMPACK-install

cd STRUMPACK-build

found_host=false

if [[ $NERSC_HOST = "edison" ]]; then
    found_host=true
    #  -DCMAKE_CXX_FLAGS="-DUSE_TASK_TIMER -DCOUNT_FLOPS" \
    cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
          -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
          -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
          -DMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/metis/include \
          -DMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DPARMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/include \
          -DPARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DSCOTCH_INCLUDES=$HOME/local/scotch_6.0.4/include \
          -DSCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4/lib/libscotch.a;$HOME/local/scotch_6.0.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4/lib/libptscotcherr.a"
fi

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true
    #  -DCMAKE_CXX_FLAGS="-DUSE_TASK_TIMER -DCOUNT_FLOPS" \
    cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
          -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
          -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
          -DMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/metis/include \
          -DMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DPARMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/include \
          -DPARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DSCOTCH_INCLUDES=$HOME/local/scotch_6.0.4/include \
          -DSCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4/lib/libscotch.a;$HOME/local/scotch_6.0.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4/lib/libptscotcherr.a"
fi

if [[ $(hostname -s) = "xps13" ]]; then
    found_host=true
    cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
          -DMETIS_INCLUDES=/home/pieterg/local/parmetis-4.0.3/metis/include \
          -DMETIS_LIBRARIES=/home/pieterg/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DPARMETIS_INCLUDES=/home/pieterg/local/parmetis-4.0.3/include \
          -DPARMETIS_LIBRARIES=/home/pieterg/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a
fi

if [[ $(hostname -s) = "pieterg-X8DA3" ]]; then
    found_host=true
    cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
          -DCMAKE_CXX_FLAGS="-DCOUNT_FLOPS -DUSE_TASK_TIMER -Wall -Wfatal-errors -Wextra -Wno-unused-parameter" \
          -DMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/metis/include \
          -DMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DPARMETIS_INCLUDES=$HOME/local/parmetis-4.0.3/include \
          -DPARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DSCOTCH_INCLUDES=$HOME/local/scotch_6.0.4/include \
          -DSCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4/lib/libscotch.a;$HOME/local/scotch_6.0.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4/lib/libptscotcherr.a"

    # cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
    #       -DUSE_OPENMP=OFF \
    #       -DMPI_CXX_COMPILER=/home/pieterg/local/mpich-3.1.4-install/bin/mpic++ \
    #       -DMPI_C_COMPILER=/home/pieterg/local/mpich-3.1.4-install/bin/mpicc \
    #       -DMPI_Fortran_COMPILER=/home/pieterg/local/mpich-3.1.4-install/bin/mpif90 \
    #       -DCMAKE_CXX_FLAGS="-DCOUNT_FLOPS -DUSE_TASK_TIMER -Wall -Wfatal-errors -Wextra -Wno-unused-parameter" \
    #       -DSCALAPACK_LIBRARIES="$HOME/local/scalapack-install/lib/libscalapack.a" \
    #       -DMETIS_INCLUDES=$HOME/local/parmetis-4.0.3_mpich-3.1.4/metis/include \
    #       -DMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3_mpich-3.1.4/build/Linux-x86_64/libmetis/libmetis.a \
    #       -DPARMETIS_INCLUDES=$HOME/local/parmetis-4.0.3_mpich-3.1.4/include \
    #       -DPARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3_mpich-3.1.4/build/Linux-x86_64/libparmetis/libparmetis.a \
    #       -DSCOTCH_INCLUDES=$HOME/local/scotch_6.0.4_mpich-3.1.4/include \
    #       -DSCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4_mpich-3.1.4/lib/libscotch.a;$HOME/local/scotch_6.0.4_mpich-3.1.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4_mpich-3.1.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4_mpich-3.1.4/lib/libptscotcherr.a"

    # cmake ../STRUMPACK -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../STRUMPACK-install \
    #       -DUSE_OPENMP=OFF \
    #       -DMPI_CXX_COMPILER=/home/pieterg/local/mpich-install/bin/mpic++ \
    #       -DMPI_C_COMPILER=/home/pieterg/local/mpich-install/bin/mpicc \
    #       -DMPI_Fortran_COMPILER=/home/pieterg/local/mpich-install/bin/mpif90 \
    #       -DCMAKE_CXX_FLAGS="-DCOUNT_FLOPS -DUSE_TASK_TIMER -Wall -Wfatal-errors -Wextra -Wno-unused-parameter" \
    #       -DSCALAPACK_LIBRARIES="$HOME/local/scalapack-install/lib/libscalapack.a" \
    #       -DMETIS_INCLUDES=$HOME/local/parmetis-4.0.3_mpich/metis/include \
    #       -DMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3_mpich/build/Linux-x86_64/libmetis/libmetis.a \
    #       -DPARMETIS_INCLUDES=$HOME/local/parmetis-4.0.3_mpich/include \
    #       -DPARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3_mpich/build/Linux-x86_64/libparmetis/libparmetis.a \
    #       -DSCOTCH_INCLUDES=$HOME/local/scotch_6.0.4_mpich/include \
    #       -DSCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4_mpich/lib/libscotch.a;$HOME/local/scotch_6.0.4_mpich/lib/libscotcherr.a;$HOME/local/scotch_6.0.4_mpich/lib/libptscotch.a;$HOME/local/scotch_6.0.4_mpich/lib/libptscotcherr.a"
fi


if ! $found_host; then
    echo "This machine was not recognized."
    echo "Open this file for examples on how to configure STRUMPACK."
    exit 1
fi

make install VERBOSE=1
cd examples
make -k
cd ../
