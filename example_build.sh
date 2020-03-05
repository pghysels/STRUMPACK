#!/bin/bash

rm -rf build
rm -rf install
mkdir build
mkdir install

cd build

found_host=false


if [[ $(dnsdomainname) = "summit.olcf.ornl.gov" ]]; then
    found_host=true

    module unload cmake
    module swap xl gcc/9.1.0
    module load essl
    module load cuda
    module load netlib-lapack
    module load netlib-scalapack
    module load cmake
    module unload darshan-runtime

    # METIS is required
    export METIS_DIR=$HOME/local/metis-5.1.0/install

    # the following are optional
    export SCOTCH_DIR=$HOME/local/scotch_6.0.9
    export ParMETIS_DIR=$HOME/local/parmetis-4.0.3/install
    export ButterflyPACK_DIR=$HOME/ButterflyPACK/install

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DCMAKE_CXX_COMPILER=mpiCC \
          -DCMAKE_C_COMPILER=mpicc \
          -DCMAKE_Fortran_COMPILER=mpif90 \
          -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so" \
          -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so" \
          -DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
          -DTPL_ENABLE_BPACK=ON \
          -DTPL_ENABLE_CUBLAS=ON \
          -DTPL_ENABLE_SLATE=OFF
fi

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true

    module unload cmake
    module load cmake
    module remove darshan
    module swap PrgEnv-intel PrgEnv-gnu
    module unload gcc
    module load gcc

    ## for MKL, use the link advisor: https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
    ## separate arguments with ;, the -Wl,--start-group ... -Wl,--end-group is a single argument
    ## Intel compiler
    #ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-liomp5;-lpthread;-lm;-ldl"

    ## GNU compiler
    #ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-lgomp;-lpthread;-lm;-ldl"

    ## cray-libsci (module cray-libsci loaded) instead of MKL
    ScaLAPACKLIBS=""

    export METIS_DIR=$HOME/local/cori/gcc/metis-5.1.0/install

    # optional dependencies
    export ParMETIS_DIR=$HOME/local/cori/gcc/parmetis-4.0.3/install
    export SCOTCH_DIR=$HOME/local/cori/gcc/scotch_6.0.9
    export ButterflyPACK_DIR=$HOME/cori/ButterflyPACK/install
    export ZFP_DIR=$HOME/local/cori/gcc/zfp/install/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DCMAKE_CXX_COMPILER=CC \
          -DCMAKE_C_COMPILER=cc \
          -DCMAKE_Fortran_COMPILER=ftn \
          -DTPL_SCALAPACK_LIBRARIES="$ScaLAPACKLIBS"
fi

if [[ $(hostname -s) = "pieterg-X8DA3" ]]; then
    found_host=true

    # METIS is required
    export METIS_DIR=$HOME/local/metis-5.1.0/install

    # the following are optional
    export SCOTCH_DIR=$HOME/local/scotch_6.0.4
    export ParMETIS_DIR=$HOME/local/parmetis-4.0.3/install
    export ZFP_DIR=$HOME/local/zfp-0.5.5/install
    export ButterflyPACK_DIR=$HOME/LBL/STRUMPACK/ButterflyPACK/install/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DTPL_ENABLE_BPACK=ON \
          -DTPL_SCALAPACK_LIBRARIES="$HOME/local/scalapack-2.1.0/install/lib/libscalapack.a"
fi


if ! $found_host; then
    echo "This machine was not recognized."
    echo "Open this file and modify the CMake command."
    echo "Running CMake ..."

    # METIS is required, but might be already be installed by the system
    #export METIS_DIR=

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install

    ## if not found automatically, you can specify BLAS/LAPACK/SCALAPACK as:
    #  -DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libopenblas.a"
    #  -DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.a"
    #  -DTPL_SCALAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so"
fi

make install -j4
make examples -j4
make tests -j4
# make test
