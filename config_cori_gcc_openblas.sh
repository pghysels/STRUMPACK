#!/bin/bash
export CRAYPE_LINK_TYPE=dynamic


rm -rf build_openblas
rm -rf install
mkdir build_openblas
mkdir install

module swap PrgEnv-intel PrgEnv-gnu
module unload cmake
module load cmake
module rm darshan
module unload cray-libsci

FASTMATH=-Ofast


export ROOT=$PWD

export SCALAPACK_LIB=$ROOT/scalapack-2.1.0/build/install/lib/libscalapack.a
export BLAS_LIB=$ROOT/OpenBLAS/build/install/lib64/libopenblas.a
export LAPACK_LIB=$ROOT/OpenBLAS/build/install/lib64/libopenblas.a


# ###################################
# cd $ROOT
# git clone https://github.com/xianyi/OpenBLAS
# cd OpenBLAS
# rm -rf build
# mkdir -p build
# cd build
# cmake .. \
# 	-DBUILD_SHARED_LIBS=OFF \
# 	-DCMAKE_C_COMPILER=cc \
# 	-DCMAKE_Fortran_COMPILER=ftn \
# 	-DCMAKE_INSTALL_PREFIX=. \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_INSTALL_PREFIX=./install \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DCMAKE_Fortran_FLAGS="${FASTMATH} -fopenmp -fallow-argument-mismatch" \
#     -DCMAKE_C_FLAGS="${FASTMATH}" 
# make -j32
# make install


# cd $ROOT
# rm -rf scalapack-2.1.0.tgz*
# wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
# tar -xf scalapack-2.1.0.tgz
# cd scalapack-2.1.0
# rm -rf build
# mkdir -p build
# cd build
# cmake .. \
# 	-DBUILD_SHARED_LIBS=OFF \
# 	-DCMAKE_C_COMPILER=cc \
# 	-DCMAKE_Fortran_COMPILER=ftn \
# 	-DCMAKE_INSTALL_PREFIX=. \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_INSTALL_PREFIX=./install \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DCMAKE_Fortran_FLAGS="${FASTMATH} -fopenmp -fallow-argument-mismatch" \
#     -DCMAKE_CXX_FLAGS="${FASTMATH}" \
#     -DCMAKE_C_FLAGS="${FASTMATH}" \
# 	-DBLAS_LIBRARIES="${BLAS_LIB}" \
# 	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
# make -j32
# make install


# cd $ROOT
# rm -rf ButterflyPACK
# git clone https://github.com/liuyangzhuan/ButterflyPACK.git
# cd ButterflyPACK
# mkdir build 
# cd build
# cmake .. \
#     -DCMAKE_INSTALL_LIBDIR=./lib \
# 	-DCMAKE_Fortran_FLAGS="-DMPIMODULE" \
#     -DCMAKE_CXX_FLAGS="" \
# 	-DBUILD_SHARED_LIBS=OFF \
# 	-Denable_doc=OFF \
# 	-DCMAKE_Fortran_COMPILER=ftn \
# 	-DCMAKE_CXX_COMPILER=CC \
# 	-DCMAKE_C_COMPILER=cc \
# 	-DCMAKE_INSTALL_PREFIX=. \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
# 	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
# 	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
# make install -j16




cd $ROOT/build_openblas


found_host=false

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true

            export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
            export METIS_DIR=/global/homes/l/liuyangz/Cori/my_software/metis-5.1.0/install_gcc
#            export PARMETIS_DIR=/global/homes/l/liuyangz/Cori/my_software/parmetis-4.0.3/build/
            # optional dependencies
            # export ParMETIS_DIR=$HOME/local/cori/gcc/parmetis-4.0.3/install
            # export SCOTCH_DIR=$HOME/local/cori/gcc/scotch_6.0.9
            export ButterflyPACK_DIR=$ROOT/ButterflyPACK/build/lib/cmake/ButterflyPACK
            export ZFP_DIR=/project/projectdirs/m2957/liuyangz/my_research/zfp-0.5.5_gcc/build/lib/cmake/zfp

    ## Use cray-libsci (module cray-libsci loaded) instead of MKL.
    ## The problem with MKL is that it uses openmpi of intel MPI.

    cmake ../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="${FASTMATH}" \
        -DCMAKE_C_FLAGS="${FASTMATH}" \
        -DCMAKE_Fortran_FLAGS=" ${FASTMATH} -fallow-argument-mismatch" \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_CXX_COMPILER=CC \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_Fortran_COMPILER=ftn \
        -DSTRUMPACK_COUNT_FLOPS=ON \
        -DSTRUMPACK_TASK_TIMERS=ON \
        -DTPL_ENABLE_SCOTCH=OFF \
        -DTPL_ENABLE_ZFP=ON \
        -DTPL_ENABLE_PTSCOTCH=OFF \
        -DTPL_ENABLE_PARMETIS=OFF \
        -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
        -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
        -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

fi

make install -j16
make examples -j16
# make tests -j4
# make test
