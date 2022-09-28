#!/bin/bash
export CRAYPE_LINK_TYPE=dynamic


rm -rf build_netlibblas
rm -rf install
mkdir build_netlibblas
mkdir install

module swap PrgEnv-intel PrgEnv-gnu
module unload cmake
module load cmake
module rm darshan
module unload cray-libsci

FASTMATH=-Ofast


export ROOT=$PWD

export SCALAPACK_LIB=$ROOT/scalapack-2.1.0-netlibblas/build/install/lib/libscalapack.a
export BLAS_LIB=$ROOT/netlib-LAPACK-3.10.1/build/install/lib64/libblas.a
export LAPACK_LIB="$ROOT/netlib-LAPACK-3.10.1/build/install/lib64/liblapack.a;$ROOT/netlib-LAPACK-3.10.1/build/install/lib64/libblas.a"


###################################
cd $ROOT
rm -rf netlib-LAPACK-3.10.1
rm -rf v3.10.1.tar.gz*
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.1.tar.gz
tar -xvf v3.10.1.tar.gz
mv lapack-3.10.1 netlib-LAPACK-3.10.1
cd netlib-LAPACK-3.10.1
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=cc \
	-DCMAKE_Fortran_COMPILER=ftn \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_Fortran_FLAGS="${FASTMATH} -fopenmp -fallow-argument-mismatch" \
    -DCMAKE_C_FLAGS="${FASTMATH}" 
make -j32
make install


cd $ROOT
rm -rf scalapack-2.1.0.tgz*
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
rm -rf scalapack-2.1.0-netlibblas
mkdir scalapack-2.1.0-netlibblas && tar xf scalapack-2.1.0.tgz -C scalapack-2.1.0-netlibblas --strip-components 1
cd scalapack-2.1.0-netlibblas
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=cc \
	-DCMAKE_Fortran_COMPILER=ftn \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_Fortran_FLAGS="${FASTMATH} -fopenmp -fallow-argument-mismatch" \
    -DCMAKE_CXX_FLAGS="${FASTMATH}" \
    -DCMAKE_C_FLAGS="${FASTMATH}" \
	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
make -j32
make install



cd $ROOT/build_netlibblas


found_host=false

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true

            # export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
            export METIS_DIR=/global/homes/l/liuyangz/Cori/my_software/metis-5.1.0/install_gcc
#            export PARMETIS_DIR=/global/homes/l/liuyangz/Cori/my_software/parmetis-4.0.3/build/
            # optional dependencies
            # export ParMETIS_DIR=$HOME/local/cori/gcc/parmetis-4.0.3/install
            # export SCOTCH_DIR=$HOME/local/cori/gcc/scotch_6.0.9
        #     export ButterflyPACK_DIR=/project/projectdirs/m2957/liuyangz/my_research/ButterflyPACK_gcc_libsci/build/lib/cmake/ButterflyPACK
            export ZFP_DIR=/project/projectdirs/m2957/liuyangz/my_research/zfp-0.5.5_gcc/build/lib/cmake/zfp

    ## Use cray-libsci (module cray-libsci loaded) instead of MKL.
    ## The problem with MKL is that it uses openmpi of intel MPI.

    cmake --debug-trycompile ../ \
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
        -DSTRUMPACK_USE_OPENMP=ON \
        -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
        -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
        -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

fi

make install -j16
make examples -j16
# make tests -j4
# make test
