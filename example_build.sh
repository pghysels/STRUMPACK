#!/bin/bash

rm -rf build
rm -rf install
mkdir build
mkdir install

cd build

found_host=false


if [[ $(dnsdomainname) = "summit.olcf.ornl.gov" ]]; then
    found_host=true

    # module unload cmake
    # module swap xl gcc/9.1.0
    # module load essl
    # module load cuda/11.0.2
    # module load netlib-lapack
    # module load netlib-scalapack
    # module load cmake

    # METIS is required
    export METIS_DIR=$HOME/local/metis-5.1.0/install

    # the following are optional
    export SCOTCH_DIR=$HOME/local/scotch_6.0.9
    export ParMETIS_DIR=$HOME/local/parmetis-4.0.3/install
    export ButterflyPACK_DIR=$HOME/ButterflyPACK/install

    SLATEHOME=$HOME/slate/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DCMAKE_CXX_COMPILER=mpiCC \
          -DCMAKE_C_COMPILER=mpicc \
          -DCMAKE_Fortran_COMPILER=mpif90 \
          -DCMAKE_CUDA_COMPILER=/sw/summit/cuda/11.0.2/bin/nvcc \
          -DSTRUMPACK_USE_CUDA=ON \
          -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so" \
          -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so" \
          -DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
          -DSTRUMPACK_COUNT_FLOPS=ON \
          -DTPL_ENABLE_BPACK=OFF \
          -DTPL_ENABLE_ZFP=OFF \
          -DTPL_ENABLE_SLATE=ON \
          -DTPL_SLATE_INCLUDE_DIRS="$SLATEHOME/include/;$SLATEHOME/blaspp/include;$SLATEHOME/lapackpp/include" \
          -DTPL_SLATE_LIBRARIES="$SLATEHOME/lib/libslate.so;$SLATEHOME/blaspp/lib/libblaspp.so;$SLATEHOME/lapackpp/lib/liblapackpp.so"
fi

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true

    # ideally cmake would handle all this
    if CC --version | grep -q ICC; then
        echo "Detected Intel compiler"

        ## for MKL, use the link advisor: https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
        ## separate arguments with ;, the -Wl,--start-group ... -Wl,--end-group is a single argument
        ## Intel compiler
        #ALL_MKL_LIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-liomp5;-lpthread;-lm;-ldl"

        export METIS_DIR=$HOME/local/cori/icc/metis-5.1.0/install
    else
        if CC --version | grep -q GCC; then
            echo "Detected GCC compiler"

            #ALL_MKL_LIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-lgomp;-lpthread;-lm;-ldl"

            export METIS_DIR=$HOME/local/cori/gcc/metis-5.1.0/install

            # optional dependencies
            export ParMETIS_DIR=$HOME/local/cori/gcc/parmetis-4.0.3/install
            export SCOTCH_DIR=$HOME/local/cori/gcc/scotch_6.0.9
            export ButterflyPACK_DIR=$HOME/cori/ButterflyPACK/install
            export ZFP_DIR=$HOME/local/cori/gcc/zfp/install/
        fi
    fi

    ## Use cray-libsci (module cray-libsci loaded) instead of MKL.
    ## The problem with MKL is that it uses openmpi of intel MPI.
    ScaLAPACKLIBS=""

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
    export COMBBLAS_DIR=$HOME/local/combinatorial-blas-2.0/CombBLAS/install/
    export COMBBLASAPP_DIR=$HOME/local/combinatorial-blas-2.0/CombBLAS/Applications/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DTPL_ENABLE_BPACK=ON \
          -DTPL_ENABLE_COMBBLAS=OFF \
          -DSTRUMPACK_TASK_TIMERS=OFF \
          -DSTRUMPACK_COUNT_FLOPS=ON \
          -DTPL_SCALAPACK_LIBRARIES="$HOME/local/scalapack-2.1.0/install/lib/libscalapack.a"
fi


if [[ $(hostname -s) = "pieter-HP-EliteDesk-800-G1-SFF" ]]; then
    found_host=true

    export HIP_PLATFORM=nvcc
    export HIP_PATH=/opt/rocm-3.5.0/hip
    export hipblas_DIR=/opt/rocm-3.5.0/hipblas/lib/cmake/hipblas/

    # -DCMAKE_CXX_COMPILER=hipcc \

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DSTRUMPACK_USE_CUDA=ON \
          -DSTRUMPACK_USE_HIP=ON \
          -DSTRUMPACK_COUNT_FLOPS=ON \
          -Dscotch_INCLUDE_DIR=/usr/include/scotch/ \
          -Dscotch_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/ \
          -DCMAKE_INSTALL_PREFIX=../install
fi


if [[ $(hostname -s) = "tulip" ]]; then
    found_host=true

    echo "Detected we are running on Tulip"
    # module load cmake
    # module load blas
    # module load lapack
    # module load scalapack/openmpi/gcc/64/2.0.2
    # module load openmpi/gcc/64/1.10.7
    # module load rocm

    export METIS_DIR=/home/users/coe0239/local/metis-5.1.0/install/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DSTRUMPACK_USE_MPI=OFF \
          -DSTRUMPACK_USE_CUDA=OFF \
          -DSTRUMPACK_USE_HIP=ON \
          -DHIP_HIPCC_FLAGS=--amdgpu-target=gfx906 \
          -DSTRUMPACK_COUNT_FLOPS=ON
fi


if [[ $(hostname -s) = "cs-it-7098760" ]]; then
    found_host=true

    export SCOTCH_DIR=$HOME/local/scotch_6.1.0
    export ButterflyPACK_DIR=$HOME/LBL/ButterflyPACK/install/
    export ZFP_DIR=$HOME/local/zfp-1.0.0/install/
    export MAGMA_DIR=$HOME/local/magma-2.7.0/install/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_CXX_COMPILER=g++-11 \
          -DCMAKE_C_COMPILER=gcc-11 \
          -DCMAKE_Fortran_COMPILER=gfortran-11 \
          -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc \
          -DSTRUMPACK_USE_MPI=ON \
          -DSTRUMPACK_USE_OPENMP=ON \
          -DBLA_VENDOR=OpenBLAS \
          -DBUILD_SHARED_LIBS=OFF \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DSTRUMPACK_COUNT_FLOPS=ON \
          -DSTRUMPACK_USE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES="75" \
          -DSTRUMPACK_USE_HIP=OFF \
          -DTPL_ENABLE_MAGMA=OFF \
          -DTPL_ENABLE_SLATE=OFF \
          -DTPL_ENABLE_COMBBLAS=OFF \
          -DTPL_SCALAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so"

fi


if ! $found_host; then
    echo "This machine was not recognized."
    echo "Open this file and modify the CMake command."
    echo "Running CMake ..."

    # METIS is required, but might be already be installed by the system
    #export METIS_DIR=

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DTPL_SCALAPACK_LIBRARIES="/usr/lib64/openmpi/lib/libscalapack.so"

    ## if not found automatically, you can specify BLAS/LAPACK/SCALAPACK as:
    #  -DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libopenblas.a"
    #  -DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.a"
    #  -DTPL_SCALAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so"
fi

make -j8 # VERBOSE=1
make install -j8
make examples -j8
# make test
