#!/bin/bash

rm -rf build
rm -rf install
mkdir build
mkdir install

cd build

found_host=false

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true

    # for MKL, use the link advisor: https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
    # Intel compiler
    #  separate arguments with ;, the -Wl,--start-group ... -Wl,--end-group is a single argument
    ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-liomp5;-lpthread;-lm;-ldl"

    # GNU compiler
    # ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-lgomp;-lpthread;-lm;-ldl"
    #ScaLAPACKLIBS=""  # use this when using libsci, (module cray-libsci loaded)

    COMBBLASHOME=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/
    COMBBLASBUILD=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/build/
    PARMETISHOME=$HOME/local/cori/parmetis-4.0.3/
    SCOTCHHOME=$HOME/local/cori/scotch_6.0.4/
    SLATEHOME=$HOME/local/cori/slate/

    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DCMAKE_CXX_COMPILER=CC \
          -DCMAKE_C_COMPILER=cc \
          -DCMAKE_Fortran_COMPILER=ftn \
          -DSTRUMPACK_USE_OPENMP=ON \
          -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
          -DTPL_BLAS_LIBRARIES="" \
          -DTPL_LAPACK_LIBRARIES="" \
          -DTPL_SCALAPACK_LIBRARIES="$ScaLAPACKLIBS" \
          -DSTRUMPACK_DEV_TESTING=OFF \
          -DSTRUMPACK_BUILD_TESTS=ON \
          -DSTRUMPACK_C_INTERFACE=OFF \
          -DSTRUMPACK_COUNT_FLOPS=ON \
          -DSTRUMPACK_TASK_TIMERS=OFF \
          -DTPL_METIS_INCLUDE_DIRS=$PARMETISHOME/metis/include \
          -DTPL_METIS_LIBRARIES=$PARMETISHOME/build/Linux-x86_64/libmetis/libmetis.a \
          -DTPL_ENABLE_COMBBLAS=OFF \
          -DTPL_COMBBLAS_INCLUDE_DIRS=$COMBBLASHOME \
          -DTPL_COMBBLAS_LIBRARIES="$COMBBLASBUILD/libCommGridlib.a;$COMBBLASBUILD/libHashlib.a;$COMBBLASBUILD/libMemoryPoollib.a;$COMBBLASBUILD/libmmiolib.a;$COMBBLASBUILD/libMPIOplib.a;$COMBBLASBUILD/libMPITypelib.a" \
          -DTPL_ENABLE_PARMETIS=ON \
          -DTPL_PARMETIS_INCLUDE_DIRS=$PARMETISHOME/include \
          -DTPL_PARMETIS_LIBRARIES=$PARMETISHOME/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DTPL_ENABLE_SCOTCH=ON \
          -DTPL_SCOTCH_INCLUDE_DIRS=$SCOTCHHOME/include \
          -DTPL_SCOTCH_LIBRARIES="$SCOTCHHOME/lib/libscotch.a;$SCOTCHHOME/lib/libscotcherr.a;$SCOTCHHOME/lib/libptscotch.a;$SCOTCHHOME/lib/libptscotcherr.a"
fi


if [[ $(dnsdomainname) = "summit.olcf.ornl.gov" ]]; then
    found_host=true

    module swap xl gcc/9.1.0
    module load mercurial
    module load cmake
    module load essl
    module load cuda
    module load netlib-lapack
    module load netlib-scalapack

    PARMETISHOME=$HOME/local/parmetis-4.0.3/
    BPACKHOME=$HOME/ButterflyPACK/

    cmake ../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_CXX_COMPILER=mpiCC \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_Fortran_COMPILER=mpif90 \
        -DSTRUMPACK_USE_OPENMP=ON \
        -DTPL_ENABLE_CUBLAS=OFF \
        -DTPL_CUBLAS_LIBRARIES="${CUDA_DIR}/lib64/libcudart.so;${CUDA_DIR}/lib64/libcublas.so;${CUDA_DIR}/lib64/libcusolver.so" \
        -DTPL_CUBLAS_INCLUDE_DIRS="${CUDA_DIR}/include" \
        -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so" \
        -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so" \
        -DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
        -DSTRUMPACK_DEV_TESTING=OFF \
        -DSTRUMPACK_BUILD_TESTS=OFF \
        -DSTRUMPACK_C_INTERFACE=OFF \
        -DSTRUMPACK_COUNT_FLOPS=ON \
        -DSTRUMPACK_TASK_TIMERS=OFF \
        -DTPL_METIS_INCLUDE_DIRS=${PARMETISHOME}/metis/include \
        -DTPL_METIS_LIBRARIES=${PARMETISHOME}/build/Linux-ppc64le/libmetis/libmetis.a \
        -DTPL_ENABLE_PARMETIS=OFF \
        -DTPL_ENABLE_SCOTCH=OFF \
        -DTPL_ENABLE_BPACK=ON \
        -DTPL_BPACK_INCLUDE_DIRS="$BPACKHOME/SRC_DOUBLE/;$BPACKHOME/SRC_DOUBLECOMPLEX" \
        -DTPL_BPACK_LIBRARIES="$BPACKHOME/build/SRC_DOUBLE/libdbutterflypack.a;$BPACKHOME/build/SRC_DOUBLECOMPLEX/libzbutterflypack.a"
fi


if [[ $(hostname -s) = "TR4-workstation" ]]; then
    found_host=true
    PARMETIS_ROOT=/home/administrator/Desktop/software/parmetis-4.0.3-intel
    PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build_static/Linux-x86_64
    BPACKHOME=/home/administrator/Desktop/research/ButterflyPACK_noarpack/
    cmake ../ \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DSTRUMPACK_USE_MPI=ON \
          -DSTRUMPACK_BUILD_TESTS=OFF \
          -DSTRUMPACK_USE_OPENMP=ON \
          -DTPL_BLAS_LIBRARIES="" \
          -DTPL_LAPACK_LIBRARIES="/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64/libmkl_intel_thread.a;/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64/libmkl_core.a;/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64/libiomp5.a" \
          -DTPL_SCALAPACK_LIBRARIES="/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a;/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64/libmkl_scalapack_lp64.a" \
          -DCMAKE_Fortran_COMPILER=mpiifort \
          -DCMAKE_EXE_LINKER_FLAGS="-qopenmp -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_core" \
          -DCMAKE_CXX_COMPILER=mpiicpc \
          -DCMAKE_C_COMPILER=mpiicc \
          -DMPI_CXX_COMPILER=/opt/intel/compilers_and_libraries_2019.1.144/linux/mpi/intel64/bin/mpiicpc \
          -DMPI_C_COMPILER=/opt/intel/compilers_and_libraries_2019.1.144/linux/mpi/intel64/bin/mpiicc \
          -DMPI_Fortran_COMPILER=/opt/intel/compilers_and_libraries_2019.1.144/linux/mpi/intel64/bin/mpiifort \
          -DSTRUMPACK_C_INTERFACE=OFF \
          -DTPL_ENABLE_PARMETIS=ON \
          -DTPL_METIS_INCLUDE_DIRS=${PARMETIS_ROOT}/metis/include \
          -DTPL_METIS_LIBRARIES=${PARMETIS_BUILD_DIR}/libmetis/libmetis.a \
          -DTPL_PARMETIS_INCLUDE_DIRS=${PARMETIS_ROOT}/include \
          -DTPL_PARMETIS_LIBRARIES=${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a \
          -DTPL_ENABLE_SCOTCH=OFF \
          -DTPL_ENABLE_BPACK=ON \
          -DTPL_BPACK_INCLUDE_DIRS="$BPACKHOME/SRC_DOUBLE/;$BPACKHOME/SRC_DOUBLECOMPLEX" \
          -DTPL_BPACK_LIBRARIES="$BPACKHOME/build/SRC_DOUBLE/libdbutterflypack.a;$BPACKHOME/build/SRC_DOUBLECOMPLEX/libzbutterflypack.a;/home/administrator/Desktop/software/arpack-ng-intel/build/lib/libparpack.a;/home/administrator/Desktop/software/arpack-ng-intel/build/lib/libarpack.a"
fi


if [[ $(hostname -s) = "xps13" ]]; then
    found_host=true
    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DTPL_METIS_INCLUDE_DIRS=/home/pieterg/local/parmetis-4.0.3/metis/include \
          -DTPL_METIS_LIBRARIES=/home/pieterg/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a
fi


if [[ $(hostname -s) = "pieterg-X8DA3" ]]; then
    found_host=true
    BPACKHOME=/home/pieterg/LBL/STRUMPACK/ButterflyPACK/
    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DBUILD_SHARED_LIBS=ON \
          -DSTRUMPACK_USE_MPI=ON \
          -DSTRUMPACK_BUILD_TESTS=ON \
          -DSTRUMPACK_USE_OPENMP=ON \
          -DSTRUMPACK_C_INTERFACE=ON \
          -DSTRUMPACK_TASK_TIMERS=ON \
          -DTPL_ENABLE_PARMETIS=ON \
          -DTPL_METIS_INCLUDE_DIRS=$HOME/local/parmetis-4.0.3/metis/include \
          -DTPL_METIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DTPL_PARMETIS_INCLUDE_DIRS=$HOME/local/parmetis-4.0.3/include \
          -DTPL_PARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DTPL_ENABLE_SCOTCH=ON \
          -DTPL_SCOTCH_INCLUDE_DIRS=$HOME/local/scotch_6.0.4/include \
          -DTPL_SCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4/lib/libscotch.a;$HOME/local/scotch_6.0.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4/lib/libptscotcherr.a" \
          -DTPL_ENABLE_BPACK=ON \
          -DTPL_BPACK_INCLUDE_DIRS="$BPACKHOME/SRC_DOUBLE/;$BPACKHOME/SRC_DOUBLECOMPLEX" \
          -DTPL_BPACK_LIBRARIES="-L$BPACKHOME/build/SRC_DOUBLE/ -ldbutterflypack -L$BPACKHOME/build/SRC_DOUBLECOMPLEX/ -lzbutterflypack"

#    -DTPL_BPACK_LIBRARIES="$BPACKHOME/build/SRC_DOUBLE/libdbutterflypack.a;$BPACKHOME/build/SRC_DOUBLECOMPLEX/libzbutterflypack.a"
fi


if ! $found_host; then
    echo "This machine was not recognized."
    echo "Open this file for examples on how to configure STRUMPACK."
    exit 1
fi

make install VERBOSE=1
make test
cd examples
make -k
cd ../../
