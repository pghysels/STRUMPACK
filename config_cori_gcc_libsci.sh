#!/bin/bash
export CRAYPE_LINK_TYPE=dynamic


rm -rf build_libsci
rm -rf install
mkdir build_libsci
mkdir install

module swap PrgEnv-intel PrgEnv-gnu
module unload cmake
module load cmake
module rm darshan
cd build_libsci



found_host=false

if [[ $NERSC_HOST = "cori" ]]; then
    found_host=true
            #ALL_MKL_LIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-lgomp;-lpthread;-lm;-ldl"

            export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
            export METIS_DIR=/global/homes/l/liuyangz/Cori/my_software/metis-5.1.0/install_gcc
#            export PARMETIS_DIR=/global/homes/l/liuyangz/Cori/my_software/parmetis-4.0.3/build/
            # optional dependencies
            # export ParMETIS_DIR=$HOME/local/cori/gcc/parmetis-4.0.3/install
            # export SCOTCH_DIR=$HOME/local/cori/gcc/scotch_6.0.9
            export ButterflyPACK_DIR=/project/projectdirs/m2957/liuyangz/my_research/ButterflyPACK_gcc_libsci/build/lib/cmake/ButterflyPACK
            export ZFP_DIR=/project/projectdirs/m2957/liuyangz/my_research/zfp-0.5.5_gcc/build/lib/cmake/zfp

    ## Use cray-libsci (module cray-libsci loaded) instead of MKL.
    ## The problem with MKL is that it uses openmpi of intel MPI.

    cmake ../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-Ofast" \
        -DCMAKE_C_FLAGS="-Ofast" \
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
	-DTPL_BLAS_LIBRARIES="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.a" \
	-DTPL_LAPACK_LIBRARIES="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.a" \
	-DTPL_SCALAPACK_LIBRARIES="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.a"

        
fi

make install -j16
make examples -j16
# make tests -j4
# make test
