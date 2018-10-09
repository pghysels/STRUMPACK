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
    #ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a;-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group;-lgomp;-lpthread;-lm;-ldl"
    #ScaLAPACKLIBS=""  # use this when using libsci, (module cray-libsci loaded)

    COMBBLASHOME=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/build/
    cmake ../ \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DCMAKE_CXX_COMPILER=CC \
          -DCMAKE_C_COMPILER=cc \
          -DCMAKE_Fortran_COMPILER=ftn \
          -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
          -DSTRUMPACK_C_INTERFACE=ON \
          -DTPL_BLAS_LIBRARIES="" \
          -DTPL_LAPACK_LIBRARIES="" \
          -DTPL_SCALAPACK_LIBRARIES="$ScaLAPACKLIBS" \
          -DTPL_METIS_INCLUDE_DIRS=$HOME/local/cori/parmetis-4.0.3/metis/include \
          -DTPL_METIS_LIBRARIES=$HOME/local/cori/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DTPL_ENABLE_COMBBLAS=ON \
          -DCOMBBLAS_INCLUDE_DIRS=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/ \
          -DCOMBBLAS_LIBRARIES="$COMBBLASHOME/libCommGridlib.a;$COMBBLASHOME/libHashlib.a;$COMBBLASHOME/libMemoryPoollib.a;$COMBBLASHOME/libmmiolib.a;$COMBBLASHOME/libMPIOplib.a;$COMBBLASHOME/libMPITypelib.a"
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
    cmake ../ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DSTRUMPACK_USE_MPI=OFF \
          -DSTRUMPACK_USE_OPENMP=ON \
          -DSTRUMPACK_C_INTERFACE=ON \
          -DTPL_ENABLE_PARMETIS=ON \
          -DTPL_ENABLE_SCOTCH=ON \
          -DTPL_METIS_INCLUDE_DIRS=$HOME/local/parmetis-4.0.3/metis/include \
          -DTPL_METIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a \
          -DTPL_PARMETIS_INCLUDE_DIRS=$HOME/local/parmetis-4.0.3/include \
          -DTPL_PARMETIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a \
          -DTPL_SCOTCH_INCLUDE_DIRS=$HOME/local/scotch_6.0.4/include \
          -DTPL_SCOTCH_LIBRARIES="$HOME/local/scotch_6.0.4/lib/libscotch.a;$HOME/local/scotch_6.0.4/lib/libscotcherr.a;$HOME/local/scotch_6.0.4/lib/libptscotch.a;$HOME/local/scotch_6.0.4/lib/libptscotcherr.a"
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
