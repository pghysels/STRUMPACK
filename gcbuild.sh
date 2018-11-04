
if [[ $1 == "r" ]];
then
rm -rf build
mkdir build
fi
cd build

#export CRAYPE_LINK_TYPE="dynamic"
# cmake .. \
# -DCMAKE_BUILD_TYPE=Debug \
# -DCMAKE_INSTALL_PREFIX=. \
# -DCMAKE_CXX_COMPILER=mpic++ \
# -DCMAKE_C_COMPILER=mpicc \
# -DCMAKE_Fortran_COMPILER=gfortran \
# -DCMAKE_CXX_FLAGS="" \
# -DCMAKE_EXE_LINKER_FLAGS="" \
# -DMETIS_INCLUDES="/usr/local/Cellar/metis/5.1.0/include" \
# -DMETIS_LIBRARIES="/usr/local/Cellar/metis/5.1.0/lib/libmetis.dylib" \
# -DBLAS_LIBRARIES="/usr/local/Cellar/openblas/0.2.20_1/lib/libblas.dylib" \
# -DLAPACK_LIBRARIES="/usr/local/Cellar/openblas/0.2.20_1/lib/liblapack.dylib" \
# -DSCALAPACK_LIBRARIES="/Users/gichavez/Documents/local/scalapack-2.0.2/libscalapack.a" \

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # intel compilers
# Using intel compilers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # intel compilers

# source /global/homes/g/gichavez/cori/configEnv_intel17_0_1_132.sh
export CRAYPE_LINK_TYPE="dynamic"
ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
-DCMAKE_CXX_FLAGS="-std=c++11" \
-DSCALAPACK_LIBRARIES="" \
-DSTRUMPACK_COUNT_FLOPS=ON \
-DSTRUMPACK_USE_OPENMP=ON \
-DSTRUMPACK_DEV_TESTING=OFF \
-DSTRUMPACK_BUILD_TESTS=OFF \
-DSTRUMPACK_C_INTERFACE=OFF \
-DSTRUMPACK_TASK_TIMERS=OFF \
-DSTRUMPACK_USE_COMBBLAS=OFF \
-DSTRUMPACK_USE_SCOTCH=OFF \
-DSTRUMPACK_USE_PARMETIS=OFF \
-DMETIS_INCLUDES=/global/cscratch1/sd/gichavez/edison/intel17/parmetis-4.0.3/metis/include \
-DMETIS_LIBRARIES=/global/cscratch1/sd/gichavez/edison/intel17/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a
make install VERBOSE=1
cd examples
make

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # Using GNU compilers. No need to specify DSCALAPACK_LIBRARIES# # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# source /global/homes/g/gichavez/cori/configEnv_prgenv_gnu.sh
# export CRAYPE_LINK_TYPE="dynamic"
# ScaLAPACKLIBS="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl"
# cmake .. \
# -DCMAKE_BUILD_TYPE=Release \
# -DCMAKE_INSTALL_PREFIX=. \
# -DCMAKE_CXX_COMPILER=CC \
# -DCMAKE_C_COMPILER=cc \
# -DCMAKE_Fortran_COMPILER=ftn \
# -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \
# -DCMAKE_CXX_FLAGS="-std=c++11" \
# -DSCALAPACK_LIBRARIES="" \
# -DSTRUMPACK_COUNT_FLOPS=ON \
# -DSTRUMPACK_USE_OPENMP=ON \
# -DSTRUMPACK_DEV_TESTING=OFF \
# -DSTRUMPACK_BUILD_TESTS=OFF \
# -DSTRUMPACK_C_INTERFACE=OFF \
# -DSTRUMPACK_TASK_TIMERS=OFF \
# -DSTRUMPACK_USE_COMBBLAS=OFF \
# -DSTRUMPACK_USE_SCOTCH=OFF \
# -DSTRUMPACK_USE_PARMETIS=OFF \
# -DMETIS_INCLUDES=/global/cscratch1/sd/gichavez/edison/intel17/parmetis-4.0.3/metis/include \
# -DMETIS_LIBRARIES=/global/cscratch1/sd/gichavez/edison/intel17/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a
# make install VERBOSE=1
# cd examples
# make
