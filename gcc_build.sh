#!/bin/bash

if [[ $2 == "r" ]];
then
rm -rf build
mkdir build
fi

cd build

if [[ $1 == "cori" ]];
then
echo $1
source /global/cscratch1/sd/gichavez/intel17/configEnv.sh
export CRAYPE_LINK_TYPE="dynamic"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_CXX_FLAGS="" \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DMETIS_INCLUDES=/global/homes/g/gichavez/cori/gnu/parmetis-4.0.3/metis/include \
-DMETIS_LIBRARIES=/global/homes/g/gichavez/cori/gnu/parmetis-4.0.3/install/lib/libparmetis.a \
-DBLAS_LIBRARIES="" \
-DLAPACK_LIBRARIES="" \
-DSCALAPACK_LIBRARIES=""
elif [[ $1 == "edison" ]];
then
echo $1
source /global/cscratch1/sd/gichavez/intel17/configEnv.sh
export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="/global/cscratch1/sd/gichavez/intel17/parmetis-4.0.3"
export SCOTCH_INSTALL="/global/cscratch1/sd/gichavez/intel17/scotch_6.0.4/build"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DCMAKE_CXX_FLAGS="" \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DPARMETIS_INCLUDES=$PARMETIS_INSTALL/install/include \
-DPARMETIS_LIBRARIES=$PARMETIS_INSTALL/install/lib/libparmetis.a \
-DSCOTCH_INCLUDES=$SCOTCH_INSTALL/include \
-DSCOTCH_LIBRARIES="$SCOTCH_INSTALL/lib/libscotch.a;$SCOTCH_INSTALL/lib/libscotcherr.a;$SCOTCH_INSTALL/lib/libptscotch.a;$SCOTCH_INSTALL/lib/libptscotcherr.a"
elif [[ $1 == "pieter" ]];
then
echo $1
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DSTRUMPACK_DEV_TESTING=OFF \
-DSTRUMPACK_C_INTERFACE=OFF \
-DSTRUMPACK_COUNT_FLOPS=ON \
-DSTRUMPACK_TASK_TIMERS=ON \
-DSTRUMPACK_USE_PARMETIS=OFF \
-DSTRUMPACK_USE_SCOTCH=OFF \
-DCMAKE_CXX_FLAGS="-Wall -Wfatal-errors -Wextra -Wno-unused-parameter" \
-DMETIS_INCLUDES=/home/pieterg/local/parmetis-4.0.3/metis/include \
-DMETIS_LIBRARIES=/home/pieterg/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a
elif [[ $1 == "imac" ]];
then
echo $1
cmake .. -DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpic++ \
-DBLAS_LIBRARIES=/usr/local/Cellar/openblas/0.2.20/lib/libblas.dylib \
-DLAPACK_LIBRARIES=/usr/local/Cellar/openblas/0.2.20/lib/liblapack.dylib \
-DSCALAPACK_LIBRARIES="/usr/local/Cellar/scalapack/2.0.2_8/lib/libscalapack.dylib" \
-DCMAKE_CXX_FLAGS="" \
-DCMAKE_Fortran_COMPILER=mpifort \
-DMETIS_INCLUDES=/usr/local/Cellar/metis/5.1.0/include \
-DMETIS_LIBRARIES=/usr/local/Cellar/metis/5.1.0/lib/libmetis.dylib \
-DPARMETIS_INCLUDES=/usr/local/Cellar/parmetis/4.0.3_4/include \
-DPARMETIS_LIBRARIES=/usr/local/Cellar/parmetis/4.0.3_4/lib/libparmetis.dylib \
-DSCOTCH_INCLUDES=/usr/local/Cellar/scotch/6.0.4_4/include \
-DSCOTCH_LIBRARIES="/usr/local/Cellar/scotch/6.0.4_4/lib/libscotch.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libscotcherr.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libptscotch.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libptscotcherr.dylib"
else
	echo "Unrecognized configuration. Try: <cori|edison|imac|pieter>"
	exit 0
fi

# -DCMAKE_BUILD_TYPE=Debug
# -DCMAKE_CXX_FLAGS="-lstdc++ -DUSE_TASK_TIMER -DCOUNT_FLOPS" \
# -DCMAKE_CXX_FLAGS="" \
# -DCMAKE_CXX_FLAGS="-L/global/common/cori/software/ipm/2.0.5/intel/lib/libipmf.a -L/global/common/cori/software/ipm/2.0.5/intel/lib/libipm.a" \
# -DHMATRIX_LIBRARIES=/global/cscratch1/sd/gichavez/intel17/h_matrix_rbf_randomization/build/SRC/libhmatrix.a \

# -DCMAKE_EXE_LINKER_FLAGS="$(IPM)" \
# -DCMAKE_EXE_LINKER_FLAGS="-dynamic" \

# -DCMAKE_EXE_LINKER_FLAGS="-L/global/cscratch1/sd/gichavez/.consulting/INC0117791/IPM/IPM_build_Intel_dynamic/lib -lipm" \

make install VERBOSE=1
cd examples
make
