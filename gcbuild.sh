
if [[ $1 == "r" ]];
then
rm -rf build
mkdir build
fi

cd build
export CRAYPE_LINK_TYPE="dynamic"

cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_CXX_FLAGS="" \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DMETIS_INCLUDES="/usr/local/Cellar/metis/5.1.0/include" \
-DMETIS_LIBRARIES="/usr/local/Cellar/metis/5.1.0/lib/libmetis.dylib" \
-DBLAS_LIBRARIES="/usr/local/Cellar/openblas/0.2.20_1/lib/libblas.dylib" \
-DLAPACK_LIBRARIES="/usr/local/Cellar/openblas/0.2.20_1/lib/liblapack.dylib" \
-DSCALAPACK_LIBRARIES="/Users/gichavez/Documents/local/scalapack-2.0.2/libscalapack.a" \

make install VERBOSE=1
cd examples
make