cmake_minimum_required(VERSION 3.2)
project(STRUMPACK CXX C Fortran)
#project(STRUMPACK NONE)

## Metis is required for now
# option(STRUMPACK_USE_METIS "Build with support for Metis" ON)

option(STRUMPACK_USE_MPI "Build with MPI support" ON)
option(STRUMPACK_USE_OPENMP "Use OpenMP for on-node threading tasking" ON)

## we don't have any CUDA kernels yet, see also TPL_ENABLE_CUBLAS
# option(STRUMPACK_USE_CUDA "Use CUDA (language) for GPU support" OFF)

option(TPL_ENABLE_SLATE "Use SLATE, the ECP ScaLAPACK replacement" OFF)
option(TPL_ENABLE_CUBLAS "Build with support for cuBLAS/cuSOLVER" OFF)
option(TPL_ENABLE_PARMETIS "Build with support for ParMetis" OFF)
option(TPL_ENABLE_SCOTCH "Build with support for Scotch/PTScotch" OFF)
option(TPL_ENABLE_PAPI "Build with support for PAPI monitoring" OFF)
option(TPL_ENABLE_COMBBLAS "Use CombBLAS for weighted matching" OFF)
option(TPL_ENABLE_BPACK "Use BPACK (ButterflyPACK) code by Yang Liu" OFF)
option(TPL_ENABLE_ZFP "Build with support for ZFP compression" OFF)

option(STRUMPACK_COUNT_FLOPS "Build with flop counters" ON)
option(STRUMPACK_TASK_TIMERS "Build with timers for internal routines" OFF)
option(STRUMPACK_BUILD_TESTS "Build the tests" ON)
option(STRUMPACK_DEV_TESTING "Enable extensive testing" OFF)
option(STRUMPACK_C_INTERFACE "Build the C interface" ON)

######################################################################
# IDEAS: xSDK standards module
# set(USE_XSDK_DEFAULTS FALSE)
# set(XSDK_ENABLE_CXX ON)
# set(XSDK_ENABLE_C ON)
# set(XSDK_ENABLE_Fortran ON)
# include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/XSDKDefaults.cmake")
# enable_language(CXX C Fortran)
######################################################################


if(NOT STRUMPACK_USE_MPI)
  if(TPL_ENABLE_PARMETIS)
    message(STATUS "Disabling ParMetis. ParMetis requires MPI support,"
      " but compiling without MPI support! ")
    set(TPL_ENABLE_PARMETIS OFF)
  endif()
  if(TPL_ENABLE_BPACK)
    message(STATUS "Disabling BPACK (ButterflyPACK). BPACK requires MPI support,"
      " but compiling without MPI support! ")
    set(TPL_ENABLE_BPACK OFF)
  endif()
endif()

set(STRUMPACK_VERSION_MAJOR 3)
set(STRUMPACK_VERSION_MINOR 3)
set(STRUMPACK_VERSION_PATCH 0)
set(STRUMPACK_VERSION_FULL
  "${STRUMPACK_VERSION_MAJOR}.${STRUMPACK_VERSION_MINOR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(TPL_ENABLE_COMBBLAS)
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED on)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX11)
  check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

if(STRUMPACK_USE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_C_FOUND)
    message(STATUS "Found C mpi compiler:       " "${MPI_C_COMPILER}")
    include_directories(${MPI_C_INCLUDE_PATH})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}" )
    set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
  endif()
  if(MPI_CXX_FOUND)
    message(STATUS "Found C++ mpi compiler:     " "${MPI_CXX_COMPILER}")
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}" )
    set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
  endif()
  if(MPI_Fortran_FOUND)
    message(STATUS "Found Fortran mpi compiler: " "${MPI_Fortran_COMPILER}")
    include_directories(${MPI_Fortran_INCLUDE_PATH})
    set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} ${MPI_Fortran_COMPILE_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${MPI_Fortran_LINK_FLAGS}" )
    set(LIBS ${LIBS} ${MPI_Fortran_LIBRARIES})
  endif()
endif()

include(FortranCInterface)
FortranCInterface_HEADER(${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  MACRO_NAMESPACE "STRUMPACK_FC_")
#include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src)
FortranCInterface_VERIFY(CXX)

if(STRUMPACK_USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP specification date: " "${OpenMP_CXX_SPEC_DATE}")
    if(DEFINED OpenMP_CXX_SPEC_DATE)
      if(OpenMP_CXX_SPEC_DATE VERSION_GREATER_EQUAL 201511)
        message(STATUS "Compiler claims to support OpenMP 4.5")
      elseif(OpenMP_CXX_SPEC_DATE VERSION_GREATER_EQUAL 201307)
        message(STATUS "Compiler claims to support OpenMP 4.0")
      endif()
    endif()
    message(STATUS "Checking OpenMP task dependencies/priorities")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
    check_cxx_source_compiles(
      "#include <iostream>
int main() {
int x=0, y=5;
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(inout:x) depend(out:y) priority(5)
{ y = x; x += 3; }
#pragma omp task depend(in:x) depend(out:y) priority(1)
{ y = 3 * x; }
}
return 0;
}" STRUMPACK_USE_OPENMP_TASK_DEPEND)
    if(STRUMPACK_USE_OPENMP_TASK_DEPEND)
      message(STATUS "Enabling OpenMP task dependencies/priorities.")
    else()
      message(STATUS
        "Failed to compile code with OpenMP task dependencies/priorities."
        " STRUMPACK will not use OpenMP tasks with dependencies/priorities.")
    endif()
    message(STATUS "Checking OpenMP taskloop")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
    check_cxx_source_compiles(
      "#include <iostream>
int main() {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
for (int i=0; i<100; i++) std::cout << \"hello\" << std::endl;
return 0;
}" STRUMPACK_USE_OPENMP_TASKLOOP)
    if(STRUMPACK_USE_OPENMP_TASKLOOP)
      message(STATUS "Enabling OpenMP taskloop construct")
    else()
      message(STATUS
        "Failed to compile code with OpenMP taskloop."
        " STRUMPACK will not use OpenMP taskloops.")
    endif()
  else()
    message(WARNING "WARNING: Could not detect OpenMP compiler
    support, will proceed without OpenMP!")
  endif()
else()
  message(STATUS "Building STRUMPACK without OpenMP support.")
  check_cxx_compiler_flag("-Wno-unknown-pragmas"
    COMPILER_SUPPORTS_NOPRAGMA_WARNING)
  if(COMPILER_SUPPORTS_NOPRAGMA_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  endif()
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  message(STATUS "Found PTHREADS library")
  set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
else()
  message(WARNING
    "WARNING: Could not find a PTHREADS library,"
    " might be needed by Scotch !!!!")
endif()

include(CheckLibraryExists)
check_cxx_source_compiles(
  "#include <cstdint>
#include <atomic>
std::atomic<uintptr_t> x;
std::atomic<uintmax_t> y;
int main() {
  bool lf = x.is_lock_free();
  return x + y;
}
" STRUMPACK_ATOMICS_WITHOUT_LIB)
if(NOT STRUMPACK_ATOMICS_WITHOUT_LIB)
  find_library(STRUMPACK_LIB_ATOMIC
    NAMES atomic atomic atomic.so.1 libatomic.so.1)
  if(STRUMPACK_LIB_ATOMIC)
    message(STATUS "Found libatomic at ${STRUMPACK_LIB_ATOMIC}")
    set(LIBS ${LIBS} ${STRUMPACK_LIB_ATOMIC})
  else()
    message(WARNING "Failed to find required -latomic, trying to link with -latomic anyway")
    set(LIBS ${LIBS} "-latomic")
  endif()
endif()

if(DEFINED TPL_BLAS_LIBRARIES)
  set(LIBS ${LIBS} ${TPL_BLAS_LIBRARIES})
else()
  find_package(BLAS)
  if(BLAS_FOUND)
    message(STATUS "Found BLAS library:   " "${BLAS_LIBRARIES}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    set(LIBS ${LIBS} ${BLAS_LIBRARIES})
  else()
    message(WARNING "WARNING: Could not find a BLAS library !!!!!!!!!!!\n"
      "Note that on some machines (like Cray's)"
      " BLAS is linked in by the compiler wrapper")
  endif()
endif()

if(DEFINED TPL_LAPACK_LIBRARIES)
  set(LIBS ${LIBS} ${TPL_LAPACK_LIBRARIES})
else()
  find_package(LAPACK)
  if(LAPACK_FOUND)
    message(STATUS "Found LAPACK library: " "${LAPACK_LIBRARIES}")
    set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    set(LIBS ${LIBS} ${LAPACK_LIBRARIES})
  else()
    message(WARNING "WARNING: Could not find a LAPACK library !!!!!!!!!\n"
      "Note that on some machines (like Cray's)"
      " LAPACK is linked in by the compiler wrapper")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(STRUMPACK_USE_MPI)
  if(DEFINED TPL_SCALAPACK_LIBRARIES)
    set(LIBS ${LIBS} ${TPL_SCALAPACK_LIBRARIES})
    # include_directories(${TPL_SCALAPACK_INCLUDE_DIRS})
  else()
    find_package(SCALAPACK)
    if(SCALAPACK_FOUND)
      set(LIBS ${LIBS} ${SCALAPACK_LIBRARIES})
      # include_directories(${SCALAPACK_INCLUDES})
    else()
      message(WARNING
        "WARNING: Could not find a ScaLAPACK/BLACS library !!!!!!!!\n"
        "Note that on some machines (like Cray's)"
        " ScaLAPACK is linked in by the compiler wrapper")
    endif()
  endif()
endif()

if(TPL_ENABLE_PAPI)
  option(STRUMPACK_USE_PAPI "" ON)
  find_package(PAPI)
  set(LIBS ${LIBS} ${PAPI_LIBRARIES})
  include_directories(${PAPI_INCLUDES})
  if(NOT PAPI_FOUND)
    message(FATAL_ERROR
      "Could not find PAPI!"
      "Set the PAPIDIR environtment variable,"
      "or specify PAPI_INCLUDES and PAPI_LIBRARIES."
      "Get PAPI from http://icl.cs.utk.edu/papi/")
  endif()
else()
  message(STATUS "Building STRUMPACK without PAPI support.")
endif()

if(TPL_ENABLE_SLATE)
  message(STATUS "Enabling SLATE")
  # option(STRUMPACK_USE_SLATE_LAPACK "" ON)
  option(STRUMPACK_USE_SLATE_SCALAPACK "" ON)
  if(DEFINED TPL_SLATE_LIBRARIES)
    set(LIBS ${LIBS} ${TPL_SLATE_LIBRARIES})
    include_directories(${TPL_SLATE_INCLUDE_DIRS})
    add_compile_definitions(STRUMPACK_PBLAS_BLOCKSIZE=256)
  else()
    message(WARNING "Could not find SLATE,"
     " please define TPL_SLATE_LIBRARIES and TPL_SLATE_INCLUDE_DIRS")
  endif()
endif()

if(TPL_ENABLE_ZFP)
  message(STATUS "Enabling ZFP support")
  option(STRUMPACK_USE_ZFP "" ON)
  set(LIBS ${TPL_ZFP_LIBRARIES} ${LIBS})
  include_directories(${TPL_ZFP_INCLUDE_DIRS})
endif()

if(STRUMPACK_USE_MPI)
  if(TPL_ENABLE_COMBBLAS)
    option(STRUMPACK_USE_COMBBLAS "" ON)
    set(LIBS ${LIBS} ${TPL_COMBBLAS_LIBRARIES})
    include_directories(${TPL_COMBBLAS_INCLUDE_DIRS})
    message(STATUS "Combinatorial BLAS support enabled")
  endif()

  if(TPL_ENABLE_PARMETIS)
    option(STRUMPACK_USE_PARMETIS "" ON)
    find_package(ParMetis)
    set(LIBS ${LIBS} ${PARMETIS_LIBRARIES})
    include_directories(${PARMETIS_INCLUDES})
    if(NOT PARMETIS_FOUND)
      message(FATAL_ERROR
        "Could not find ParMETIS!"
        "Get ParMetis from http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview")
    endif()
  else()
    message(STATUS "Building STRUMPACK without ParMetis support.")
  endif()
endif()

# at the moment, Metis is required to build STRUMPACK
find_package(Metis)
set(LIBS ${LIBS} ${METIS_LIBRARIES})
include_directories(${METIS_INCLUDES})
if(METIS_FOUND)
  message(STATUS "Found Metis library\n")
else()
  message(FATAL_ERROR
    "Could not find Metis (required)!"
    "Get Metis from http://glaros.dtc.umn.edu/gkhome/metis/metis/overview\n")
endif()

if(TPL_ENABLE_SCOTCH)
  option(STRUMPACK_USE_SCOTCH "" ON)
  find_package(Scotch)
  set(LIBS ${LIBS} ${SCOTCH_LIBRARIES})
  include_directories(${SCOTCH_INCLUDES})
else()
  message(STATUS "Building STRUMPACK without Scotch/PTScotch support.")
endif()

if(TPL_ENABLE_BPACK)
  option(STRUMPACK_USE_BPACK "" ON)
  set(LIBS ${TPL_BPACK_LIBRARIES} ${LIBS})
  include_directories(${TPL_BPACK_INCLUDE_DIRS})
endif()

if(TPL_ENABLE_CUBLAS)
  option(STRUMPACK_USE_CUBLAS "" ON)
  set(LIBS ${LIBS} ${TPL_CUBLAS_LIBRARIES})
  include_directories(${TPL_CUBLAS_INCLUDE_DIRS})
endif()

message(STATUS "Linking with: " "${LIBS}")
message(STATUS "Implicit C libs: " "${CMAKE_C_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "Implicit CXX libs: " "${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "Implicit Fortran libs: "
  "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")

foreach(arg ${LIBS})
  set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} ${arg}")
endforeach(arg ${LIBS})
# foreach(arg ${CMAKE_C_IMPLICIT_LINK_LIBRARIES})
#   set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} -l${arg}")
# endforeach(arg ${CMAKE_C_IMPLICIT_LINK_LIBRARIES})
# foreach(arg ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
#   set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} -l${arg}")
# endforeach(arg ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
foreach(arg ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} -l${arg}")
endforeach(arg ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  set(STRUMPACK_ALL_INCLUDE_DIRS "${STRUMPACK_ALL_INCLUDE_DIRS} -I${dir}")
endforeach()
set(STRUMPACK_ALL_INCLUDE_DIRS "${STRUMPACK_ALL_INCLUDE_DIRS} -I${CMAKE_INSTALL_PREFIX}/include")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
  set(STRUMPACK_ALL_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
  set(STRUMPACK_ALL_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_MINSIZEREL}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
  set(STRUMPACK_ALL_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_MINSIZEREL}")
else()
  if(NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(WARNING "WARNING: CMAKE_BUILD_TYPE not recognized!")
  endif()
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(STRUMPACK_ALL_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_RELWITHDEBINFO}")
endif()


include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src)

set(LIB_STRUMPACK_SOURCES
  src/StrumpackParameters.cpp
  src/misc/TaskTimer.cpp
  src/kernel/Kernel.cpp
  src/sparse/rcm/rcm.f
  src/sparse/rcm/degree.f
  src/sparse/rcm/fnroot.f
  src/sparse/rcm/genrcm.f
  src/sparse/rcm/rootls.f
  src/sparse/strumpack_mc64ad.c
  src/sparse/iterative/BiCGStab.cpp
  src/sparse/iterative/GMRes.cpp
  src/sparse/iterative/IterativeRefinement.cpp
  src/sparse/fronts/FrontFactory.cpp
  src/sparse/fronts/FrontalMatrix.cpp
  src/sparse/fronts/FrontalMatrixDense.cpp
  src/sparse/fronts/FrontalMatrixHSS.cpp
  src/clustering/PCAPartitioning.cpp
  src/clustering/CobblePartitioning.cpp
  src/clustering/KMeans.cpp
  src/clustering/KDTree.cpp
  src/HSS/HSSMatrixBase.cpp
  src/HSS/HSSMatrix.cpp
  src/dense/BLASLAPACKWrapper.cpp
  src/dense/BLASLAPACKOpenMPTask.cpp
  src/dense/lapack/sgeqp3tol.f src/dense/lapack/dgeqp3tol.f
  src/dense/lapack/cgeqp3tol.f src/dense/lapack/zgeqp3tol.f
  src/dense/lapack/sgeqrfmod.f src/dense/lapack/dgeqrfmod.f
  src/dense/lapack/cgeqrfmod.f src/dense/lapack/zgeqrfmod.f
  src/dense/lapack/sgeqr2mod.f src/dense/lapack/dgeqr2mod.f
  src/dense/lapack/cgeqr2mod.f src/dense/lapack/zgeqr2mod.f
  src/dense/lapack/slaqpsmod.f src/dense/lapack/dlaqpsmod.f
  src/dense/lapack/claqpsmod.f src/dense/lapack/zlaqpsmod.f
  src/dense/lapack/slaqp2mod.f src/dense/lapack/dlaqp2mod.f
  src/dense/lapack/claqp2mod.f src/dense/lapack/zlaqp2mod.f
  src/dense/lapack/sgetrfmod.f src/dense/lapack/dgetrfmod.f
  src/dense/lapack/cgetrfmod.f src/dense/lapack/zgetrfmod.f
  src/dense/lapack/sgetf2mod.f src/dense/lapack/dgetf2mod.f
  src/dense/lapack/cgetf2mod.f src/dense/lapack/zgetf2mod.f
  src/dense/lapack/sorglqmod.f src/dense/lapack/dorglqmod.f
  src/dense/lapack/cunglqmod.f src/dense/lapack/zunglqmod.f
  src/dense/lapack/sorgl2mod.f src/dense/lapack/dorgl2mod.f
  src/dense/lapack/cungl2mod.f src/dense/lapack/zungl2mod.f
  src/dense/lapack/slarfbmod.f src/dense/lapack/dlarfbmod.f
  src/dense/lapack/clarfbmod.f src/dense/lapack/zlarfbmod.f
  src/dense/lapack/slarftmod.f src/dense/lapack/dlarftmod.f
  src/dense/lapack/clarftmod.f src/dense/lapack/zlarftmod.f
  src/dense/lapack/slarfmod.f  src/dense/lapack/dlarfmod.f
  src/dense/lapack/clarfmod.f  src/dense/lapack/zlarfmod.f
  src/dense/lapack/sgelqfmod.f src/dense/lapack/dgelqfmod.f
  src/dense/lapack/cgelqfmod.f src/dense/lapack/zgelqfmod.f
  src/dense/lapack/sgelq2mod.f src/dense/lapack/dgelq2mod.f
  src/dense/lapack/cgelq2mod.f src/dense/lapack/zgelq2mod.f)

if(TPL_ENABLE_CUBLAS)
  list(APPEND LIB_STRUMPACK_SOURCES
    src/sparse/fronts/FrontalMatrixCUBLAS.cpp)
endif()

if(TPL_ENABLE_ZFP)
  list(APPEND LIB_STRUMPACK_SOURCES
    src/sparse/fronts/FrontalMatrixLossy.cpp)
endif()

if(STRUMPACK_C_INTERFACE)
  list(APPEND LIB_STRUMPACK_SOURCES
    src/StrumpackSparseSolver.cpp)
endif()

if(STRUMPACK_USE_MPI)
  list(APPEND LIB_STRUMPACK_SOURCES
    src/dense/ScaLAPACKWrapper.cpp
    src/dense/lapack/pcgeqpftol.f
    src/dense/lapack/pdgeqpftol.f
    src/dense/lapack/psgeqpftol.f
    src/dense/lapack/pzgeqpftol.f
    src/sparse/MUMPS/ana_orderings.F
    src/sparse/fronts/FrontalMatrixMPI.cpp
    src/sparse/fronts/FrontalMatrixDenseMPI.cpp
    src/sparse/fronts/FrontalMatrixHSSMPI.cpp
    src/sparse/fronts/ExtendAdd.cpp
    src/sparse/iterative/GMResMPI.cpp
    src/sparse/iterative/BiCGStabMPI.cpp
    src/sparse/iterative/IterativeRefinementMPI.cpp
    src/HSS/HSSMatrixMPI.cpp)

  if(TPL_ENABLE_BPACK)
    list(APPEND LIB_STRUMPACK_SOURCES
      src/HODLR/HODLRMatrix.cpp
      src/HODLR/LRBFMatrix.cpp
      src/HODLR/HODLRWrapper.cpp
      src/sparse/fronts/FrontalMatrixHODLR.cpp
      src/sparse/fronts/FrontalMatrixHODLRMPI.cpp)
  endif()
endif()

add_library(strumpack ${LIB_STRUMPACK_SOURCES})
target_link_libraries(strumpack ${LIBS})

install(TARGETS strumpack LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

configure_file(${CMAKE_SOURCE_DIR}/src/StrumpackConfig.hpp.in
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp)

install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp
  src/StrumpackSparseSolver.h
  src/StrumpackSparseSolver.hpp
  src/StrumpackOptions.hpp
  src/StrumpackParameters.hpp
  DESTINATION include)

configure_file(${CMAKE_SOURCE_DIR}/src/python/STRUMPACKKernel.py.in
  ${PROJECT_BINARY_DIR}/STRUMPACKKernel.py)

install(FILES
  ${PROJECT_BINARY_DIR}/STRUMPACKKernel.py
  DESTINATION include/python)

install(FILES
  src/misc/TaskTimer.hpp
  src/misc/RandomWrapper.hpp
  src/misc/Tools.hpp
  DESTINATION include/misc)

install(FILES
  src/clustering/NeighborSearch.hpp
  src/clustering/Clustering.hpp
  DESTINATION include/clustering)

install(FILES
  src/kernel/Kernel.hpp
  src/kernel/KernelRegression.hpp
  src/kernel/Kernel.h
  src/kernel/Metrics.hpp
  DESTINATION include/kernel)

install(FILES
  src/dense/DistributedMatrix.hpp
  src/dense/ACA.hpp
  src/dense/BACA.hpp
  src/dense/DenseMatrix.hpp
  src/dense/BLASLAPACKOpenMPTask.hpp
  src/dense/BLASLAPACKWrapper.hpp
  DESTINATION include/dense)

install(FILES
  src/sparse/CSRGraph.hpp
  src/sparse/CSRMatrix.hpp
  src/sparse/CompressedSparseMatrix.hpp
  src/sparse/ETree.hpp
  src/sparse/EliminationTree.hpp
  src/sparse/GeometricReordering.hpp
  src/sparse/MatrixReordering.hpp
  src/sparse/MetisReordering.hpp
  src/sparse/Redistribute.hpp
  src/sparse/ScotchReordering.hpp
  src/sparse/RCMReordering.hpp
  src/sparse/SeparatorTree.hpp
  DESTINATION include/sparse)

install(FILES
  src/sparse/fronts/FrontFactory.hpp
  src/sparse/fronts/FrontalMatrix.hpp
  DESTINATION include/sparse/fronts)

install(FILES
  src/sparse/iterative/IterativeSolvers.hpp
  DESTINATION include/sparse/iterative)

if(TPL_ENABLE_CUBLAS)
  install(FILES
    src/dense/CUDAWrapper.hpp
    DESTINATION include/dense)
endif()

install(FILES
  src/HSS/HSSMatrix.hpp
  src/HSS/HSSBasisID.hpp
  src/HSS/HSSExtra.hpp
  src/HSS/HSSMatrixBase.hpp
  src/HSS/HSSPartitionTree.hpp
  src/HSS/HSSOptions.hpp
  DESTINATION include/HSS)

install(FILES
  src/BLR/BLROptions.hpp
  src/BLR/BLRTile.hpp
  src/BLR/DenseTile.hpp
  src/BLR/LRTile.hpp
  src/BLR/BLRTileBLAS.hpp
  src/BLR/BLRMatrix.hpp
  DESTINATION include/BLR)

if(TPL_ENABLE_BPACK)
  install(FILES
    src/HODLR/HODLROptions.hpp
    src/HODLR/HODLRMatrix.hpp
    src/HODLR/HODLRWrapper.hpp
    src/HODLR/LRBFMatrix.hpp
    DESTINATION include/HODLR)
endif()

if(STRUMPACK_USE_MPI)
  install(FILES
    src/StrumpackSparseSolverMPI.hpp
    src/StrumpackSparseSolverMPIDist.hpp
    DESTINATION include)

  install(FILES
    src/misc/MPIWrapper.hpp
    DESTINATION include/misc)

  install(FILES
    src/dense/DistributedMatrix.hpp
    src/dense/DistributedVector.hpp
    src/dense/ScaLAPACKWrapper.hpp
    src/dense/BLACSGrid.hpp
    DESTINATION include/dense)

  install(FILES
    src/sparse/CSRMatrixMPI.hpp
    src/sparse/EliminationTreeMPI.hpp
    src/sparse/EliminationTreeMPIDist.hpp
    src/sparse/GeometricReorderingMPI.hpp
    src/sparse/MatrixReorderingMPI.hpp
    src/sparse/PTScotchReordering.hpp
    src/sparse/ParMetisReordering.hpp
    src/sparse/PropMapSparseMatrix.hpp
    src/sparse/Redistribute.hpp
    src/sparse/mumps_symqamd.hpp
    DESTINATION include/sparse)

  install(FILES
    src/sparse/fronts/FrontalMatrixMPI.hpp
    DESTINATION include/sparse/fronts)

  install(FILES
    src/sparse/iterative/IterativeSolversMPI.hpp
    DESTINATION include/sparse/iterative)

  install(FILES
    src/HSS/HSSMatrixMPI.hpp
    src/HSS/HSSExtraMPI.hpp
    src/HSS/HSSBasisIDMPI.hpp
    src/HSS/DistSamples.hpp
    src/HSS/DistElemMult.hpp
    DESTINATION include/HSS)

  install(FILES
    src/BLR/BLRMatrixMPI.hpp
    DESTINATION include/BLR)
endif()

if(TPL_ENABLE_COMBBLAS)
  install(FILES
    src/sparse/AWPMCombBLAS.hpp
    DESTINATION include/sparse)
endif()


set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION_MAJOR ${STRUMPACK_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${STRUMPACK_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${STRUMPACK_VERSION_PATCH})
include(CPack)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/doc/doxygen/doxygen.dox.in
    ${CMAKE_BINARY_DIR}/doxygen.dox @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doxygen.dox
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

if(STRUMPACK_C_INTERFACE)
  set(C_EXAMPLES "sexample dexample cexample zexample")
endif()

# build an example makefile
if(STRUMPACK_USE_MPI)
  set(STRUMPACK_C_COMPILER "${MPI_C_COMPILER}")
  set(STRUMPACK_CXX_COMPILER "${MPI_CXX_COMPILER}")
  set(MPI_EXAMPLES "KernelRegressionMPI testPoisson2dMPIDist testPoisson3dMPIDist testMMdoubleMPIDist testMMdoubleMPIDist64 testHODLR testHelmholtz")
else()
  set(STRUMPACK_C_COMPILER "${CMAKE_C_COMPILER}")
  set(STRUMPACK_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
endif()
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile.in
  ${CMAKE_SOURCE_DIR}/examples/Makefile @ONLY)

# copy the examples and Makefile to the build directory as well
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile
  ${CMAKE_BINARY_DIR}/examples/Makefile COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/README
  ${CMAKE_BINARY_DIR}/examples/README COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/examples/data/pde900.mtx
  ${CMAKE_BINARY_DIR}/examples/data/pde900.mtx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/data/susy_10Kn_train.csv
  ${CMAKE_BINARY_DIR}/examples/data/susy_10Kn_train.csv COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/data/susy_10Kn_train_label.csv
  ${CMAKE_BINARY_DIR}/examples/data/susy_10Kn_train_label.csv COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/data/susy_10Kn_test.csv
  ${CMAKE_BINARY_DIR}/examples/data/susy_10Kn_test.csv COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/data/susy_10Kn_test_label.csv
  ${CMAKE_BINARY_DIR}/examples/data/susy_10Kn_test_label.csv COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/examples/testHODLR.cpp
  ${CMAKE_BINARY_DIR}/examples/testHODLR.cpp COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/examples/testHelmholtz.cpp
  ${CMAKE_BINARY_DIR}/examples/testHelmholtz.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/genmatrix3D_core.f
  ${CMAKE_BINARY_DIR}/examples/genmatrix3D_core.f COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2d.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson2d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson3d.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson3d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdouble.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdouble.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/mtx2bin.cpp
  ${CMAKE_BINARY_DIR}/examples/mtx2bin.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/bin2mtx.cpp
  ${CMAKE_BINARY_DIR}/examples/bin2mtx.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegression.cpp
  ${CMAKE_BINARY_DIR}/examples/KernelRegression.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegression.py
  ${CMAKE_BINARY_DIR}/examples/KernelRegression.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegressionMPI.py
  ${CMAKE_BINARY_DIR}/examples/KernelRegressionMPI.py COPYONLY)

if(STRUMPACK_C_INTERFACE)
  configure_file(${CMAKE_SOURCE_DIR}/examples/sexample.c
    ${CMAKE_BINARY_DIR}/examples/sexample.c COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/dexample.c
    ${CMAKE_BINARY_DIR}/examples/dexample.c COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/cexample.c
    ${CMAKE_BINARY_DIR}/examples/cexample.c COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/zexample.c
    ${CMAKE_BINARY_DIR}/examples/zexample.c COPYONLY)
endif()

if(STRUMPACK_USE_MPI)
  configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2dMPIDist.cpp
    ${CMAKE_BINARY_DIR}/examples/testPoisson2dMPIDist.cpp COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson3dMPIDist.cpp
    ${CMAKE_BINARY_DIR}/examples/testPoisson3dMPIDist.cpp COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist.cpp
    ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist.cpp COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist64.cpp
    ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist64.cpp COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegressionMPI.cpp
    ${CMAKE_BINARY_DIR}/examples/KernelRegressionMPI.cpp COPYONLY)
endif()

if(STRUMPACK_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()
