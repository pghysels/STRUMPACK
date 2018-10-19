cmake_minimum_required(VERSION 2.8)
project(STRUMPACK CXX C Fortran)

# option(STRUMPACK_USE_METIS "Build with support for Metis" ON)
option(STRUMPACK_USE_PARMETIS "Build with support for ParMetis" OFF)
option(STRUMPACK_USE_SCOTCH "Build with support for Scotch/PTScotch" OFF)
option(STRUMPACK_USE_OPENMP "Use OpenMP for on-node threading tasking" OFF)
option(STRUMPACK_USE_PAPI "Build with support for PAPI monitoring" OFF)
option(STRUMPACK_USE_COMBBLAS "Use CombBLAS for weighted matching" OFF)
option(STRUMPACK_COUNT_FLOPS "Build with flop counters" OFF)
option(STRUMPACK_TASK_TIMERS "Build with timers for internal routines" OFF)
option(STRUMPACK_BUILD_TESTS "Build the tests" OFF)
option(STRUMPACK_DEV_TESTING "Enable extensive testing" OFF)
option(STRUMPACK_C_INTERFACE "Build the C interface" OFF)

set(STRUMPACK_VERSION_MAJOR 2)
set(STRUMPACK_VERSION_MINOR 3)
set(STRUMPACK_VERSION_PATCH 0)
set(STRUMPACK_VERSION_FULL
  "${STRUMPACK_VERSION_MAJOR}.${STRUMPACK_VERSION_MINOR}")

if(STRUMPACK_USE_COMBBLAS)
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

include(FortranCInterface)
FortranCInterface_HEADER(${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  MACRO_NAMESPACE "FC_")
include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src)
FortranCInterface_VERIFY(CXX)

if(STRUMPACK_USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "WARNING: Could not detect OpenMP compiler
    support, will proceed without OpenMP!")
  endif()
else()
  message(STATUS "Building STRUMPACK without OpenMP support.")
  check_cxx_compiler_flag("-Wno-unknown-pragmas"
    COMPILER_SUPPORTS_NOPRAGMA_WARNING)
  if (COMPILER_SUPPORTS_NOPRAGMA_WARNING)
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

if(DEFINED BLAS_LIBRARIES)
  set(LIBS ${LIBS} ${BLAS_LIBRARIES})
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

if(DEFINED LAPACK_LIBRARIES)
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES})
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

if(DEFINED SCALAPACK_LIBRARIES)
  set(LIBS ${LIBS} ${SCALAPACK_LIBRARIES})
  include_directories(${SCALAPACK_INCLUDES})
else()
  find_package(SCALAPACK)
  if(SCALAPACK_FOUND)
    set(LIBS ${LIBS} ${SCALAPACK_LIBRARIES})
    include_directories(${SCALAPACK_INCLUDES})
  else()
    message(WARNING
      "WARNING: Could not find a ScaLAPACK/BLACS library !!!!!!!!\n"
      "Note that on some machines (like Cray's)"
      " ScaLAPACK is linked in by the compiler wrapper")
  endif()
endif()

if(STRUMPACK_USE_PAPI)
  find_package(PAPI)
  set(LIBS ${LIBS} ${PAPI_LIBRARIES})
  include_directories(${PAPI_INCLUDES})
  if(NOT PAPI_FOUND)
    message(FATAL_ERROR
      "Could not find PAPI!"
      "Set the PAPIDIR environtment variable,"
      "or set specify PAPI_INCLUDES and PAPI_LIBRARIES."
      "Get PAPI from http://icl.cs.utk.edu/papi/")
  endif()
else()
  message(STATUS "Building STRUMPACK without PAPI support.")
endif()

if(STRUMPACK_USE_COMBBLAS)
  set(LIBS ${LIBS} ${COMBBLAS_LIBRARIES})
  include_directories(${COMBBLAS_INCLUDES})
  message(STATUS "Combinatorial BLAS support enabled")
endif()

if(STRUMPACK_USE_PARMETIS)
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

if(STRUMPACK_USE_SCOTCH)
  find_package(Scotch)
  set(LIBS ${LIBS} ${SCOTCH_LIBRARIES})
  include_directories(${SCOTCH_INCLUDES})
else()
  message(STATUS "Building STRUMPACK without Scotch/PTScotch support.")
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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  set(STRUMPACK_ALL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
  set(STRUMPACK_ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()


set(LIBSRC
  src/StrumpackParameters.cpp
  src/misc/TaskTimer.cpp)

if(STRUMPACK_C_INTERFACE)
  list(APPEND LIBSRC src/StrumpackSparseSolver.cpp)
endif()

set(LIBSRCDENSE
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
  src/dense/lapack/slarfmod.f src/dense/lapack/dlarfmod.f
  src/dense/lapack/clarfmod.f src/dense/lapack/zlarfmod.f
  src/dense/lapack/sgelqfmod.f src/dense/lapack/dgelqfmod.f
  src/dense/lapack/cgelqfmod.f src/dense/lapack/zgelqfmod.f
  src/dense/lapack/sgelq2mod.f src/dense/lapack/dgelq2mod.f
  src/dense/lapack/cgelq2mod.f src/dense/lapack/zgelq2mod.f
  src/dense/lapack/pcgeqpfmod.f src/dense/lapack/pdgeqpfmod.f
  src/dense/lapack/psgeqpfmod.f src/dense/lapack/pzgeqpfmod.f
  src/dense/lapack/cgeqp3mod.f src/dense/lapack/dgeqp3mod.f
  src/dense/lapack/sgeqp3mod.f src/dense/lapack/zgeqp3mod.f)

set(LIBSRCSPARSE
  src/sparse/rcm/rcm.f
  src/sparse/rcm/degree.f
  src/sparse/rcm/fnroot.f
  src/sparse/rcm/genrcm.f
  src/sparse/rcm/rootls.f
  src/sparse/MUMPS/ana_orderings.F
  src/sparse/strumpack_mc64ad.c)

add_library(strumpack ${LIBSRC} ${LIBSRCSPARSE} ${LIBSRCDENSE})
target_link_libraries(strumpack ${LIBS})


install(TARGETS strumpack LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

configure_file(${CMAKE_SOURCE_DIR}/src/StrumpackConfig.hpp.in
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp)

install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp
  src/StrumpackSparseSolver.h
  src/StrumpackSparseSolver.hpp
  src/StrumpackSparseSolverMPI.hpp
  src/StrumpackSparseSolverMPIDist.hpp
  src/StrumpackOptions.hpp
  src/StrumpackParameters.hpp
  DESTINATION include)

install(FILES
  src/misc/TaskTimer.hpp
  src/misc/MPIWrapper.hpp
  src/misc/RandomWrapper.hpp
  src/misc/Tools.hpp
  DESTINATION include/misc)

install(FILES
  src/clustering/NeighborSearch.hpp
  src/clustering/KMeans.hpp
  src/clustering/KDTree.hpp
  src/clustering/PCAPartitioning.hpp
  src/clustering/CobblePartitioning.hpp
  DESTINATION include/clustering)

install(FILES
  src/dense/DistributedMatrix.hpp
  src/dense/DenseMatrix.hpp
  src/dense/BLASLAPACKOpenMPTask.hpp
  src/dense/BLASLAPACKWrapper.hpp
  src/dense/ScaLAPACKWrapper.hpp
  src/dense/BLACSGrid.hpp
  DESTINATION include/dense)

install(FILES
  src/sparse/BiCGStab.hpp
  src/sparse/BiCGStabMPI.hpp
  src/sparse/CSCMatrix.hpp
  src/sparse/CSRGraph.hpp
  src/sparse/CSRMatrix.hpp
  src/sparse/CSRMatrixMPI.hpp
  src/sparse/CompressedSparseMatrix.hpp
  src/sparse/ETree.hpp
  src/sparse/EliminationTree.hpp
  src/sparse/EliminationTreeMPI.hpp
  src/sparse/EliminationTreeMPIDist.hpp
  src/sparse/FrontalMatrix.hpp
  src/sparse/FrontalMatrixDense.hpp
  src/sparse/FrontalMatrixDenseMPI.hpp
  src/sparse/FrontalMatrixHSS.hpp
  src/sparse/FrontalMatrixHSSMPI.hpp
  src/sparse/FrontalMatrixMPI.hpp
  src/sparse/FrontalMatrixBLR.hpp
  src/sparse/FrontalMatrixBLRMPI.hpp
  src/sparse/GMRes.hpp
  src/sparse/GMResMPI.hpp
  src/sparse/GeometricReordering.hpp
  src/sparse/GeometricReorderingMPI.hpp
  src/sparse/IterativeRefinement.hpp
  src/sparse/IterativeRefinementMPI.hpp
  src/sparse/MatrixReordering.hpp
  src/sparse/MatrixReorderingMPI.hpp
  src/sparse/MetisReordering.hpp
  src/sparse/PTScotchReordering.hpp
  src/sparse/ParMetisReordering.hpp
  src/sparse/ProportionallyDistributedSparseMatrix.hpp
  src/sparse/Redistribute.hpp
  src/sparse/ScotchReordering.hpp
  src/sparse/RCMReordering.hpp
  src/sparse/ExtendAdd.hpp
  src/sparse/SeparatorTree.hpp
  src/sparse/mumps_symqamd.hpp
  DESTINATION include/sparse)

if(STRUMPACK_USE_COMBBLAS)
  install(FILES src/sparse/AWPMCombBLAS.hpp DESTINATION include/sparse)
endif()

install(FILES src/BLR/BLRMatrix.hpp
  src/BLR/BLRMatrixMPI.hpp
  src/BLR/BLROptions.hpp
  DESTINATION include/BLR)

install(FILES src/HSS/HSSMatrix.hpp
  src/HSS/HSSMatrix.factor.hpp
  src/HSS/HSSMatrix.compress.hpp
  src/HSS/HSSMatrix.compress_stable.hpp
  src/HSS/HSSMatrix.compress.kernel.hpp
  src/HSS/HSSMatrix.extract.hpp
  src/HSS/HSSBasisID.hpp
  src/HSS/HSSMatrix.Schur.hpp
  src/HSS/HSSMatrix.apply.hpp
  src/HSS/HSSMatrix.solve.hpp
  src/HSS/HSSExtra.hpp
  src/HSS/HSSMatrixBase.hpp
  src/HSS/HSSPartitionTree.hpp
  src/HSS/HSSOptions.hpp
  src/HSS/HSSMatrixMPI.hpp
  src/HSS/HSSMatrixMPI.apply.hpp
  src/HSS/HSSMatrixMPI.compress.hpp
  src/HSS/HSSMatrixMPI.factor.hpp
  src/HSS/HSSMatrixMPI.extract.hpp
  src/HSS/HSSMatrixMPI.extract_blocks.hpp
  src/HSS/HSSMatrixMPI.solve.hpp
  src/HSS/HSSExtraMPI.hpp
  src/HSS/HSSBasisIDMPI.hpp
  src/HSS/DistSamples.hpp
  src/HSS/DistElemMult.hpp
  src/HSS/HSSMatrixMPI.compress_stable.hpp
  src/HSS/HSSMatrixMPI.compress_kernel.hpp
  src/HSS/HSSMatrixMPI.Schur.hpp
  src/HSS/BlockCyclic2BlockRow.hpp
  DESTINATION include/HSS)

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
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile.in
  ${CMAKE_SOURCE_DIR}/examples/Makefile @ONLY)

# copy the examples and Makefile to the build directory as well
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile
  ${CMAKE_BINARY_DIR}/examples/Makefile COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/pde900.mtx
  ${CMAKE_BINARY_DIR}/examples/pde900.mtx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/README
  ${CMAKE_BINARY_DIR}/examples/README COPYONLY)

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
# configure_file(${CMAKE_SOURCE_DIR}/examples/DenseBEMColStateMPI.cpp
#   ${CMAKE_BINARY_DIR}/examples/DenseBEMColStateMPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2d.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson2d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson3d.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson3d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2dMPI.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson2dMPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2dMPIDist.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson2dMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson3dMPIDist.cpp
  ${CMAKE_BINARY_DIR}/examples/testPoisson3dMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdouble.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdouble.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPI.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdouble64.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdouble64.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist64.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist64.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMfloat.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMfloat.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMfloatMPIDist.cpp
  ${CMAKE_BINARY_DIR}/examples/testMMfloatMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/mtx2bin.cpp
  ${CMAKE_BINARY_DIR}/examples/mtx2bin.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/bin2mtx.cpp
  ${CMAKE_BINARY_DIR}/examples/bin2mtx.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/MLkernel.cpp
  ${CMAKE_BINARY_DIR}/examples/MLkernel.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegression_ann.cpp
  ${CMAKE_BINARY_DIR}/examples/KernelRegression_ann.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/FileManipulation.h
  ${CMAKE_BINARY_DIR}/examples/FileManipulation.h COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/preprocessing.h
  ${CMAKE_BINARY_DIR}/examples/preprocessing.h COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/find_ann.h
  ${CMAKE_BINARY_DIR}/examples/find_ann.h COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegression_ann_MPI.cpp
  ${CMAKE_BINARY_DIR}/examples/KernelRegression_ann_MPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/KernelRegression_rs_MPI.cpp
  ${CMAKE_BINARY_DIR}/examples/KernelRegression_rs_MPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/scratch.cpp
  ${CMAKE_BINARY_DIR}/examples/scratch.cpp COPYONLY)

if(STRUMPACK_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()
