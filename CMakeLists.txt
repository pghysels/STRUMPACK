cmake_minimum_required(VERSION 2.8)
project(STRUMPACK CXX C Fortran)

set(STRUMPACK_VERSION_MAJOR 1)
set(STRUMPACK_VERSION_MINOR 9)
set(STRUMPACK_VERSION_PATCH 9)
set(STRUMPACK_VERSION_FULL "${STRUMPACK_VERSION_MAJOR}.${STRUMPACK_VERSION_MINOR}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
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
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}" )
  set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
endif()
if(MPI_Fortran_FOUND)
  message(STATUS "Found Fortran mpi compiler: " "${MPI_Fortran_COMPILER}")
  include_directories(${MPI_Fortran_INCLUDE_PATH})
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_Fortran_COMPILE_FLAGS}" )
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_Fortran_LINK_FLAGS}" )
  set(LIBS ${LIBS} ${MPI_Fortran_LIBRARIES})
endif()

include(FortranCInterface)
FortranCInterface_HEADER("${PROJECT_BINARY_DIR}/strumpack_config.h" MACRO_NAMESPACE "FC_")
include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src)
FortranCInterface_VERIFY(CXX)

option(USE_OPENMP "Use OpenMP for on-node threading tasking" ON)
if(USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "WARNING: Could not detect OpenMP compiler support, will proceed without OpenMP!")
  endif()
else()
  check_cxx_compiler_flag("-Wno-unknown-pragmas" COMPILER_SUPPORTS_NOPRAGMA_WARNING)
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
  message(WARNING "WARNING: Could not find a PTHREADS library, might be needed by Scotch !!!!")
endif()

find_package(BLAS)
if(BLAS_FOUND)
  message(STATUS "Found BLAS library:   " "${BLAS_LIBRARIES}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  set(LIBS ${LIBS} ${BLAS_LIBRARIES})
else()
  message(WARNING "WARNING: Could not find a BLAS library !!!!!!!!!!!\n"
    "Note that on some machines (like Cray's) BLAS is linked in by the compiler wrapper")
endif()

find_package(LAPACK)
if(LAPACK_FOUND)
  message(STATUS "Found LAPACK library: " "${LAPACK_LIBRARIES}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES})
else()
  message(WARNING "WARNING: Could not find a LAPACK library !!!!!!!!!\n"
    "Note that on some machines (like Cray's) LAPACK is linked in by the compiler wrapper")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(SCALAPACK)
if (SCALAPACK_FOUND)
  set(LIBS ${LIBS} ${SCALAPACK_LIBRARIES})
  include_directories(${SCALAPACK_INCLUDES})
else()
  message(WARNING "WARNING: Could not find a ScaLAPACK/BLACS library !!!!!!!!\n"
    "Note that on some machines (like Cray's) ScaLAPACK is linked in by the compiler wrapper")
endif()

find_package(ParMetis)
set(LIBS ${LIBS} ${PARMETIS_LIBRARIES})
include_directories(${PARMETIS_INCLUDES})

find_package(Metis)
set(LIBS ${LIBS} ${METIS_LIBRARIES})
include_directories(${METIS_INCLUDES})

find_package(Scotch)
set(LIBS ${LIBS} ${SCOTCH_LIBRARIES})
include_directories(${SCOTCH_INCLUDES})

message(STATUS "Linking with: " "${LIBS}")
message(STATUS "Implicit C libs: " "${CMAKE_C_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "Implicit CXX libs: " "${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "Implicit Fortran libs: " "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")

foreach(arg ${LIBS})
  set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} ${arg}")
endforeach(arg ${LIBS})
foreach(arg ${CMAKE_C_IMPLICIT_LINK_LIBRARIES})
  set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} -l${arg}")
endforeach(arg ${CMAKE_C_IMPLICIT_LINK_LIBRARIES})
foreach(arg ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
  set(STRUMPACK_ALL_LIBS "${STRUMPACK_ALL_LIBS} -l${arg}")
endforeach(arg ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
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

# src/StrumpackSparseSolver.cpp
set(LIBSRC src/blas_lapack_omp_task.cpp src/strumpack_mc64ad.c
  src/strumpack_parameters.cpp src/TaskTimer.cpp
  src/lapack/sgeqp3tol.f src/lapack/dgeqp3tol.f src/lapack/cgeqp3tol.f src/lapack/zgeqp3tol.f
  src/lapack/sgeqrfmod.f src/lapack/dgeqrfmod.f src/lapack/cgeqrfmod.f src/lapack/zgeqrfmod.f
  src/lapack/sgeqr2mod.f src/lapack/dgeqr2mod.f src/lapack/cgeqr2mod.f src/lapack/zgeqr2mod.f
  src/lapack/slaqpsmod.f src/lapack/dlaqpsmod.f src/lapack/claqpsmod.f src/lapack/zlaqpsmod.f
  src/lapack/slaqp2mod.f src/lapack/dlaqp2mod.f src/lapack/claqp2mod.f src/lapack/zlaqp2mod.f
  src/lapack/sgetrfmod.f src/lapack/dgetrfmod.f src/lapack/cgetrfmod.f src/lapack/zgetrfmod.f
  src/lapack/sgetf2mod.f src/lapack/dgetf2mod.f src/lapack/cgetf2mod.f src/lapack/zgetf2mod.f
  src/lapack/sorglqmod.f src/lapack/dorglqmod.f src/lapack/cunglqmod.f src/lapack/zunglqmod.f
  src/lapack/sorgl2mod.f src/lapack/dorgl2mod.f src/lapack/cungl2mod.f src/lapack/zungl2mod.f
  src/lapack/slarfbmod.f src/lapack/dlarfbmod.f src/lapack/clarfbmod.f src/lapack/zlarfbmod.f
  src/lapack/slarftmod.f src/lapack/dlarftmod.f src/lapack/clarftmod.f src/lapack/zlarftmod.f
  src/lapack/slarfmod.f  src/lapack/dlarfmod.f  src/lapack/clarfmod.f  src/lapack/zlarfmod.f
  src/lapack/sgelqfmod.f src/lapack/dgelqfmod.f src/lapack/cgelqfmod.f src/lapack/zgelqfmod.f
  src/lapack/sgelq2mod.f src/lapack/dgelq2mod.f src/lapack/cgelq2mod.f src/lapack/zgelq2mod.f
  src/lapack/pcgeqpfmod.f src/lapack/pdgeqpfmod.f src/lapack/psgeqpfmod.f src/lapack/pzgeqpfmod.f
  src/lapack/cgeqp3mod.f  src/lapack/dgeqp3mod.f  src/lapack/sgeqp3mod.f  src/lapack/zgeqp3mod.f
  src/rcm/rcm.f src/rcm/degree.f src/rcm/fnroot.f src/rcm/genrcm.f src/rcm/rootls.f src/MUMPS/ana_orderings.F)
add_library(strumpack ${LIBSRC})
target_link_libraries(strumpack ${LIBS})

install(TARGETS strumpack LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${PROJECT_BINARY_DIR}/strumpack_config.h src/StrumpackSparseSolver.h src/BiCGStab.hpp
  src/BiCGStabMPI.hpp src/CSCMatrix.hpp src/CSRGraph.hpp src/CSRMatrix.hpp src/CSRMatrixMPI.hpp
  src/CompressedSparseMatrix.hpp src/DistributedMatrix.hpp src/ETree.hpp
  src/EliminationTree.hpp src/EliminationTreeMPI.hpp src/EliminationTreeMPIDist.hpp
  src/FrontalMatrix.hpp src/FrontalMatrixDense.hpp src/FrontalMatrixDenseMPI.hpp
  src/FrontalMatrixHSS.hpp src/FrontalMatrixHSSMPI.hpp src/FrontalMatrixMPI.hpp
  src/GMRes.hpp src/GMResMPI.hpp src/GeometricReordering.hpp src/GeometricReorderingMPI.hpp
  src/IterativeRefinement.hpp src/IterativeRefinementMPI.hpp src/DenseMatrix.hpp
  src/MPI_wrapper.hpp src/MatrixReordering.hpp src/MatrixReorderingMPI.hpp src/MetisReordering.hpp
  src/PTScotchReordering.hpp src/ParMetisReordering.hpp src/ProportionallyDistributedSparseMatrix.hpp
  src/Random_wrapper.hpp src/Redistribute.hpp src/ScotchReordering.hpp src/StrumpackSparseSolver.hpp
  src/StrumpackSparseSolverMPI.hpp src/StrumpackSparseSolverMPIDist.hpp src/TaskTimer.hpp
  src/blas_lapack_omp_task.hpp src/blas_lapack_wrapper.hpp src/SPOptions.hpp src/ExtendAdd.hpp
  src/SeparatorTree.hpp src/strumpack_parameters.hpp src/tools.hpp src/RCMReordering.hpp
  src/mumps_symqamd.hpp src/scalapack.hpp src/blacs.h DESTINATION include)

install(FILES src/HSS/HSSMatrix.hpp src/HSS/HSSMatrix.factor.hpp src/HSS/HSSMatrix.compress.hpp
  src/HSS/HSSMatrix.compress_stable.hpp src/HSS/HSSMatrix.extract.hpp src/HSS/HSSBasisID.hpp
  src/HSS/HSSMatrix.Schur.hpp src/HSS/HSSMatrix.apply.hpp src/HSS/HSSMatrix.solve.hpp src/HSS/HSSExtra.hpp
  src/HSS/HSSMatrixBase.hpp src/HSS/HSSPartitionTree.hpp src/HSS/HSSOptions.hpp
  src/HSS/HSSMatrixMPI.hpp src/HSS/HSSMatrixMPI.apply.hpp src/HSS/HSSMatrixMPI.compress.hpp
  src/HSS/HSSMatrixMPI.factor.hpp src/HSS/HSSMatrixMPI.extract.hpp src/HSS/HSSMatrixMPI.solve.hpp
  src/HSS/HSSExtraMPI.hpp src/HSS/HSSBasisIDMPI.hpp src/HSS/DistSamples.hpp src/HSS/DistElemMult.hpp
  src/HSS/HSSMatrixMPI.compress_stable.hpp src/HSS/HSSMatrixMPI.Schur.hpp
  src/HSS/BlockCyclic2BlockRow.hpp DESTINATION include/HSS)

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION_MAJOR ${STRUMPACK_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${STRUMPACK_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${STRUMPACK_VERSION_PATCH})
include(CPack)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/doc/doxygen/doxygen.dox.in ${CMAKE_BINARY_DIR}/doxygen.dox @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doxygen.dox
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)


# build an example makefile
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile.in ${CMAKE_SOURCE_DIR}/examples/Makefile)

# copy the examples and Makefile to the build directory as well
configure_file(${CMAKE_SOURCE_DIR}/examples/Makefile ${CMAKE_BINARY_DIR}/examples/Makefile COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/pde900.mtx ${CMAKE_BINARY_DIR}/examples/pde900.mtx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/README ${CMAKE_BINARY_DIR}/examples/README COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/examples/sexample.c ${CMAKE_BINARY_DIR}/examples/sexample.c COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/dexample.c ${CMAKE_BINARY_DIR}/examples/dexample.c COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/cexample.c ${CMAKE_BINARY_DIR}/examples/cexample.c COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/zexample.c ${CMAKE_BINARY_DIR}/examples/zexample.c COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2d.cpp ${CMAKE_BINARY_DIR}/examples/testPoisson2d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson3d.cpp ${CMAKE_BINARY_DIR}/examples/testPoisson3d.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2dMPI.cpp ${CMAKE_BINARY_DIR}/examples/testPoisson2dMPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testPoisson2dMPIDist.cpp ${CMAKE_BINARY_DIR}/examples/testPoisson2dMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdouble.cpp ${CMAKE_BINARY_DIR}/examples/testMMdouble.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPI.cpp ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPI.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist.cpp ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdouble64.cpp ${CMAKE_BINARY_DIR}/examples/testMMdouble64.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMdoubleMPIDist64.cpp ${CMAKE_BINARY_DIR}/examples/testMMdoubleMPIDist64.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/testMMfloat.cpp ${CMAKE_BINARY_DIR}/examples/testMMfloat.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/mtx2bin.cpp ${CMAKE_BINARY_DIR}/examples/mtx2bin.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/bin2mtx.cpp ${CMAKE_BINARY_DIR}/examples/bin2mtx.cpp COPYONLY)


include(CTest)
add_subdirectory(test)

