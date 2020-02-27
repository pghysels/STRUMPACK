cmake_minimum_required(VERSION 3.13)
project(STRUMPACK VERSION 3.3.0 LANGUAGES CXX C Fortran)


option(STRUMPACK_USE_MPI     "Build with MPI support" ON)
option(STRUMPACK_USE_OPENMP  "Use OpenMP for on-node threading tasking" ON)

option(TPL_ENABLE_SLATE      "Use SLATE, the ECP ScaLAPACK replacement" OFF)
option(TPL_ENABLE_CUBLAS     "Build with support for cuBLAS/cuSOLVER" OFF)
option(TPL_ENABLE_PARMETIS   "Build with support for ParMetis" ON)
option(TPL_ENABLE_SCOTCH     "Build with support for Scotch" ON)
option(TPL_ENABLE_PTSCOTCH   "Build with support for PTScotch" ON)
option(TPL_ENABLE_PAPI       "Build with support for PAPI monitoring" OFF)
option(TPL_ENABLE_COMBBLAS   "Use CombBLAS for weighted matching" OFF)
option(TPL_ENABLE_BPACK      "Use BPACK (ButterflyPACK) code by Yang Liu" OFF)
option(TPL_ENABLE_ZFP        "Build with support for ZFP compression" ON)

option(STRUMPACK_COUNT_FLOPS "Build with flop counters" OFF)
option(STRUMPACK_TASK_TIMERS "Build with timers for internal routines" OFF)
# option(STRUMPACK_BUILD_TESTS "Build the tests" ON)
option(STRUMPACK_DEV_TESTING "Enable extensive testing" OFF)
option(STRUMPACK_C_INTERFACE "Build the C interface" ON)


# figure out Fortran/C name mangling
include(FortranCInterface)
FortranCInterface_HEADER(
  ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  MACRO_NAMESPACE "STRUMPACK_FC_")
FortranCInterface_VERIFY(CXX)

if(STRUMPACK_USE_MPI)
  find_package(MPI)
  if(NOT MPI_FOUND)
    set(STRUMPACK_USE_MPI OFF)
    message(WARNING "Configuring witout MPI support.")
  endif()
endif()
if(NOT STRUMPACK_USE_MPI)
  if(TPL_ENABLE_PARMETIS)
    message(STATUS "Disabling ParMetis, since it requires MPI support.")
    set(TPL_ENABLE_PARMETIS OFF)
  endif()
  if(TPL_ENABLE_PTSCOTCH)
    message(STATUS "Disabling PTScotch, since it requires MPI support.")
    set(TPL_ENABLE_PTSCOTCH OFF)
  endif()
  if(TPL_ENABLE_BPACK)
    message(STATUS "Disabling ButterflyPACK, since it requires MPI support.")
    set(TPL_ENABLE_BPACK OFF)
  endif()
endif()


if(STRUMPACK_USE_OPENMP)
  include(FindOpenMP)
  if(OpenMP_FOUND)
    set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/omp_compile_test)
    try_compile(STRUMPACK_USE_OPENMP_TASK_DEPEND ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/taskdepend.cpp
      LINK_LIBRARIES OpenMP::OpenMP_CXX)
    message(STATUS "Support for OpenMP task dependencies: "
      ${STRUMPACK_USE_OPENMP_TASK_DEPEND})
    try_compile(STRUMPACK_USE_OPENMP_TASKLOOP ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/taskloop.cpp
      LINK_LIBRARIES OpenMP::OpenMP_CXX)
    message(STATUS "Support for OpenMP taskloop: "
      ${STRUMPACK_USE_OPENMP_TASKLOOP})
  else()
    message(WARNING "Compiler does not support OpenMP,"
      "proceeding without OpenMP support.")
  endif()
else()
  message(STATUS "OpenMP support was not enabled.")
endif()


# TODO also look at TPL_... dirs
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake/ModulesModern")

if(STRUMPACK_USE_MPI)
  # TODO find a better way for this?
  if(DEFINED TPL_SCALAPACK_LIBRARIES)
    set(LIBS ${SCALAPACK_LIBRARIES} ${TPL_SCALAPACK_LIBRARIES})
  else()
    find_package(SCALAPACK REQUIRED)
  endif()
endif()

set(metis_PREFIX ${TPL_METIS_PREFIX})
set(metis_INCLUDE_DIR ${TPL_METIS_INCLUDE_DIRS})
set(metis_LIBRARY_DIR ${TPL_METIS_LIBRARY_DIR})
set(metis_LIBRARIES ${TPL_METIS_LIBRARIES})
find_package(METIS REQUIRED)

if(TPL_ENABLE_SCOTCH)
  set(scotch_PREFIX ${TPL_SCOTCH_PREFIX})
  set(scotch_INCLUDE_DIR ${TPL_SCOTCH_INCLUDE_DIRS})
  set(scotch_LIBRARY_DIR ${TPL_SCOTCH_LIBRARY_DIR})
  set(scotch_LIBRARIES ${TPL_SCOTCH_LIBRARIES})
  find_package(SCOTCH)
  if(SCOTCH_FOUND)
    option(STRUMPACK_USE_SCOTCH "" ON)
  endif()
endif()

if(TPL_ENABLE_PTSCOTCH)
  # TODO if these TPL_PTSCOTCH_.. variables do not exist,
  # then check TPL_SCOTCH_..
  set(ptscotch_PREFIX ${TPL_PTSCOTCH_PREFIX})
  set(ptscotch_INCLUDE_DIR ${TPL_PTSCOTCH_INCLUDE_DIRS})
  set(ptscotch_LIBRARY_DIR ${TPL_PTSCOTCH_LIBRARY_DIR})
  set(ptscotch_LIBRARIES ${TPL_PTSCOTCH_LIBRARIES})
  find_package(PTSCOTCH)
  if(PTSCOTCH_FOUND)
    option(STRUMPACK_USE_PTSCOTCH "" ON)
  endif()
endif()

if(TPL_ENABLE_PARMETIS)
  set(parmetis_PREFIX ${TPL_PARMETIS_PREFIX})
  set(parmetis_INCLUDE_DIR ${TPL_PARMETIS_INCLUDE_DIRS})
  set(parmetis_LIBRARY_DIR ${TPL_PARMETIS_LIBRARY_DIR})
  set(parmetis_LIBRARIES ${TPL_PARMETIS_LIBRARIES})
  find_package(ParMETIS)
  if(PARMETIS_FOUND)
    option(STRUMPACK_USE_PARMETIS "" ON)
  endif()
endif()

if(TPL_ENABLE_ZFP)
  set(zfp_PREFIX ${TPL_ZFP_PREFIX})
  set(zfp_INCLUDE_DIR ${TPL_ZFP_INCLUDE_DIRS})
  set(zfp_LIBRARY_DIR ${TPL_ZFP_LIBRARY_DIR})
  set(zfp_LIBRARIES ${TPL_ZFP_LIBRARIES})
  find_package(ZFP)
  if(ZFP_FOUND)
    option(STRUMPACK_USE_ZFP "" ON)
  endif()
endif()

if(TPL_ENABLE_BPACK)
  option(STRUMPACK_USE_BPACK "" ON)
  # TODO in BPACK, export targets, then find his .cmake file?
endif()

# TODO CombBLAS, needs c++14
# TODO PIC
# TODO libatomic?


add_library(strumpack "")
add_subdirectory(src)

target_include_directories(strumpack PUBLIC ${PROJECT_BINARY_DIR})
target_sources(strumpack
  PRIVATE ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h)

target_compile_features(strumpack PUBLIC cxx_std_11)
set_target_properties(strumpack PROPERTIES CXX_EXTENSIONS OFF)

if(STRUMPACK_USE_MPI)
  target_link_libraries(strumpack PUBLIC MPI::MPI_C)
  target_link_libraries(strumpack PUBLIC MPI::MPI_CXX)
  target_link_libraries(strumpack PUBLIC MPI::MPI_Fortran)
endif()

if(OpenMP_FOUND)
  if(OpenMP_C_FOUND)
    target_link_libraries(strumpack PUBLIC OpenMP::OpenMP_C)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(strumpack PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(OpenMP_Fortran_FOUND)
    target_link_libraries(strumpack PUBLIC OpenMP::OpenMP_Fortran)
  endif()
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Wno-unknown-pragmas
    COMPILER_SUPPORTS_NOPRAGMA_WARNING)
  if(COMPILER_SUPPORTS_NOPRAGMA_WARNING)
    target_compile_options(strumpack PUBLIC -Wno-unknown-pragmas)
  endif()
endif()


target_link_libraries(strumpack PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(strumpack PUBLIC ${LAPACK_LIBRARIES})
if(STRUMPACK_USE_MPI)
  target_link_libraries(strumpack PUBLIC ${SCALAPACK_LIBRARIES})
endif()

# the order is important
#   - ParMETIS before METIS, since ParMETIS needs METIS
#   - (ParMETIS) before Scotch to avoid confusion with the Scotch
#     ParMETIS compatibility layer
if(PARMETIS_FOUND)
  target_link_libraries(strumpack PUBLIC ParMETIS::parmetis)
endif()
target_link_libraries(strumpack PUBLIC METIS::metis)
if(SCOTCH_FOUND)
  target_link_libraries(strumpack PUBLIC SCOTCH::scotch)
endif()
if(PTSCOTCH_FOUND)
  target_link_libraries(strumpack PUBLIC PTSCOTCH::ptscotch)
endif()

if(ZFP_FOUND)
  target_link_libraries(strumpack PUBLIC ZFP::zfp)
endif()

if(TPL_ENABLE_BPACK)
  option(STRUMPACK_USE_BPACK "" ON)
  # TODO export BPACK targets, then find his .cmake file?
  target_link_libraries(strumpack PUBLIC ${TPL_BPACK_LIBRARIES})
  target_include_directories(strumpack PRIVATE ${TPL_BPACK_INCLUDE_DIRS})
endif()

add_subdirectory(examples)

include(CTest)
add_subdirectory(test)

install(TARGETS strumpack LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  DESTINATION include)
