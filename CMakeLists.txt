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
option(TPL_ENABLE_BPACK      "Use BPACK (ButterflyPACK) code by Yang Liu" ON)
option(TPL_ENABLE_ZFP        "Build with support for ZFP compression" ON)

option(STRUMPACK_COUNT_FLOPS "Build with flop counters" OFF)
option(STRUMPACK_TASK_TIMERS "Build with timers for internal routines" OFF)
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
    message(WARNING "Configuring without MPI support.")
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
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/omp_compile_test)
    try_compile(STRUMPACK_USE_OPENMP_TASK_DEPEND ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/taskdepend.cpp
      LINK_LIBRARIES OpenMP::OpenMP_CXX)
    message(STATUS "Support for OpenMP task depend/priority: "
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


if(DEFINED TPL_BLAS_LIBRARIES)
  # xSDK policies require the user be able to specify BLAS libraries
  # through the TPL_BLAS_LIBRARIES variable. If these do not work,
  # then an error should be generated, and we shouldn't just continue.
  set(BLAS_LIBRARIES ${TPL_BLAS_LIBRARIES})
  set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/blas_compile_test)
  try_compile(STRUMPACK_TPL_BLAS_WORKS ${_SCRATCH_DIR}
    SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
    LINK_LIBRARIES ${TPL_BLAS_LIBRARIES})
  if(STRUMPACK_TPL_BLAS_WORKS)
    message(STATUS
      "Using BLAS from TPL_BLAS_LIBRARIES (${TPL_BLAS_LIBRARIES})")
  else()
    message(STATUS
      "Linking with TPL_BLAS_LIBRARIES did not work,"
      " trying again with additional threading library linked in.")
    # Compilation with TPL_BLAS_LIBRARIES failed. Perhaps we were
    # missing a threading library? Just guessing. One more try with
    # threading (pthreads) linked in and then we give up.
    find_package(Threads)
    try_compile(STRUMPACK_TPL_BLAS_WITH_THREADS_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
      LINK_LIBRARIES ${TPL_BLAS_LIBRARIES} Threads::Threads)
    if(STRUMPACK_TPL_BLAS_WITH_THREADS_WORKS)
      message(STATUS
        "Using BLAS (${TPL_BLAS_LIBRARIES}), "
        "with additional threading library.")
    else()
      message(FATAL_ERROR
        "BLAS libraries defined in TPL_BLAS_LIBRARIES"
        " (${TPL_BLAS_LIBRARIES}) cannot be used.")
    endif()
  endif()
else()
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    # BLAS was not specified in TPL_BLAS_LIBRARIES, and not found
    # elsewhere. We will still try to compile a BLAS example, because
    # perhaps BLAS is automatically linked by the compiler wrapper, as
    # is the case for instance on Cray machines.
    message(STATUS
      "A BLAS library could not be found on the system, "
      "checking if BLAS is implicitly linked by the compiler(wrapper).")
    set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/blas_compile_test)
    try_compile(STRUMPACK_BLAS_JUST_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
      LINK_LIBRARIES "${TPL_BLAS_LIBRARIES}")
    if(NOT STRUMPACK_BLAS_JUST_WORKS)
      message(FATAL_ERROR
        "Could not find a BLAS library.\n"
         " You can specify a BLAS library using -DTPL_BLAS_LIBRARIES=\"...\"")
    endif()
  endif()
endif()


if(DEFINED TPL_LAPACK_LIBRARIES)
  # xSDK policies require the user be able to specify LAPACK libraries
  # through the TPL_LAPACK_LIBRARIES variable. If these do not work,
  # then an error should be generated, and we shouldn't just continue.
  set(LAPACK_LIBRARIES ${LAPACK_BLAS_LIBRARIES})
  set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack_compile_test)
  try_compile(STRUMPACK_TPL_LAPACK_WORKS ${_SCRATCH_DIR}
    SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
    LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES})
  if(STRUMPACK_TPL_LAPACK_WORKS)
    message(STATUS
      "Using LAPACK from TPL_LAPACK_LIBRARIES (${TPL_LAPACK_LIBRARIES})")
  else()
    message(STATUS
      "Linking with TPL_LAPACK_LIBRARIES did not work,"
      " trying again with additional threading library linked in.")
    # Compilation with TPL_LAPACK_LIBRARIES failed. Perhaps we were
    # missing a threading library? Just guessing. One more try with
    # threading (pthreads) linked in and then we give up.
    find_package(Threads)
    try_compile(STRUMPACK_TPL_LAPACK_WITH_THREADS_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
      LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES} Threads::Threads)
    if(STRUMPACK_TPL_LAPACK_WITH_THREADS_WORKS)
      message(STATUS
        "Using LAPACK (${TPL_LAPACK_LIBRARIES}), "
        "with additional threading library.")
    else()
      message(FATAL_ERROR
        "LAPACK libraries defined in TPL_LAPACK_LIBRARIES"
        " (${TPL_LAPACK_LIBRARIES}) cannot be used.")
    endif()
  endif()
else()
  if(DEFINED TPL_BLAS_LIBRARIES)
    message(WARNING
      "TPL_BLAS_LIBRARIES was specified. "
      "Consider specifying TPL_LAPACK_LIBRARIES as well to ensure "
      "compatible BLAS and LAPACK libraries.")
  endif()
  find_package(LAPACK REQUIRED)
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")


if(STRUMPACK_USE_MPI)
  if(DEFINED TPL_SCALAPACK_LIBRARIES)
    check_library_exists(${TPL_SCALAPACK_LIBRARIES} pdgemm_ ""
      TPL_SCALAPACK_WORKS)
    if(TPL_SCALAPACK_WORKS)
      message(STATUS
        "Using ScaLAPACK from TPL_SCALAPACK_LIBRARIES"
        " (${TPL_SCALAPACK_LIBRARIES})")
      set(${SCALAPACK_LIBRARIES} ${TPL_SCALAPACK_LIBRARIES})
    else()
      message(FATAL_ERROR
        "Cannot use TPL_SCALAPACK_LIBRARIES (${TPL_SCALAPACK_LIBRARIES})")
    endif()
  else()
    find_package(SCALAPACK REQUIRED)
  endif()
endif()


list(APPEND CMAKE_PREFIX_PATH
  ${TPL_METIS_PREFIX} $ENV{METIS_DIR} $ENV{METIS_ROOT})
if(NOT DEFINED metis_INCLUDE_DIR)
  set(metis_INCLUDE_DIR ${TPL_METIS_INCLUDE_DIRS})
endif()
if(NOT DEFINED metis_LIBRARY_DIR)
  set(metis_LIBRARY_DIR ${TPL_METIS_LIBRARY_DIR})
endif()
if(NOT DEFINED metis_LIBRARIES)
  set(metis_LIBRARIES ${TPL_METIS_LIBRARIES})
endif()
find_package(METIS REQUIRED)

if(TPL_ENABLE_SCOTCH)
  list(APPEND CMAKE_PREFIX_PATH
    ${TPL_SCOTCH_PREFIX} $ENV{SCOTCH_DIR} $ENV{SCOTCH_ROOT})
  if(NOT DEFINED scotch_INCLUDE_DIR)
    set(scotch_INCLUDE_DIR ${TPL_SCOTCH_INCLUDE_DIRS})
  endif()
  if(NOT DEFINED scotch_LIBRARY_DIR)
    set(scotch_LIBRARY_DIR ${TPL_SCOTCH_LIBRARY_DIR})
  endif()
  if(NOT DEFINED scotch_LIBRARIES)
    set(scotch_LIBRARIES ${TPL_SCOTCH_LIBRARIES})
  endif()
  find_package(SCOTCH)
  if(SCOTCH_FOUND)
    option(STRUMPACK_USE_SCOTCH "" ON)
  endif()
endif()

if(TPL_ENABLE_PTSCOTCH)
  list(APPEND CMAKE_PREFIX_PATH
    ${TPL_PTSCOTCH_PREFIX} $ENV{PTSCOTCH_DIR} $ENV{PTSCOTCH_ROOT})
  if(NOT DEFINED ptscotch_INCLUDE_DIR)
    set(ptscotch_INCLUDE_DIR ${TPL_PTSCOTCH_INCLUDE_DIRS})
  endif()
  if(NOT DEFINED ptscotch_LIBRARY_DIR)
    set(ptscotch_LIBRARY_DIR ${TPL_PTSCOTCH_LIBRARY_DIR})
  endif()
  if(NOT DEFINED ptscotch_LIBRARIES)
    set(ptscotch_LIBRARIES ${TPL_PTSCOTCH_LIBRARIES})
  endif()
  find_package(PTSCOTCH)
  if(PTSCOTCH_FOUND)
    option(STRUMPACK_USE_PTSCOTCH "" ON)
  endif()
endif()

if(TPL_ENABLE_PARMETIS)
  list(APPEND CMAKE_PREFIX_PATH
    ${TPL_PARMETIS_PREFIX} $ENV{PARMETIS_DIR} $ENV{PARMETIS_ROOT}
    $ENV{ParMETIS_DIR} $ENV{ParMETIS_ROOT})
  if(NOT DEFINED parmetis_INCLUDE_DIR)
    set(parmetis_INCLUDE_DIR ${TPL_PARMETIS_INCLUDE_DIRS})
  endif()
  if(NOT DEFINED parmetis_LIBRARY_DIR)
    set(parmetis_LIBRARY_DIR ${TPL_PARMETIS_LIBRARY_DIR})
  endif()
  if(NOT DEFINED parmetis_LIBRARIES)
    set(parmetis_LIBRARIES ${TPL_PARMETIS_LIBRARIES})
  endif()
  find_package(ParMETIS)
  if(ParMETIS_FOUND)
    option(STRUMPACK_USE_PARMETIS "" ON)
  endif()
endif()

if(TPL_ENABLE_ZFP)
  list(APPEND CMAKE_PREFIX_PATH
    ${TPL_ZFP_PREFIX} $ENV{ZFP_DIR} $ENV{ZFP_ROOT})
  if(NOT DEFINED zfp_INCLUDE_DIR)
    set(zfp_INCLUDE_DIR ${TPL_ZFP_INCLUDE_DIRS})
  endif()
  if(NOT DEFINED zfp_LIBRARY_DIR)
    set(zfp_LIBRARY_DIR ${TPL_ZFP_LIBRARY_DIR})
  endif()
  if(NOT DEFINED zfp_LIBRARIES)
    set(zfp_LIBRARIES ${TPL_ZFP_LIBRARIES})
  endif()
  find_package(ZFP)
  if(ZFP_FOUND)
    option(STRUMPACK_USE_ZFP "" ON)
  endif()
endif()

if(TPL_ENABLE_BPACK)
  list(APPEND CMAKE_PREFIX_PATH ${TPL_BUTTERFLYPACK_PREFIX}
    $ENV{ButterflyPACK_DIR} $ENV{ButterflyPACK_ROOT}
    $ENV{BUTTERFLYPACK_DIR} $ENV{BUTTERFLYPACK_ROOT})
  find_package(ButterflyPACK)
  if(ButterflyPACK_FOUND)
    option(STRUMPACK_USE_BPACK "" ON)
  endif()
endif()

# TODO CombBLAS, needs c++14
# TODO PIC
# TODO libatomic?


add_library(strumpack "")
add_subdirectory(src)

target_sources(strumpack
  PRIVATE ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h)

target_compile_features(strumpack INTERFACE cxx_std_11)
set_target_properties(strumpack PROPERTIES CXX_EXTENSIONS OFF)

if(STRUMPACK_USE_MPI)
  target_link_libraries(strumpack PUBLIC MPI::MPI_CXX)
  target_link_libraries(strumpack PRIVATE MPI::MPI_C)
  target_link_libraries(strumpack PRIVATE MPI::MPI_Fortran)
endif()

if(OpenMP_FOUND)
  # if(OpenMP_C_FOUND)
  #   target_link_libraries(strumpack PRIVATE OpenMP::OpenMP_C)
  # endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(strumpack PUBLIC OpenMP::OpenMP_CXX)
  endif()
  # if(OpenMP_Fortran_FOUND)
  #   target_link_libraries(strumpack PRIVATE OpenMP::OpenMP_Fortran)
  # endif()
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
if(ParMETIS_FOUND)
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

if(ButterflyPACK_FOUND)
  target_link_libraries(strumpack PUBLIC
    ButterflyPACK::dbutterflypack ButterflyPACK::zbutterflypack)
endif()


# examples
add_subdirectory(examples)

# testing
include(CTest)
add_subdirectory(test)

# documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/doc/doxygen/doxygen.dox.in
    ${CMAKE_BINARY_DIR}/doxygen.dox @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doxygen.dox
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with doxygen" VERBATIM)
endif()


# installation
install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h
  DESTINATION include)

install(TARGETS strumpack EXPORT STRUMPACKTargets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)


# export targets and all
set(ConfigPackageLocation lib/cmake/STRUMPACK)

install(EXPORT STRUMPACKTargets FILE strumpack-targets.cmake
  NAMESPACE STRUMPACK:: DESTINATION ${ConfigPackageLocation})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(strumpack-config-version.cmake
  VERSION ${STRUMPACK_VERSION} COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_BINARY_DIR}/strumpack-config-version.cmake"
  DESTINATION ${ConfigPackageLocation})

# stupid CMake makes me write one more file
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/strumpack-config.cmake.in
  ${CMAKE_BINARY_DIR}/strumpack-config.cmake
  INSTALL_DESTINATION ${ConfigPackageLocation})

install(FILES "${CMAKE_BINARY_DIR}/strumpack-config.cmake"
  DESTINATION ${ConfigPackageLocation})

# install the FindMETIS, etc modules for STRUMPACK users
install(DIRECTORY cmake/Modules/
  DESTINATION ${ConfigPackageLocation}
  FILES_MATCHING PATTERN "*.cmake")
