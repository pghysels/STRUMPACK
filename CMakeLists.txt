cmake_minimum_required(VERSION 3.17)
project(STRUMPACK VERSION 7.1.3 LANGUAGES CXX C Fortran)
cmake_policy(SET CMP0074 NEW)

option(STRUMPACK_USE_MPI     "Build with MPI support" ON)
option(STRUMPACK_USE_OPENMP  "Use OpenMP for on-node threading tasking" ON)
option(STRUMPACK_USE_CUDA    "Use CUDA for NVIDIA GPU acceleration" OFF)
option(STRUMPACK_USE_HIP     "Use HIP for AMD or NVIDIA GPU acceleration" OFF)
option(STRUMPACK_USE_SYCL    "Use SYCL/DPC++ for Intel GPU acceleration" OFF)
option(STRUMPACK_USE_BLAS64  "Use 64 bit interfaces to BLAS and LAPACK, e.g., MKL ILP64 or openblas64" OFF)

option(TPL_ENABLE_SLATE      "Use SLATE, the ECP ScaLAPACK replacement" ON)
option(TPL_ENABLE_PARMETIS   "Build with support for ParMetis" ON)
option(TPL_ENABLE_SCOTCH     "Build with support for Scotch" ON)
option(TPL_ENABLE_PTSCOTCH   "Build with support for PTScotch" ON)
option(TPL_ENABLE_PAPI       "Build with support for PAPI monitoring" OFF)
option(TPL_ENABLE_COMBBLAS   "Use CombBLAS for weighted matching" OFF)
option(TPL_ENABLE_BPACK      "Use BPACK (ButterflyPACK) code by Yang Liu" ON)
option(TPL_ENABLE_ZFP        "Build with support for ZFP compression" ON)
option(TPL_ENABLE_MAGMA      "Build with support for MAGMA" OFF)
option(TPL_ENABLE_MATLAB     "Build with Matlab interface" OFF)

option(STRUMPACK_COUNT_FLOPS "Build with flop counters" OFF)
option(STRUMPACK_TASK_TIMERS "Build with timers for internal routines" OFF)
option(STRUMPACK_MESSAGE_COUNTER "Build with counter for MPI messages" OFF)

include(CheckLibraryExists)
# include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

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
  if(TPL_ENABLE_COMBBLAS)
    message(STATUS "Disabling CombBLAS, since it requires MPI support.")
    set(TPL_ENABLE_COMBBLAS OFF)
  endif()
  if(TPL_ENABLE_SLATE)
    message(STATUS "Disabling SLATE, since it requires MPI support.")
    set(TPL_ENABLE_SLATE OFF)
  endif()
endif()
if(TPL_ENABLE_PTSCOTCH)
  set(TPL_ENABLE_SCOTCH ON)
endif()
if(NOT (STRUMPACK_USE_CUDA OR STRUMPACK_USE_HIP))
  if(TPL_ENABLE_MAGMA)
    message(STATUS "Disabling MAGMA, it requires CUDA or HIP to be enabled.")
    set(TPL_ENABLE_MAGMA OFF)
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
      " proceeding without OpenMP support.")
  endif()
else()
  message(STATUS "OpenMP support was not enabled.")
endif()

if(STRUMPACK_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(STRUMPACK_USE_HIP)
  if(STRUMPACK_USE_CUDA)
    message(WARNING "A CUDA compiler was found,"
      " proceeding with CUDA support instead of HIP.")
    set(STRUMPACK_USE_HIP OFF)
  else()
    find_package(HIP MODULE)
    if(HIP_FOUND)
      set(STRUMPACK_HIP_PLATFORM_HCC ON)
      find_package(hipblas REQUIRED)
      find_package(hipsparse REQUIRED)
      find_package(rocsolver REQUIRED)
      find_package(rocblas REQUIRED)
      find_package(rocprim REQUIRED)
      find_package(rocthrust REQUIRED)
    else()
      message(WARNING "HIP compiler not found,"
        " proceeding without HIP support.")
      set(STRUMPACK_USE_HIP OFF)
    endif()
  endif()
endif()

if(STRUMPACK_USE_SYCL)
  message(STATUS "SYCL/DPC++ enabled, looking for MKL")
  list(APPEND CMAKE_PREFIX_PATH $ENV{MKLROOT} $ENV{TBBROOT})
  # set(MKL_LINK dynamic)
  # set(MKL_LINK static)
  # set(MKL_ARCH intel64)
  # set(MKL_INTERFACE lp64)   ## this doesn't work, only ilp64 works
  set(STRUMPACK_USE_BLAS64 ON)
  find_package(MKL REQUIRED)
endif()


if(TPL_ENABLE_MATLAB)
  find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)
  if(Matlab_FOUND)
    message("Found MATLAB, make sure to use BLAS/LAPACK with 64bit interface, such as Intel MKL ILP64")
    set(STRUMPACK_USE_BLAS64 ON)
    option(STRUMPACK_USE_MATLAB "" ON)
  endif()
endif()


if(STRUMPACK_USE_BLAS64)
  # BLA_SIZEOF_INTEGER is new in CMake 3.22
  set(BLA_SIZEOF_INTEGER 8)
  set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
    ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
    PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:Fortran>:$<$<OR:$<Fortran_COMPILER_ID:Clang>,$<Fortran_COMPILER_ID:AppleClang>,$<Fortran_COMPILER_ID:GNU>>: -fdefault-integer-8>> $<$<COMPILE_LANGUAGE:Fortran>:$<$<Fortran_COMPILER_ID:Intel>: -i8>>")
endif()


if(TPL_ENABLE_MAGMA)
  list(APPEND CMAKE_PREFIX_PATH
    ${TPL_MAGMA_PREFIX} $ENV{MAGMA_DIR} $ENV{MAGMA_ROOT})
  find_package(MAGMA)
  if(MAGMA_FOUND)
    option(STRUMPACK_USE_MAGMA "" ON)
  endif()
endif()


if(DEFINED TPL_BLAS_LIBRARIES)
  # xSDK policies require the user be able to specify BLAS libraries
  # through the TPL_BLAS_LIBRARIES variable. If these do not work,
  # then an error should be generated, and we shouldn't just continue.
  set(BLAS_LIBRARIES ${TPL_BLAS_LIBRARIES})
  set(BLAS_FOUND TRUE)
  set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/blas_compile_test)
  if(TARGET OpenMP::OpenMP_Fortran)
    try_compile(STRUMPACK_TPL_BLAS_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
      LINK_LIBRARIES ${TPL_BLAS_LIBRARIES} OpenMP::OpenMP_Fortran)
  else()
    try_compile(STRUMPACK_TPL_BLAS_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
      LINK_LIBRARIES ${TPL_BLAS_LIBRARIES})
  endif()
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
    if(TARGET OpenMP::OpenMP_Fortran)
      try_compile(STRUMPACK_TPL_BLAS_WITH_THREADS_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
        LINK_LIBRARIES ${TPL_BLAS_LIBRARIES}
        Threads::Threads OpenMP::OpenMP_Fortran)
    else()
      try_compile(STRUMPACK_TPL_BLAS_WITH_THREADS_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
        LINK_LIBRARIES ${TPL_BLAS_LIBRARIES} Threads::Threads)
    endif()
    if(STRUMPACK_TPL_BLAS_WITH_THREADS_WORKS)
      message(STATUS
        "Using BLAS (${TPL_BLAS_LIBRARIES}), with additional threading library.")
    else()
      message(FATAL_ERROR
        "BLAS libraries defined in TPL_BLAS_LIBRARIES (${TPL_BLAS_LIBRARIES}) cannot be used. "
        "If your BLAS library links to OpenMP, enable OpenMP with -DSTRUMPACK_USE_OPENMP=ON. "
        "If that doesn't work, try running CMake with --debug-trycompile and check "
        "the output in build/blas_compile_test/CMakeFiles/CMakeTmp/ .")
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
    find_package(Threads)
    if(TARGET OpenMP::OpenMP_Fortran)
      try_compile(STRUMPACK_BLAS_JUST_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
        LINK_LIBRARIES Threads::Threads OpenMP::OpenMP_Fortran)
    else()
      try_compile(STRUMPACK_BLAS_JUST_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_blas.f90
        LINK_LIBRARIES Threads::Threads)
    endif()
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
  set(LAPACK_LIBRARIES ${TPL_LAPACK_LIBRARIES})
  set(LAPACK_FOUND TRUE)
  set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack_compile_test)
  if(TARGET OpenMP::OpenMP_Fortran)
    try_compile(STRUMPACK_TPL_LAPACK_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
      LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES} ${BLAS_LIBRARIES} OpenMP::OpenMP_Fortran)
  else()
    try_compile(STRUMPACK_TPL_LAPACK_WORKS ${_SCRATCH_DIR}
      SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
      LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
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
    if(TARGET OpenMP::OpenMP_Fortran)
      try_compile(STRUMPACK_TPL_LAPACK_WITH_THREADS_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
        LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
        Threads::Threads OpenMP::OpenMP_Fortran)
    else()
      try_compile(STRUMPACK_TPL_LAPACK_WITH_THREADS_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
        LINK_LIBRARIES ${TPL_LAPACK_LIBRARIES} ${BLAS_LIBRARIES} Threads::Threads)
    endif()
    if(STRUMPACK_TPL_LAPACK_WITH_THREADS_WORKS)
      message(STATUS
        "Using LAPACK (${TPL_LAPACK_LIBRARIES}), "
        "with additional threading library.")
    else()
      message(FATAL_ERROR
        "LAPACK libraries defined in TPL_LAPACK_LIBRARIES (${TPL_LAPACK_LIBRARIES}) cannot be used. "
        "If your LAPACK library links to OpenMP, enable OpenMP with -DSTRUMPACK_USE_OPENMP=ON. "
        "If that doesn't work, try running CMake with --debug-trycompile and check "
        "the output in build/lapack_compile_test/CMakeFiles/CMakeTmp/")
    endif()
  endif()
else()
  if(DEFINED TPL_BLAS_LIBRARIES)
    message(WARNING
      "TPL_BLAS_LIBRARIES was specified. "
      "Consider specifying TPL_LAPACK_LIBRARIES as well to ensure "
      "compatible BLAS and LAPACK libraries.")
  endif()
  find_package(LAPACK)
  if(NOT LAPACK_FOUND)
    # LAPACK was not specified in TPL_LAPACK_LIBRARIES, and not found
    # elsewhere. We will still try to compile a LAPACK example,
    # because perhaps LAPACK is automatically linked by the compiler
    # wrapper, as is the case for instance on Cray machines.
    message(STATUS
      "A LAPACK library could not be found on the system, "
      "checking if LAPACK is implicitly linked by the compiler(wrapper).")
    set(_SCRATCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack_compile_test)
    find_package(Threads)
    if(TARGET OpenMP::OpenMP_Fortran)
      try_compile(STRUMPACK_LAPACK_JUST_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
        LINK_LIBRARIES Threads::Threads OpenMP::OpenMP_Fortran)
    else()
      try_compile(STRUMPACK_LAPACK_JUST_WORKS ${_SCRATCH_DIR}
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_lapack.f90
        LINK_LIBRARIES Threads::Threads)
    endif()
    if(NOT STRUMPACK_LAPACK_JUST_WORKS)
      message(FATAL_ERROR
        "Could not find a LAPACK library.\n"
         " You can specify a LAPACK library using -DTPL_LAPACK_LIBRARIES=\"...\"")
    endif()
  endif()
endif()


# check if getopt.h is available
check_cxx_source_compiles("#include <getopt.h>
int main(int argc, char* argv[]) { getopt_long(argc, argv, \"\", 0, 0); }"
  STRUMPACK_USE_GETOPT)


if(STRUMPACK_USE_MPI)
  # the default blocksize when using ScaLAPACK,
  # might be changed when using SLATE, see below
  set(STRUMPACK_PBLAS_BLOCKSIZE "32")
  if(DEFINED TPL_SCALAPACK_LIBRARIES)
    set(SCALAPACK_FOUND TRUE)
    set(SCALAPACK_LIBRARIES ${TPL_SCALAPACK_LIBRARIES})
    message(STATUS
      "Using ScaLAPACK from TPL_SCALAPACK_LIBRARIES"
      " (${TPL_SCALAPACK_LIBRARIES})")
    ### this check doesn't work: needs MPI, BLAS, LAPACK, threads, libm, ..?
    # cmake_push_check_state()
    # set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)
    # if(BLAS_FOUND)
    #   list(APPEND CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    # endif()
    # if(LAPACK_FOUND)
    #   list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    # endif()
    # check_library_exists("${TPL_SCALAPACK_LIBRARIES}" pdgemm_ "" TPL_SCALAPACK_WORKS_U)
    # check_library_exists("${TPL_SCALAPACK_LIBRARIES}" pdgemm  "" TPL_SCALAPACK_WORKS_NOU)
    # cmake_pop_check_state()
    # if(TPL_SCALAPACK_WORKS_U OR TPL_SCALAPACK_WORKS_NOU)
    #   message(STATUS
    #     "Using ScaLAPACK from TPL_SCALAPACK_LIBRARIES"
    #     " (${TPL_SCALAPACK_LIBRARIES})")
    # else()
    #   message(FATAL_ERROR
    #     "Cannot use TPL_SCALAPACK_LIBRARIES (${TPL_SCALAPACK_LIBRARIES})")
    # endif()
  else()
    find_package(SCALAPACK)
    if(NOT SCALAPACK_FOUND)
      # SCALAPACK was not specified in TPL_SCALAPACK_LIBRARIES, and
      # not found elsewhere. We will still try to compile a SCALAPACK
      # example, because perhaps SCALAPACK is automatically linked by
      # the compiler wrapper, as is the case for instance on Cray
      # machines.
      message(STATUS
        "A SCALAPACK library could not be found on the system, "
        "checking if SCALAPACK is implicitly linked by the compiler(wrapper).")
      check_library_exists("" pdgemm_ "" SCALAPACK_JUST_WORKS_U)
      check_library_exists("" pdgemm  "" SCALAPACK_JUST_WORKS_NOU)
      if(NOT (SCALAPACK_JUST_WORKS_U OR SCALAPACK_JUST_WORKS_NOU))
        message(FATAL_ERROR
          "Could not find a SCALAPACK library.\n"
          " You can specify a SCALAPACK library using -DTPL_SCALAPACK_LIBRARIES=\"...\"")
      endif()
    else()
      set(SCALAPACK_LIBRARIES scalapack)
      message(STATUS "Found SCALAPACK " ${SCALAPACK_LIBRARIES})
    endif()
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
    if(SCOTCH_USES_PTHREADS)
      message(STATUS "Scotch was compiled with pthread support, "
        "looking for pthreads.")
      find_package(Threads)
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
  list(APPEND CMAKE_PREFIX_PATH ${TPL_ZFP_PREFIX}
    $ENV{ZFP_ROOT} $ENV{ZFP_DIR})
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

if(TPL_ENABLE_SLATE)
  list(APPEND CMAKE_PREFIX_PATH ${TPL_SLATE_PREFIX}
    $ENV{slate_DIR} $ENV{slate_ROOT}
    $ENV{SLATE_DIR} $ENV{SLATE_ROOT})
  find_package(blaspp)
  find_package(lapackpp)
  find_package(slate)
  if(slate_FOUND)
    message(STATUS "Found SLATE")
    # option(STRUMPACK_USE_SLATE_LAPACK "" ON)
    option(STRUMPACK_USE_SLATE "" ON)
    option(STRUMPACK_USE_SLATE_SCALAPACK "" ON)
    set(STRUMPACK_PBLAS_BLOCKSIZE "512")
  else()
    message(WARNING "SLATE was not found."
      " SLATE is required for GPU support in the distributed memory code.")
  endif()
endif()

if(TPL_ENABLE_COMBBLAS)
  list(APPEND CMAKE_PREFIX_PATH ${TPL_COMBBLAS_PREFIX}
    $ENV{COMBBLAS_DIR} $ENV{COMBBLAS_ROOT}
    $ENV{CombBLAS_DIR} $ENV{CombBLAS_ROOT})
  find_package(CombBLAS)
  if(CombBLAS_FOUND)
    message(STATUS "Found CombBLAS")
    option(STRUMPACK_USE_COMBBLAS "" ON)
  else()
    message(STATUS "CombBLAS not found")
  endif()
endif()

if(STRUMPACK_USE_HIP)
  set_source_files_properties(
    src/sparse/fronts/FrontalMatrixHIP.hip.cpp
    src/dense/HIPWrapper.hip.cpp
    PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  if(BUILD_SHARED_LIBS)
    set(STATIC_OR_SHARED SHARED)
  else()
    set(STATIC_OR_SHARED STATIC)
  endif()
  hip_add_library(
    strumpack
    src/sparse/fronts/FrontalMatrixHIP.hip.cpp
    src/dense/HIPWrapper.hip.cpp
    ${STATIC_OR_SHARED})
else()
  add_library(strumpack "")
endif()

# this is to fix a link issue for the fortran example when compiling
# with clang
set_property(TARGET strumpack PROPERTY POSITION_INDEPENDENT_CODE ON)


# see https://github.com/pghysels/STRUMPACK/issues/82
# TODO check if 32bit, check compiler??
set(ATOMIC_TEST_SOURCE "#include <atomic>
      int main() { std::atomic<int64_t> i(0); i++; return 0; }")
check_cxx_source_compiles("${ATOMIC_TEST_SOURCE}" ATOMIC_INT64_IS_BUILTIN)
if(NOT ATOMIC_INT64_IS_BUILTIN)
  set(CMAKE_REQUIRED_LIBRARIES atomic)
  check_cxx_source_compiles(
    "${ATOMIC_TEST_SOURCE}" ATOMIC_INT64_REQUIRES_LIBATOMIC)
  if (ATOMIC_INT64_REQUIRES_LIBATOMIC)
    target_link_libraries(strumpack PRIVATE atomic)
  endif()
  unset(CMAKE_REQUIRED_LIBRARIES)
endif()


add_subdirectory(src)

target_sources(strumpack
  PRIVATE ${PROJECT_BINARY_DIR}/StrumpackFortranCInterface.h)

target_compile_features(strumpack PUBLIC cxx_std_14)
set_target_properties(strumpack PROPERTIES CXX_EXTENSIONS OFF)

target_compile_options(strumpack PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>: -Wall>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>: /W4>>)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-overloaded-virtual
  COMPILER_SUPPORTS_NOOVERLOAD_WARNING)
if(COMPILER_SUPPORTS_NOOVERLOAD_WARNING)
  target_compile_options(strumpack PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-overloaded-virtual>)
endif()

if(STRUMPACK_USE_BLAS64)
  target_compile_options(strumpack PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:$<$<OR:$<Fortran_COMPILER_ID:Clang>,$<Fortran_COMPILER_ID:AppleClang>,$<Fortran_COMPILER_ID:GNU>>: -fdefault-integer-8>>
    $<$<COMPILE_LANGUAGE:Fortran>:$<$<Fortran_COMPILER_ID:Intel>: -i8>>)
endif()


if(STRUMPACK_USE_MPI)
  target_link_libraries(strumpack PUBLIC
    MPI::MPI_CXX MPI::MPI_C ${MPI_Fortran_LIBRARIES})
  ## adding the MPI::MPI_Fortran target risks adding unwanted compile flags
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
endif()

if(NOT STRUMPACK_USE_OPENMP)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Wno-unknown-pragmas
    COMPILER_SUPPORTS_NOPRAGMA_WARNING)
  if(COMPILER_SUPPORTS_NOPRAGMA_WARNING)
    target_compile_options(strumpack PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
  endif()
endif()


if(STRUMPACK_USE_MPI)
  if(SCALAPACK_FOUND)
    target_link_libraries(strumpack PUBLIC ${SCALAPACK_LIBRARIES})
  endif()
endif()
if(LAPACK_FOUND)
  target_link_libraries(strumpack PUBLIC ${LAPACK_LIBRARIES})
endif()
if(BLAS_FOUND)
  target_link_libraries(strumpack PUBLIC ${BLAS_LIBRARIES})
endif()

if(MKL_FOUND)
  target_link_libraries(strumpack PUBLIC MKL::MKL)
  # target_link_libraries(strumpack PUBLIC MKL::MKL_DPCPP)
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
  if(SCOTCH_USES_PTHREADS)
    target_link_libraries(strumpack PUBLIC Threads::Threads)
  endif()
endif()
if(PTSCOTCH_FOUND)
  target_link_libraries(strumpack PUBLIC PTSCOTCH::ptscotch)
endif()

if(ZFP_FOUND)
  target_link_libraries(strumpack PUBLIC zfp::zfp)
endif()

if(ButterflyPACK_FOUND)
  target_link_libraries(strumpack PUBLIC
    ButterflyPACK::sbutterflypack
    ButterflyPACK::dbutterflypack
    ButterflyPACK::cbutterflypack
    ButterflyPACK::zbutterflypack)
endif()

if(slate_FOUND)
  target_link_libraries(strumpack PUBLIC slate lapackpp blaspp)
endif()

if(CUDAToolkit_FOUND)
  target_link_libraries(strumpack PUBLIC
    CUDA::cudart CUDA::cusolver CUDA::cublas)
endif()

if(MAGMA_FOUND)
  target_link_libraries(strumpack PUBLIC MAGMA::magma)
endif()

if(hipblas_FOUND)
  target_link_libraries(strumpack PUBLIC
    roc::hipblas roc::rocblas roc::rocsolver roc::hipsparse roc::rocthrust)
endif()

if(CombBLAS_FOUND)
  target_link_libraries(strumpack PUBLIC CombBLAS::CombBLAS)
  target_include_directories(strumpack PRIVATE $ENV{COMBBLASAPP_DIR})
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
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})


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
