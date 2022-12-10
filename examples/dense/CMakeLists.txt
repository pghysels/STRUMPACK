add_executable(KernelRegression   EXCLUDE_FROM_ALL KernelRegression.cpp)
add_executable(testStructured     EXCLUDE_FROM_ALL testStructured.cpp)
add_executable(dstructured        EXCLUDE_FROM_ALL dstructured.c)
add_executable(test2DSIE          EXCLUDE_FROM_ALL test2DSIE.cpp)
add_executable(testCovariance     EXCLUDE_FROM_ALL testCovariance.cpp)
add_executable(testQC             EXCLUDE_FROM_ALL testQC.cpp)
add_executable(testFront          EXCLUDE_FROM_ALL testFront.cpp)
add_executable(fstructured        EXCLUDE_FROM_ALL fstructured.f90)
set_target_properties(fstructured PROPERTIES LINKER_LANGUAGE Fortran)

target_link_libraries(KernelRegression strumpack)
target_link_libraries(testStructured strumpack)
target_link_libraries(dstructured strumpack)
target_link_libraries(fstructured strumpack)
target_link_libraries(test2DSIE strumpack)
target_link_libraries(testCovariance strumpack)
target_link_libraries(testQC strumpack)
target_link_libraries(testFront strumpack)

target_include_directories(testCovariance PRIVATE /home/pieterg/LBL/summer2022/random/mfem/mfem-4.4)
target_link_libraries(testCovariance /home/pieterg/LBL/summer2022/random/mfem/mfem-4.4/libmfem.a)

add_dependencies(examples
  KernelRegression
  testStructured
  dstructured
  fstructured
  test2DSIE
  testCovariance
  testQC
  testFront)

if(STRUMPACK_USE_MPI)
  add_executable(KernelRegressionMPI   EXCLUDE_FROM_ALL
    KernelRegressionMPI.cpp)
  add_executable(testBLRMPI            EXCLUDE_FROM_ALL
    testBLRMPI.cpp)
  add_executable(testStructuredMPI     EXCLUDE_FROM_ALL
    testStructuredMPI.cpp)
  add_executable(dstructured_mpi       EXCLUDE_FROM_ALL
    dstructured_mpi.c)

  target_link_libraries(KernelRegressionMPI strumpack)
  target_link_libraries(testBLRMPI strumpack)
  target_link_libraries(testStructuredMPI strumpack)
  target_link_libraries(dstructured_mpi strumpack)

  add_dependencies(examples
    KernelRegressionMPI
    testBLRMPI
    testStructuredMPI
    dstructured_mpi)

  if(STRUMPACK_USE_BPACK)
    add_executable(testHODLR             EXCLUDE_FROM_ALL
      testHODLR.cpp)
    add_executable(test_HODLR_HODBF      EXCLUDE_FROM_ALL
      test_HODLR_HODBF.cpp)
    target_link_libraries(testHODLR strumpack)
    target_link_libraries(test_HODLR_HODBF strumpack)

    add_dependencies(examples testHODLR test_HODLR_HODBF)
  endif()
endif()
