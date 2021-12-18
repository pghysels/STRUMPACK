add_executable(KernelRegression   EXCLUDE_FROM_ALL KernelRegression.cpp)
add_executable(testStructured     EXCLUDE_FROM_ALL testStructured.cpp)
add_executable(dstructured        EXCLUDE_FROM_ALL dstructured.c)
add_executable(fstructured        EXCLUDE_FROM_ALL fstructured.f90)
set_target_properties(fstructured PROPERTIES LINKER_LANGUAGE Fortran)

target_link_libraries(KernelRegression strumpack)
target_link_libraries(testStructured strumpack)
target_link_libraries(dstructured strumpack)
target_link_libraries(fstructured strumpack)

add_dependencies(examples
  KernelRegression
  testStructured
  dstructured
  fstructured)

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
