add_executable(KernelRegression   EXCLUDE_FROM_ALL KernelRegression.cpp)
add_executable(testStructured     EXCLUDE_FROM_ALL testStructured.cpp)

target_link_libraries(KernelRegression strumpack)
target_link_libraries(testStructured strumpack)

add_dependencies(examples
  KernelRegression
  testStructured)

if(STRUMPACK_USE_MPI)
  add_executable(KernelRegressionMPI   EXCLUDE_FROM_ALL
    KernelRegressionMPI.cpp)
  add_executable(testBLRMPI            EXCLUDE_FROM_ALL
    testBLRMPI.cpp)
  add_executable(testStructuredMPI     EXCLUDE_FROM_ALL
    testStructuredMPI.cpp)

  target_link_libraries(KernelRegressionMPI strumpack)
  target_link_libraries(testBLRMPI strumpack)
  target_link_libraries(testStructuredMPI strumpack)

  add_dependencies(examples
    KernelRegressionMPI
    testBLRMPI
    testStructuredMPI)

  if(STRUMPACK_USE_BPACK)
    add_executable(testHODLR EXCLUDE_FROM_ALL testHODLR.cpp)
    target_link_libraries(testHODLR strumpack)

    add_dependencies(examples testHODLR)
  endif()
endif()
