add_custom_target(examples)

add_executable(testPoisson2d    EXCLUDE_FROM_ALL testPoisson2d.cpp)
add_executable(testMMdouble     EXCLUDE_FROM_ALL testMMdouble.cpp)
add_executable(KernelRegression EXCLUDE_FROM_ALL KernelRegression.cpp)
add_executable(testPoisson3d    EXCLUDE_FROM_ALL testPoisson3d.cpp)
add_executable(testPoisson3dMixedPrecision EXCLUDE_FROM_ALL testPoisson3dMixedPrecision.cpp)
add_executable(sexample         EXCLUDE_FROM_ALL sexample.c)
add_executable(dexample         EXCLUDE_FROM_ALL dexample.c)
add_executable(cexample         EXCLUDE_FROM_ALL cexample.c)
add_executable(zexample         EXCLUDE_FROM_ALL zexample.c)
add_executable(fexample         EXCLUDE_FROM_ALL fexample.f90)

target_link_libraries(testPoisson2d strumpack)
target_link_libraries(testMMdouble strumpack)
target_link_libraries(KernelRegression strumpack)
target_link_libraries(testPoisson3d strumpack)
target_link_libraries(testPoisson3dMixedPrecision strumpack)
target_link_libraries(sexample strumpack)
target_link_libraries(dexample strumpack)
target_link_libraries(cexample strumpack)
target_link_libraries(zexample strumpack)
target_link_libraries(fexample strumpack)

add_dependencies(examples
  testPoisson2d
  testMMdouble
  KernelRegression
  testPoisson3d
  testPoisson3dMixedPrecision
  sexample
  dexample
  cexample
  zexample
  fexample)

if(STRUMPACK_USE_MPI)
  add_executable(testHelmholtz         EXCLUDE_FROM_ALL
    testHelmholtz.cpp genmatrix3D_core.f)
  add_executable(testPoisson2dMPIDist  EXCLUDE_FROM_ALL
    testPoisson2dMPIDist.cpp)
  add_executable(KernelRegressionMPI   EXCLUDE_FROM_ALL
    KernelRegressionMPI.cpp)
  add_executable(testMMdoubleMPIDist   EXCLUDE_FROM_ALL
    testMMdoubleMPIDist.cpp)
  add_executable(testPoisson3dMPIDist  EXCLUDE_FROM_ALL
    testPoisson3dMPIDist.cpp)
  add_executable(testBLRMPI            EXCLUDE_FROM_ALL
    testBLRMPI.cpp)

  target_link_libraries(testHelmholtz strumpack)
  target_link_libraries(testPoisson2dMPIDist strumpack)
  target_link_libraries(KernelRegressionMPI strumpack)
  target_link_libraries(testMMdoubleMPIDist strumpack)
  target_link_libraries(testPoisson3dMPIDist strumpack)
  target_link_libraries(testBLRMPI strumpack)

  add_dependencies(examples
    testHelmholtz
    testPoisson2dMPIDist
    KernelRegressionMPI
    testMMdoubleMPIDist
    testPoisson3dMPIDist
    testBLRMPI)

  if(STRUMPACK_USE_BPACK)
    add_executable(testHODLR EXCLUDE_FROM_ALL testHODLR.cpp)
    target_link_libraries(testHODLR strumpack)

    add_dependencies(examples testHODLR)
  endif()
endif()
