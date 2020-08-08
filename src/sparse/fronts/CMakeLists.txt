target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/FrontFactory.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixDense.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixDense.hpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHSS.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHSS.hpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixBLR.hpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontFactory.hpp
  ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrix.hpp)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixDenseMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixDenseMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHSSMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHSSMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixBLRMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ExtendAdd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ExtendAdd.hpp)
endif()

if(STRUMPACK_USE_CUDA)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixGPU.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixGPU.hpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixCUDA.cu
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixGPUKernels.hpp)
endif()

if(STRUMPACK_USE_ZFP)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixLossy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixLossy.hpp)
endif()

if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHODLR.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FrontalMatrixHODLRMPI.cpp)
endif()
