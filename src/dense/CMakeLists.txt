target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HIPWrapper.hip
  ${CMAKE_CURRENT_LIST_DIR}/BLASLAPACKWrapper.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BLASLAPACKOpenMPTask.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ACA.hpp
  ${CMAKE_CURRENT_LIST_DIR}/ACA.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BACA.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BACA.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DenseMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/DenseMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BLASLAPACKOpenMPTask.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BLASLAPACKWrapper.hpp
  ${CMAKE_CURRENT_LIST_DIR}/GPUWrapper.hpp)

install(FILES
  DistributedMatrix.hpp
  ACA.hpp
  BACA.hpp
  DenseMatrix.hpp
  BLASLAPACKOpenMPTask.hpp # TODO do not install?
  BLASLAPACKWrapper.hpp  # TODO do not install?
  GPUWrapper.hpp
  DESTINATION include/dense)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ScaLAPACKWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/BLACSGrid.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DistributedVector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DistributedMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DistributedMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ScaLAPACKWrapper.hpp)

  install(FILES
    BLACSGrid.hpp
    DistributedVector.hpp
    DistributedMatrix.hpp
    ScaLAPACKWrapper.hpp
    DESTINATION include/dense)
endif()


if(STRUMPACK_USE_CUDA)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDAWrapper.cu
    ${CMAKE_CURRENT_LIST_DIR}/CUDAWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAWrapper.cu
    ${CMAKE_CURRENT_LIST_DIR}/CUDAWrapper.hpp)
 # install(FILES CUDAWrapper.hpp DESTINATION include/dense)
endif()

if(STRUMPACK_USE_HIP)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HIPWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HIPWrapper.hpp)
 # install(FILES HIPWrapper.hpp DESTINATION include/dense)
endif()

if(STRUMPACK_USE_SYCL)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/SYCLWrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/SYCLWrapper.cpp)
 # install(FILES DPCPPWrapper.hpp DESTINATION include/dense)
endif()

if(STRUMPACK_USE_MAGMA)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/MAGMAWrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/MAGMAWrapper.cpp)
  # install(FILES MAGMAWrapper.hpp DESTINATION include/dense)
endif()

if(STRUMPACK_USE_KBLAS)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/KBLASWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/KBLASWrapper.hpp)
  # install(FILES KBLASWrapper.hpp DESTINATION include/dense)
endif()

add_subdirectory(lapack)
