target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/BLRMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BLRMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BLROptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BLROptions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BLRTileBLAS.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BLRTile.hpp
  ${CMAKE_CURRENT_LIST_DIR}/DenseTile.hpp
  ${CMAKE_CURRENT_LIST_DIR}/DenseTile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LRTile.hpp
  ${CMAKE_CURRENT_LIST_DIR}/LRTile.cpp)

if(STRUMPACK_USE_CUDA OR STRUMPACK_USE_HIP OR STRUMPACK_USE_SYCL)
  target_sources(strumpack PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/BLRMatrix.GPU.cpp
    ${CMAKE_CURRENT_LIST_DIR}/BLRBatch.hpp
    ${CMAKE_CURRENT_LIST_DIR}/BLRBatch.cpp)
endif()


install(FILES
  BLRMatrix.hpp
  BLRMatrixMPI.hpp
  BLROptions.hpp
  BLRTileBLAS.hpp  # TODO don't install these
  BLRTile.hpp      #
  DenseTile.hpp    #
  LRTile.hpp       #
  DESTINATION include/BLR)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/BLRMatrixMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/BLRMatrixMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/BLRExtendAdd.cpp)

  if(STRUMPACK_USE_CUDA OR STRUMPACK_USE_HIP OR STRUMPACK_USE_SYCL)
    target_sources(strumpack
      PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/BLRMatrixMPI.GPU.cpp)
  endif()

  install(FILES
    BLRMatrixMPI.hpp
    BLRExtendAdd.hpp
    DESTINATION include/BLR)
endif()
