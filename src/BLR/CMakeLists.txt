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

  install(FILES
    BLRMatrixMPI.hpp
    BLRExtendAdd.hpp
    DESTINATION include/BLR)
endif()
