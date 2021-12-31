target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/PPt.hpp
  ${CMAKE_CURRENT_LIST_DIR}/Graph.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BLASLAPACKWrapper.hpp
  ${CMAKE_CURRENT_LIST_DIR}/GeometricReordering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/GeometricReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MatrixReordering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RCMReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/ANDSparspak.hpp
  ${CMAKE_CURRENT_LIST_DIR}/ANDSparspak.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ScotchReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MatrixReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MetisReordering.hpp)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/GeometricReorderingMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/GeometricReorderingMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ParMetisReordering.hpp
    ${CMAKE_CURRENT_LIST_DIR}/PTScotchReordering.hpp
    ${CMAKE_CURRENT_LIST_DIR}/MatrixReorderingMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MatrixReorderingMPI.hpp)
endif()

add_subdirectory(rcm)
add_subdirectory(minimum_degree)
add_subdirectory(spectral)
