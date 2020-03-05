target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/GeometricReordering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/GeometricReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MatrixReordering.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RCMReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/ScotchReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MatrixReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MetisReordering.hpp)

install(FILES
  MatrixReordering.hpp
  MetisReordering.hpp
  DESTINATION include/sparse/ordering)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/GeometricReorderingMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/GeometricReorderingMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ParMetisReordering.hpp
    ${CMAKE_CURRENT_LIST_DIR}/PTScotchReordering.hpp
    ${CMAKE_CURRENT_LIST_DIR}/MatrixReorderingMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MatrixReorderingMPI.hpp)

  install(FILES
    MatrixReorderingMPI.hpp
    DESTINATION include/sparse/ordering)
endif()

add_subdirectory(rcm)
