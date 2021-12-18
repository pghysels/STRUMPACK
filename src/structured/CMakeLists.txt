target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ClusterTree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrixC.cpp)

install(FILES
  StructuredMatrix.hpp
  StructuredMatrix.h
  ClusterTree.hpp
  StructuredOptions.hpp
  DESTINATION include/structured)

add_subdirectory(fortran)
