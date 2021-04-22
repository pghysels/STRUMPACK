target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ClusterTree.hpp)

install(FILES
  StructuredMatrix.hpp
  ClusterTree.hpp
  DESTINATION include/structured)
