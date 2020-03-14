target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/PCAPartitioning.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CobblePartitioning.cpp
  ${CMAKE_CURRENT_LIST_DIR}/KMeans.cpp
  ${CMAKE_CURRENT_LIST_DIR}/KDTree.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Clustering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/NeighborSearch.hpp
  ${CMAKE_CURRENT_LIST_DIR}/NeighborSearch.cpp)

install(FILES
  Clustering.hpp
  NeighborSearch.hpp
  DESTINATION include/clustering)
