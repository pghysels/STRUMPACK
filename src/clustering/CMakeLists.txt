target_sources(strumpack
  PRIVATE
  PCAPartitioning.cpp
  CobblePartitioning.cpp
  KMeans.cpp
  KDTree.cpp
  Clustering.hpp
  NeighborSearch.hpp)

install(FILES
  Clustering.hpp
  NeighborSearch.hpp
  DESTINATION include/clustering)
