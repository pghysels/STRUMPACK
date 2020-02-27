target_sources(strumpack
  PUBLIC
  BLRMatrix.hpp
  BLRMatrixMPI.hpp
  BLROptions.hpp
  BLRTileBLAS.hpp
  BLRTile.hpp
  DenseTile.hpp
  LRTile.hpp)


install(FILES
  BLRMatrix.hpp
  BLRMatrixMPI.hpp
  BLROptions.hpp
  BLRTileBLAS.hpp
  BLRTile.hpp
  DenseTile.hpp
  LRTile.hpp
  DESTINATION include/BLR)
