target_sources(strumpack
  PRIVATE
  strumpack_mc64ad.c
  CompressedSparseMatrix.hpp
  CSRGraph.hpp
  CSRMatrix.hpp
  EliminationTree.hpp
  ETree.hpp
  SeparatorTree.hpp)

install(FILES
  CompressedSparseMatrix.hpp
  CSRGraph.hpp
  CSRMatrix.hpp
  EliminationTree.hpp
  ETree.hpp
  SeparatorTree.hpp
  DESTINATION include/sparse)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    CSRMatrixMPI.hpp
    ETree.hpp
    EliminationTreeMPI.hpp
    EliminationTreeMPIDist.hpp
    mumps_symqamd.hpp
    PropMapSparseMatrix.hpp
    Redistribute.hpp)

  install(FILES
    CSRMatrixMPI.hpp
    ETree.hpp
    EliminationTreeMPI.hpp
    EliminationTreeMPIDist.hpp
    mumps_symqamd.hpp
    PropMapSparseMatrix.hpp
    Redistribute.hpp
    DESTINATION include/sparse)
endif()

# TODO AWPMCombBLAS.hpp

add_subdirectory(iterative)
add_subdirectory(fronts)
add_subdirectory(ordering)
add_subdirectory(MUMPS)
