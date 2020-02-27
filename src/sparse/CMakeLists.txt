target_sources(strumpack
  PRIVATE
  strumpack_mc64ad.c
  PUBLIC
  CompressedSparseMatrix.hpp
  CSRGraph.hpp
  CSRMatrix.hpp
  EliminationTree.hpp)

install(FILES
  CompressedSparseMatrix.hpp
  CSRGraph.hpp
  CSRMatrix.hpp
  EliminationTree.hpp
  DESTINATION include/sparse)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PUBLIC
    CSRMatrixMPI.hpp
    ETree.hpp
    EliminationTreeMPI.hpp
    EliminationTreeMPIDist.hpp
    mumps_symqamd.hpp
    PropMapSparseMatrix.hpp
    Redistribute.hpp
    SeparatorTree.hpp)

  install(FILES
    CSRMatrixMPI.hpp
    ETree.hpp
    EliminationTreeMPI.hpp
    EliminationTreeMPIDist.hpp
    mumps_symqamd.hpp
    PropMapSparseMatrix.hpp
    Redistribute.hpp
    SeparatorTree.hpp
    DESTINATION include/sparse)
endif()

# TODO AWPMCombBLAS.hpp

add_subdirectory(iterative)
add_subdirectory(fronts)
add_subdirectory(ordering)
add_subdirectory(MUMPS)
