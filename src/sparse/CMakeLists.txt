target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/strumpack_mc64ad.c
  ${CMAKE_CURRENT_LIST_DIR}/CompressedSparseMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRGraph.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/EliminationTree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/EliminationTree.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ETree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SeparatorTree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SeparatorTree.cpp)

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
    ${CMAKE_CURRENT_LIST_DIR}/CSRMatrixMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ETree.hpp
    ${CMAKE_CURRENT_LIST_DIR}/EliminationTreeMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/EliminationTreeMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/EliminationTreeMPIDist.hpp
    ${CMAKE_CURRENT_LIST_DIR}/EliminationTreeMPIDist.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mumps_symqamd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/PropMapSparseMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/PropMapSparseMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Redistribute.hpp)

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
