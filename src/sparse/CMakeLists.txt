target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/MC64ad.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MC64ad.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CompressedSparseMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CompressedSparseMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRGraph.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRGraph.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/CSRMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/EliminationTree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/EliminationTree.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ETree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SeparatorTree.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SeparatorTree.cpp)

install(FILES
  CompressedSparseMatrix.hpp
  CSRMatrix.hpp
  CSRGraph.hpp
  DESTINATION include/sparse)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CSRMatrixMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/CSRMatrixMPI.cpp
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
    DESTINATION include/sparse)
endif()

# TODO AWPMCombBLAS.hpp

add_subdirectory(fronts)
add_subdirectory(ordering)
add_subdirectory(MUMPS)
