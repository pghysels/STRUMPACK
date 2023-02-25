target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.apply.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.compress.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.compress_kernel.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.compress_stable.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.extract.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.factor.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.Schur.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.solve.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSBasisID.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSExtra.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixBase.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSMatrix.sketch.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HSSOptions.cpp)

install(FILES
  HSSMatrix.hpp
  HSSBasisID.hpp
  HSSExtra.hpp
  HSSMatrixBase.hpp
  HSSOptions.hpp
  DESTINATION include/HSS)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/BlockCyclic2BlockRow.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.apply.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.compress.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.compress_kernel.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.compress_stable.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.extract_blocks.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.extract.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.factor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.Schur.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.solve.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSMatrixMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSExtraMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HSSBasisIDMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DistSamples.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DistElemMult.hpp)

  install(FILES
    HSSMatrixMPI.hpp
    HSSExtraMPI.hpp
    HSSBasisIDMPI.hpp
    DistSamples.hpp
    DistElemMult.hpp
    DESTINATION include/HSS)
endif()
