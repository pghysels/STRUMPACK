target_sources(strumpack
  PRIVATE
  HSSMatrixBase.cpp
  HSSMatrix.cpp
  HSSMatrix.apply.hpp
  HSSMatrix.compress.hpp
  HSSMatrix.compress_kernel.hpp
  HSSMatrix.compress_stable.hpp
  HSSMatrix.extract.hpp
  HSSMatrix.factor.hpp
  HSSMatrix.Schur.hpp
  HSSMatrix.solve.hpp
  HSSMatrix.hpp
  HSSBasisID.hpp
  HSSExtra.hpp
  HSSMatrixBase.hpp
  HSSPartitionTree.hpp
  HSSOptions.hpp)

install(FILES
  HSSMatrix.hpp
  HSSBasisID.hpp
  HSSExtra.hpp
  HSSMatrixBase.hpp
  HSSPartitionTree.hpp
  HSSOptions.hpp
  DESTINATION include/HSS)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    HSSMatrixMPI.cpp
    BlockCyclic2BlockRow.hpp
    HSSMatrixMPI.apply.hpp
    HSSMatrixMPI.compress.hpp
    HSSMatrixMPI.compress_kernel.hpp
    HSSMatrixMPI.compress_stable.hpp
    HSSMatrixMPI.extract_blocks.hpp
    HSSMatrixMPI.extract.hpp
    HSSMatrixMPI.factor.hpp
    HSSMatrixMPI.Schur.hpp
    HSSMatrixMPI.solve.hpp
    HSSMatrixMPI.hpp
    HSSExtraMPI.hpp
    HSSBasisIDMPI.hpp
    DistSamples.hpp
    DistElemMult.hpp)

  install(FILES
    HSSMatrixMPI.hpp
    HSSExtraMPI.hpp
    HSSBasisIDMPI.hpp
    DistSamples.hpp
    DistElemMult.hpp
    DESTINATION include/HSS)
endif()
