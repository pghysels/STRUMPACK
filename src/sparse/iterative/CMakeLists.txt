target_sources(strumpack
  PRIVATE
  BiCGStab.cpp
  GMRes.cpp
  IterativeRefinement.cpp
  PUBLIC
  IterativeSolvers.hpp)

install(FILES
  IterativeSolvers.hpp
  DESTINATION include/sparse/iterative)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    GMResMPI.cpp
    BiCGStabMPI.cpp
    IterativeRefinementMPI.cpp
    PUBLIC
    IterativeSolversMPI.hpp)

  install(FILES
    IterativeSolversMPI.hpp
    DESTINATION include/sparse/iterative)
endif()
