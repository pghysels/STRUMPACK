target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/BiCGStab.cpp
  ${CMAKE_CURRENT_LIST_DIR}/GMRes.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IterativeRefinement.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IterativeSolvers.hpp)

install(FILES
  IterativeSolvers.hpp
  DESTINATION include/iterative)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/GMResMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/BiCGStabMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/IterativeRefinementMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/IterativeSolversMPI.hpp)

  install(FILES
    IterativeSolversMPI.hpp
    DESTINATION include/iterative)
endif()
