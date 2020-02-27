target_sources(strumpack
  PRIVATE
  GeometricReordering.cpp
  GeometricReordering.hpp
  MatrixReordering.cpp
  RCMReordering.hpp
  ScotchReordering.hpp
  PUBLIC
  MatrixReordering.hpp
  MetisReordering.hpp)

install(FILES
  MatrixReordering.hpp
  MetisReordering.hpp
  DESTINATION include/sparse/ordering)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    GeometricReorderingMPI.cpp
    GeometricReorderingMPI.hpp
    ParMetisReordering.hpp  # TODO split into hpp/cpp
    PTScotchReordering.hpp
    MatrixReorderingMPI.cpp
    PUBLIC
    MatrixReorderingMPI.hpp)

  install(FILES
    MatrixReorderingMPI.hpp
    DESTINATION include/sparse/ordering)
endif()

add_subdirectory(rcm)
