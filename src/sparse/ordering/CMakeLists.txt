target_sources(strumpack
  PRIVATE
  GeometricReordering.cpp
  GeometricReordering.hpp
  MatrixReordering.cpp
  RCMReordering.hpp
  ScotchReordering.hpp
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
    ParMetisReordering.hpp
    PTScotchReordering.hpp
    MatrixReorderingMPI.cpp
    MatrixReorderingMPI.hpp)

  install(FILES
    MatrixReorderingMPI.hpp
    DESTINATION include/sparse/ordering)
endif()

add_subdirectory(rcm)
