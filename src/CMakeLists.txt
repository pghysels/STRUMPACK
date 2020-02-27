target_sources(strumpack
  PRIVATE
  StrumpackParameters.cpp
  PUBLIC
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  StrumpackSparseSolver.hpp)

install(FILES
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  StrumpackSparseSolver.hpp
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp
  DESTINATION include)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PUBLIC
    StrumpackSparseSolverMPIDist.hpp
    StrumpackSparseSolverMPI.hpp)

  install(FILES
    StrumpackSparseSolverMPIDist.hpp
    StrumpackSparseSolverMPI.hpp
    DESTINATION include)
endif()

if(STRUMPACK_C_INTERFACE)
  target_sources(strumpack
    PRIVATE
    StrumpackSparseSolver.cpp
    StrumpackSparseSolver.h)
endif()

configure_file(StrumpackConfig.hpp.in
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp)

target_include_directories(strumpack PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(misc)
add_subdirectory(sparse)
add_subdirectory(kernel)
add_subdirectory(clustering)
add_subdirectory(python)
add_subdirectory(dense)
add_subdirectory(HSS)
add_subdirectory(HODLR)
add_subdirectory(BLR)
