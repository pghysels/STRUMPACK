target_sources(strumpack
  PRIVATE
  StrumpackParameters.cpp #PUBLIC
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  StrumpackSparseSolver.hpp)

install(FILES
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  StrumpackSparseSolver.hpp
  DESTINATION include)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
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

install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackConfig.hpp
  DESTINATION include)

# add build folder as include dir for StrumpackConfig.hpp and
# StrumpackFortranCInterface.h, also add the current src/ folder as
# include dir
target_include_directories(strumpack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)


add_subdirectory(misc)
add_subdirectory(sparse)
add_subdirectory(kernel)
add_subdirectory(clustering)
add_subdirectory(python)
add_subdirectory(dense)
add_subdirectory(HSS)
add_subdirectory(HODLR)
add_subdirectory(BLR)
