target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackParameters.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackOptions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackParameters.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolver.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverC.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolver.h)

install(FILES
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  StrumpackSparseSolver.hpp
  DESTINATION include)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMPI.hpp
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMPI.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMPIDist.hpp
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMPIDist.cpp)

  install(FILES
    StrumpackSparseSolverMPIDist.hpp
    StrumpackSparseSolverMPI.hpp
    DESTINATION include)
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
