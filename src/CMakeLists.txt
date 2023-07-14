target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackParameters.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackOptions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackParameters.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SparseSolverBase.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SparseSolverBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolver.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SparseSolver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SparseSolverMixedPrecision.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMixedPrecision.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverC.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolver.h)

install(FILES
  StrumpackOptions.hpp
  StrumpackParameters.hpp
  SparseSolverBase.hpp
  StrumpackSparseSolver.hpp
  StrumpackSparseSolver.h
  StrumpackConfig.hpp
  DESTINATION include)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/SparseSolverMPIDist.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMPIDist.hpp
    ${CMAKE_CURRENT_LIST_DIR}/SparseSolverMixedPrecisionMPIDist.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StrumpackSparseSolverMixedPrecisionMPIDist.hpp)

  install(FILES
    StrumpackSparseSolverMPIDist.hpp
    StrumpackSparseSolverMixedPrecisionMPIDist.hpp
    DESTINATION include)
endif()


configure_file(StrumpackConfig.h.in
  ${PROJECT_BINARY_DIR}/StrumpackConfig.h)

install(FILES
  ${PROJECT_BINARY_DIR}/StrumpackConfig.h
  DESTINATION include)

# add build folder as include dir for StrumpackConfig.h and
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
add_subdirectory(fortran)
add_subdirectory(structured)
add_subdirectory(iterative)
if(STRUMPACK_USE_MATLAB)
  add_subdirectory(matlab)
endif()
