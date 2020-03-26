target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/TaskTimer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TaskTimer.hpp
  ${CMAKE_CURRENT_LIST_DIR}/RandomWrapper.hpp
  ${CMAKE_CURRENT_LIST_DIR}/Triplet.hpp
  ${CMAKE_CURRENT_LIST_DIR}/Triplet.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Tools.hpp)

install(FILES
  TaskTimer.hpp
  RandomWrapper.hpp
  Triplet.hpp
  Tools.hpp
  DESTINATION include/misc)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/MPIWrapper.hpp)

  install(FILES
    MPIWrapper.hpp
    DESTINATION include/misc)
endif()


