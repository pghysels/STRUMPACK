target_sources(strumpack
  PRIVATE
  TaskTimer.cpp
  PUBLIC
  TaskTimer.hpp
  RandomWrapper.hpp
  Tools.hpp)

install(FILES
  TaskTimer.hpp
  MPIWrapper.hpp
  RandomWrapper.hpp
  Tools.hpp
  DESTINATION include/misc)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PUBLIC
    MPIWrapper.hpp)

  install(FILES
    MPIWrapper.hpp
    DESTINATION include/misc)
endif()


