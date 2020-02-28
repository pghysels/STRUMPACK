target_sources(strumpack
  PRIVATE
  BLASLAPACKWrapper.cpp
  BLASLAPACKOpenMPTask.cpp
  DistributedMatrix.hpp
  ACA.hpp
  BACA.hpp
  DenseMatrix.hpp
  BLASLAPACKOpenMPTask.hpp
  BLASLAPACKWrapper.hpp)

install(FILES
  DistributedMatrix.hpp
  ACA.hpp
  BACA.hpp
  DenseMatrix.hpp
  BLASLAPACKOpenMPTask.hpp
  BLASLAPACKWrapper.hpp
  DESTINATION include/dense)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ScaLAPACKWrapper.cpp
    BLACSGrid.hpp
    DistributedVector.hpp
    DistributedMatrix.hpp
    ScaLAPACKWrapper.hpp)

  install(FILES
    BLACSGrid.hpp
    DistributedVector.hpp
    DistributedMatrix.hpp
    ScaLAPACKWrapper.hpp
    DESTINATION include/dense)
endif()


if(TPL_ENABLE_CUBLAS)
  target_sources(strumpack
    PUBLIC
    CUDAWrapper.hpp)

  install(FILES
    CUDAWrapper.hpp
    DESTINATION include/dense)
endif()


add_subdirectory(lapack)
