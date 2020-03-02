target_sources(strumpack
  PRIVATE
  FrontFactory.cpp
  FrontalMatrix.cpp
  FrontalMatrixDense.cpp
  FrontalMatrixDense.hpp
  FrontalMatrixHSS.cpp
  FrontalMatrixHSS.hpp
  FrontalMatrixBLR.hpp
  FrontFactory.hpp
  FrontalMatrix.hpp)

install(FILES
  FrontFactory.hpp
  FrontalMatrix.hpp
  DESTINATION include/sparse/fronts)


if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    FrontalMatrixMPI.cpp
    FrontalMatrixMPI.hpp
    FrontalMatrixDenseMPI.cpp
    FrontalMatrixDenseMPI.hpp
    FrontalMatrixHSSMPI.cpp
    FrontalMatrixHSSMPI.hpp
    FrontalMatrixBLRMPI.hpp
    ExtendAdd.cpp
    ExtendAdd.hpp)

  install(FILES
    FrontalMatrixMPI.hpp
    DESTINATION include/sparse/fronts)
endif()

if(STRUMPACK_USE_CUBLAS)
  target_sources(strumpack
    PRIVATE
    FrontalMatrixCUBLAS.cpp
    FrontalMatrixCUBLAS.hpp)
endif()

if(STRUMPACK_USE_ZFP)
  target_sources(strumpack
    PRIVATE
    FrontalMatrixLossy.cpp
    FrontalMatrixLossy.hpp)
endif()

if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    FrontalMatrixHODLR.cpp
    FrontalMatrixHODLRMPI.cpp)
endif()
