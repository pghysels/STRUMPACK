if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    HODLRMatrix.cpp
    LRBFMatrix.cpp
    HODLRWrapper.cpp
    HODLRWrapper.hpp
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    HODLROptions.hpp)

  install(FILES
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    HODLROptions.hpp
    DESTINATION include/HODLR)
endif()
