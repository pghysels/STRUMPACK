if(TPL_ENABLE_BPACK)
  target_sources(strumpack
    PRIVATE
    HODLRMatrix.cpp
    LRBFMatrix.cpp
    HODLRWrapper.cpp
    HODLRWrapper.hpp
    PUBLIC
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    HODLROptions.hpp)

  install(FILES
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    HODLROptions.hpp
    DESTINATION include/HODLR)
endif()
