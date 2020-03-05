if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LRBFMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/LRBFMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLROptions.hpp)

  install(FILES
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    HODLROptions.hpp
    DESTINATION include/HODLR)
endif()
