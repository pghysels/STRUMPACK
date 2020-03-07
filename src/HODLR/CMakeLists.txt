target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HODLROptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HODLROptions.cpp)

install(FILES
  HODLROptions.hpp
  DESTINATION include/HODLR)


if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LRBFMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/LRBFMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.cpp)

  install(FILES
    HODLRMatrix.hpp
    LRBFMatrix.hpp
    DESTINATION include/HODLR)
endif()
