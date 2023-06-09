target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HBSMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HBSMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HBSMatrix.compress.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HBSOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HBSOptions.cpp)

install(FILES
  HBSMatrix.hpp
  HBSOptions.hpp
  DESTINATION include/HBS)
