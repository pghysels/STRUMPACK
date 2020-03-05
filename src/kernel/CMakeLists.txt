target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Kernel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Kernel.hpp
  ${CMAKE_CURRENT_LIST_DIR}/KernelRegression.hpp
  ${CMAKE_CURRENT_LIST_DIR}/Kernel.h
  ${CMAKE_CURRENT_LIST_DIR}/Metrics.hpp)

install(FILES
  Kernel.hpp
  KernelRegression.hpp
  Kernel.h
  Metrics.hpp
  DESTINATION include/kernel)
