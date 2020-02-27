target_sources(strumpack
  PRIVATE
  Kernel.cpp
  PUBLIC
  Kernel.hpp
  KernelRegression.hpp
  Kernel.h
  Metrics.hpp)

install(FILES
  Kernel.hpp
  KernelRegression.hpp
  Kernel.h
  Metrics.hpp
  DESTINATION include/kernel)
