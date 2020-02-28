target_sources(strumpack
  PRIVATE
  Kernel.cpp
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
