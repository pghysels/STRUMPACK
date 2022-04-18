target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/NDOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SpectralStats.hpp
  ${CMAKE_CURRENT_LIST_DIR}/SpectralReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/LanczosFiedlerSolver.hpp)

install(FILES
  NDOptions.hpp
  SpectralStats.hpp
  SpectralReordering.hpp
  LanczosFiedlerSolver.hpp
  DESTINATION include/sparse/ordering/spectral)
