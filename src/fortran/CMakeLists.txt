target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/strumpack.f90)

# set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran)

install(FILES
  ${CMAKE_BINARY_DIR}/strumpack.mod
  DESTINATION include)
