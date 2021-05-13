target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/strumpack_dense.f90)

install(FILES
  ${CMAKE_BINARY_DIR}/strumpack_dense.mod
  DESTINATION include)
