target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack_dense.f90)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DESTINATION include FILES_MATCHING PATTERN *.mod)
