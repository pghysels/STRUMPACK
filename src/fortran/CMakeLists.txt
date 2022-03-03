target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack.f90)

install(DIRECTORY ${CMAKE_BINARY_DIR}/
  DESTINATION include FILES_MATCHING PATTERN *.mod)
