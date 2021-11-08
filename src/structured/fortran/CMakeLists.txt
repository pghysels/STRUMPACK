target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack_dense.f90)

if(FortranCInterface_MODULE_CASE STREQUAL "LOWER")
  install(FILES ${CMAKE_BINARY_DIR}/strumpack_dense.mod
    DESTINATION include)
else()
  install(FILES ${CMAKE_BINARY_DIR}/STRUMPACK_DENSE.mod
    DESTINATION include)
endif()
