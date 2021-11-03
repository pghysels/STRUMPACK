target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack.f90)

if(FortranCInterface_MODULE_CASE STREQUAL "LOWER")
  # message("Lower ${FortranCInterface_MODULE_CASE}")
  install(FILES ${CMAKE_BINARY_DIR}/strumpack.mod
    DESTINATION include)
else()
  message("Upper ${FortranCInterface_MODULE_CASE}")
  install(FILES ${CMAKE_BINARY_DIR}/STRUMPACK.mod
    DESTINATION include)
endif()
