target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack.f90)

# This is really a problem. Using a wildcard to install the mod makes
# CMake install a bunch of empty directories, which leads to an
# infinite recursion when building in the src tree, which we do for
# testing.
# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   DESTINATION include FILES #_MATCHING PATTERN
#   strumpack.mod STRUMPACK.mod STRUMPACK.MOD)

install(FILES
  ${CMAKE_BINARY_DIR}/strumpack.mod
  ${CMAKE_BINARY_DIR}/STRUMPACK.mod
  ${CMAKE_BINARY_DIR}/STRUMPACK.MOD
  DESTINATION include OPTIONAL)
