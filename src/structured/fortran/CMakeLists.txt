target_sources(strumpack
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/strumpack_dense.f90)

# This is really a problem. Using a wildcard to install the mod makes
# CMake install a bunch of empty directories, which leads to an
# infinite recursion when building in the src tree, which we do for
# testing.
# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   DESTINATION include FILES #_MATCHING PATTERN
#   strumpack_dense.mod STRUMPACK_DENSE.mod STRUMPACK_DENSE.MOD)

install(FILES
  ${CMAKE_BINARY_DIR}/strumpack_dense.mod
  ${CMAKE_BINARY_DIR}/STRUMPACK_DENSE.mod
  ${CMAKE_BINARY_DIR}/STRUMPACK_DENSE.MOD
  DESTINATION include OPTIONAL)
