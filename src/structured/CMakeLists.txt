target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/StructuredOptions.cpp)

install(FILES
  StructuredMatrix.hpp
  DESTINATION include/structured)


# if(STRUMPACK_USE_MPI)
#   target_sources(strumpack
#     PRIVATE
#     ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrixMPI.hpp
#     ${CMAKE_CURRENT_LIST_DIR}/StructuredMatrix.cpp)

#   install(FILES
#     StructuredMatrixMPI.hpp
#     DESTINATION include/structured)
# endif()
