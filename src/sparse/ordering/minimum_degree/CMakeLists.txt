target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/amdbar.F
  ${CMAKE_CURRENT_LIST_DIR}/AMDReordering.hpp
  ${CMAKE_CURRENT_LIST_DIR}/genmmd.F
  ${CMAKE_CURRENT_LIST_DIR}/mmdelm.F
  ${CMAKE_CURRENT_LIST_DIR}/mmdint.F
  ${CMAKE_CURRENT_LIST_DIR}/mmdnum.F
  ${CMAKE_CURRENT_LIST_DIR}/mmdupd.F
  ${CMAKE_CURRENT_LIST_DIR}/ordmmd.F
  ${CMAKE_CURRENT_LIST_DIR}/MMDReordering.hpp)

#install(FILES
#  MMDReordering.hpp
#  AMDReordering.hpp include/sparse/ordering/)
