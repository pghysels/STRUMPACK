target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/sgeqp3tol.f
  ${CMAKE_CURRENT_LIST_DIR}/dgeqp3tol.f
  ${CMAKE_CURRENT_LIST_DIR}/cgeqp3tol.f
  ${CMAKE_CURRENT_LIST_DIR}/zgeqp3tol.f
  ${CMAKE_CURRENT_LIST_DIR}/slapmr.f
  ${CMAKE_CURRENT_LIST_DIR}/dlapmr.f
  ${CMAKE_CURRENT_LIST_DIR}/clapmr.f
  ${CMAKE_CURRENT_LIST_DIR}/zlapmr.f)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pcgeqpftol.f
    ${CMAKE_CURRENT_LIST_DIR}/pdgeqpftol.f
    ${CMAKE_CURRENT_LIST_DIR}/psgeqpftol.f
    ${CMAKE_CURRENT_LIST_DIR}/pzgeqpftol.f)
endif()

