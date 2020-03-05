target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/sgeqp3tol.f ${CMAKE_CURRENT_LIST_DIR}/dgeqp3tol.f ${CMAKE_CURRENT_LIST_DIR}/cgeqp3tol.f ${CMAKE_CURRENT_LIST_DIR}/zgeqp3tol.f
  ${CMAKE_CURRENT_LIST_DIR}/sgeqrfmod.f ${CMAKE_CURRENT_LIST_DIR}/dgeqrfmod.f ${CMAKE_CURRENT_LIST_DIR}/cgeqrfmod.f ${CMAKE_CURRENT_LIST_DIR}/zgeqrfmod.f
  ${CMAKE_CURRENT_LIST_DIR}/sgeqr2mod.f ${CMAKE_CURRENT_LIST_DIR}/dgeqr2mod.f ${CMAKE_CURRENT_LIST_DIR}/cgeqr2mod.f ${CMAKE_CURRENT_LIST_DIR}/zgeqr2mod.f
  ${CMAKE_CURRENT_LIST_DIR}/slaqpsmod.f ${CMAKE_CURRENT_LIST_DIR}/dlaqpsmod.f ${CMAKE_CURRENT_LIST_DIR}/claqpsmod.f ${CMAKE_CURRENT_LIST_DIR}/zlaqpsmod.f
  ${CMAKE_CURRENT_LIST_DIR}/slaqp2mod.f ${CMAKE_CURRENT_LIST_DIR}/dlaqp2mod.f ${CMAKE_CURRENT_LIST_DIR}/claqp2mod.f ${CMAKE_CURRENT_LIST_DIR}/zlaqp2mod.f
  ${CMAKE_CURRENT_LIST_DIR}/sgetrfmod.f ${CMAKE_CURRENT_LIST_DIR}/dgetrfmod.f ${CMAKE_CURRENT_LIST_DIR}/cgetrfmod.f ${CMAKE_CURRENT_LIST_DIR}/zgetrfmod.f
  ${CMAKE_CURRENT_LIST_DIR}/sgetf2mod.f ${CMAKE_CURRENT_LIST_DIR}/dgetf2mod.f ${CMAKE_CURRENT_LIST_DIR}/cgetf2mod.f ${CMAKE_CURRENT_LIST_DIR}/zgetf2mod.f
  ${CMAKE_CURRENT_LIST_DIR}/sorglqmod.f ${CMAKE_CURRENT_LIST_DIR}/dorglqmod.f ${CMAKE_CURRENT_LIST_DIR}/cunglqmod.f ${CMAKE_CURRENT_LIST_DIR}/zunglqmod.f
  ${CMAKE_CURRENT_LIST_DIR}/sorgl2mod.f ${CMAKE_CURRENT_LIST_DIR}/dorgl2mod.f ${CMAKE_CURRENT_LIST_DIR}/cungl2mod.f ${CMAKE_CURRENT_LIST_DIR}/zungl2mod.f
  ${CMAKE_CURRENT_LIST_DIR}/slarfbmod.f ${CMAKE_CURRENT_LIST_DIR}/dlarfbmod.f ${CMAKE_CURRENT_LIST_DIR}/clarfbmod.f ${CMAKE_CURRENT_LIST_DIR}/zlarfbmod.f
  ${CMAKE_CURRENT_LIST_DIR}/slarftmod.f ${CMAKE_CURRENT_LIST_DIR}/dlarftmod.f ${CMAKE_CURRENT_LIST_DIR}/clarftmod.f ${CMAKE_CURRENT_LIST_DIR}/zlarftmod.f
  ${CMAKE_CURRENT_LIST_DIR}/slarfmod.f  ${CMAKE_CURRENT_LIST_DIR}/dlarfmod.f  ${CMAKE_CURRENT_LIST_DIR}/clarfmod.f  ${CMAKE_CURRENT_LIST_DIR}/zlarfmod.f
  ${CMAKE_CURRENT_LIST_DIR}/sgelqfmod.f ${CMAKE_CURRENT_LIST_DIR}/dgelqfmod.f ${CMAKE_CURRENT_LIST_DIR}/cgelqfmod.f ${CMAKE_CURRENT_LIST_DIR}/zgelqfmod.f
  ${CMAKE_CURRENT_LIST_DIR}/sgelq2mod.f ${CMAKE_CURRENT_LIST_DIR}/dgelq2mod.f ${CMAKE_CURRENT_LIST_DIR}/cgelq2mod.f ${CMAKE_CURRENT_LIST_DIR}/zgelq2mod.f)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pcgeqpftol.f ${CMAKE_CURRENT_LIST_DIR}/pdgeqpftol.f ${CMAKE_CURRENT_LIST_DIR}/psgeqpftol.f ${CMAKE_CURRENT_LIST_DIR}/pzgeqpftol.f)
endif()

