target_sources(strumpack
  PRIVATE
  sgeqp3tol.f dgeqp3tol.f cgeqp3tol.f zgeqp3tol.f
  sgeqrfmod.f dgeqrfmod.f cgeqrfmod.f zgeqrfmod.f
  sgeqr2mod.f dgeqr2mod.f cgeqr2mod.f zgeqr2mod.f
  slaqpsmod.f dlaqpsmod.f claqpsmod.f zlaqpsmod.f
  slaqp2mod.f dlaqp2mod.f claqp2mod.f zlaqp2mod.f
  sgetrfmod.f dgetrfmod.f cgetrfmod.f zgetrfmod.f
  sgetf2mod.f dgetf2mod.f cgetf2mod.f zgetf2mod.f
  sorglqmod.f dorglqmod.f cunglqmod.f zunglqmod.f
  sorgl2mod.f dorgl2mod.f cungl2mod.f zungl2mod.f
  slarfbmod.f dlarfbmod.f clarfbmod.f zlarfbmod.f
  slarftmod.f dlarftmod.f clarftmod.f zlarftmod.f
  slarfmod.f  dlarfmod.f  clarfmod.f  zlarfmod.f
  sgelqfmod.f dgelqfmod.f cgelqfmod.f zgelqfmod.f
  sgelq2mod.f dgelq2mod.f cgelq2mod.f zgelq2mod.f)

if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    pcgeqpftol.f pdgeqpftol.f psgeqpftol.f pzgeqpftol.f)
endif()

