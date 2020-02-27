if(STRUMPACK_USE_MPI)
  target_sources(strumpack
    PRIVATE
    ana_orderings.F)
endif()
