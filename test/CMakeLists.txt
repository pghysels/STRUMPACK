include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/test)

add_executable(test_HSS_seq test_HSS_seq)
add_executable(test_sparse_seq test_sparse_seq)
add_executable(test_BLR_seq test_BLR_seq)

target_link_libraries(test_HSS_seq strumpack ${LIB})
target_link_libraries(test_sparse_seq strumpack ${LIB})
target_link_libraries(test_BLR_seq strumpack ${LIB})

add_test("user_test_HSS_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 100)
add_test("user_test_sparse_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
  ../examples/pde900.mtx)



if(STRUMPACK_USE_MPI)
  add_executable(test_HSS_mpi test_HSS_mpi)
  add_executable(test_sparse_mpi test_sparse_mpi)
  target_link_libraries(test_HSS_mpi strumpack ${LIB})
  target_link_libraries(test_sparse_mpi strumpack ${LIB})

  add_test("user_test_HSS_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi T 100)
  add_test("user_test_sparse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi m
    ../examples/pde900.mtx)
endif()


if(STRUMPACK_DEV_TESTING)
  set(THREADS 1 3 8)
  set(PROCS   2 4 9 13 16 17 19)
  set(MATRIX  T U L)
  set(SIZES   1 10 200 500)
  set(LEAF    1 3 16 128)
  set(RTOL    1e-1 1e-5 1e-10)
  set(ATOL    1e-2 1e-10 1e-13)
  set(SYNC    enable disable)
  set(COMP    original stable)

  ## TODO try smaller values as well
  #set(D0      1 2 4 8 64 128)
  set(D0      16 32 64 128)
  set(DD      4 8)

  function(random_element list element)
    list(LENGTH ${list} len)
    string(RANDOM LENGTH 4 ALPHABET "0123456789" rn)
    math(EXPR index '${rn}%${len}')
    list(GET ${list} ${index} el)
    set(${element} ${el} PARENT_SCOPE)
  endfunction(random_element)

  set(test_output ${CMAKE_BINARY_DIR}/test/HSS_seq.out)
  set(test_names "test")
  foreach(ntests RANGE 20)
    random_element(THREADS th)
    random_element(MATRIX m)
    random_element(SIZES n)
    random_element(LEAF l)
    random_element(RTOL t)
    random_element(ATOL a)
    random_element(SYNC s)
    random_element(COMP c)
    random_element(D0 d)
    random_element(DD deltad)
    set(test_name "HSS_seq_Th${th}_M${m}_N${n}_L${l}_T${t}_A${a}_S${s}_C${c}_D0${d}_DD${deltad}")
    list(FIND ${test_names} ${test_name} tindex)
    if ("${tindex}" STREQUAL "-1")
      list(APPEND ${test_names} ${test_name})
      set(CTEST_ENVIRONMENT "OMP_NUM_THREADS=${th}")
      add_test(${test_name}
        ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq ${m} ${n}
        --hss_leaf_size ${l} --hss_rel_tol ${t} --hss_abs_tol ${a}
        --hss_${s}_sync --hss_compression_algorithm ${c} --hss_d0 ${d}
        --hss_dd ${deltad})
      set_property(TEST ${test_name} PROPERTY
        ENVIRONMENT "OMP_NUM_THREADS=${th}")
    endif()
  endforeach()

  if(STRUMPACK_USE_MPI)
    ## Distributed HSS test
    set(test_output ${CMAKE_BINARY_DIR}/test/HSS_mpi.out)
    set(test_names "test")

    foreach(c ${COMP})
      foreach(s ${SYNC})
        set(th 1)
        set(mpi 13)
        set(m L)
        set(n 500)
        set(l 8)
        set(t 1e-4)
        set(a 1e-10)
        set(d 16)
        set(deltad 8)
        set(test_name "HSS_mpi_P${mpi}_Th${th}_M${m}_N${n}_L${l}_T${t}_A${a}_S${s}_C${c}_D0${d}_DD${deltad}")
        list(APPEND ${test_names} ${test_name})
        add_test(${test_name}
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${mpi}
          ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
          ${MPIEXEC_POSTFLAGS} ${m} ${n} --hss_leaf_size ${l}
          --hss_rel_tol ${t} --hss_abs_tol ${a}
          --hss_${s}_sync --hss_compression_algorithm ${c}
          --hss_d0 ${d} --hss_dd ${deltad})
        set_property(TEST ${test_name} PROPERTY
          ENVIRONMENT "OMP_NUM_THREADS=${th}")
      endforeach()
    endforeach()


    foreach(ntests RANGE 50)
      #random_element(THREADS th)
      set(th 1)
      random_element(PROCS mpi)
      random_element(MATRIX m)
      random_element(SIZES n)
      random_element(LEAF l)
      random_element(RTOL t)
      random_element(ATOL a)
      random_element(SYNC s)
      random_element(COMP c)
      random_element(D0 d)
      random_element(DD deltad)
      set(test_name "HSS_mpi_P${mpi}_Th${th}_M${m}_N${n}_L${l}_T${t}_A${a}_S${s}_C${c}_D0${d}_DD${deltad}")
      list(FIND ${test_names} ${test_name} tindex)
      if ("${tindex}" STREQUAL "-1")
        list(APPEND ${test_names} ${test_name})
        add_test(${test_name}
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${mpi}
          ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
          ${MPIEXEC_POSTFLAGS} ${m} ${n} --hss_leaf_size ${l}
          --hss_rel_tol ${t} --hss_abs_tol ${a} --hss_${s}_sync
          --hss_compression_algorithm ${c} --hss_d0 ${d}
          --hss_dd ${deltad})
        set_property(TEST ${test_name} PROPERTY
          ENVIRONMENT "OMP_NUM_THREADS=${th}")
      endif()
    endforeach()
  endif()

  add_test(NAME "Download_sparse_test_matrices"
    COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/test/download_mtx.sh)

  ## Sequential sparse test
  set(test_output ${CMAKE_BINARY_DIR}/test/SPARSE_seq.out)
  set(MTX utm300 mesh3e1 t2dal bcsstk28 cavity16 bcsstm08
    sherman4 rdb968 cz10228 cbuckle)
  set(NESTEDDISSECTION metis scotch)
  foreach(m ${MTX})
    foreach(nd ${NESTEDDISSECTION})
      foreach(th ${THREADS})
        set(c enable)
        set(t 1e-3)
        set(a 1e-10)
        set(d 16)
        set(deltad 8)
        set(sep 25)
        set(l 8)
        set(test_name "SPARSE_seq_Th${th}_M${m}_ND${nd}_C${c}_L${l}_T${t}_A${a}_D0${d}_DD${deltad}_SEP${sep}")
        add_test(${test_name}
          ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
          ${m}/${m}.mtx --sp_${c}_hss --hss_leaf_size ${l}
          --hss_rel_tol ${t} --hss_abs_tol ${a} --hss_d0 ${d}
          --hss_dd ${deltad} --sp_reordering_method ${nd}
          --sp_hss_min_sep_size ${sep})
        set_property(TEST ${test_name} PROPERTY
          ENVIRONMENT "OMP_NUM_THREADS=${th}")
      endforeach()
    endforeach()
  endforeach()

  if(STRUMPACK_USE_MPI)
    ## Distributed sparse test
    set(test_output ${CMAKE_BINARY_DIR}/test/SPARSE_mpi.out)
    set(NESTEDDISSECTIONMPI metis scotch parmetis ptscotch)
    foreach(ntests RANGE 50)
      set(th 1)
      random_element(NESTEDDISSECTIONMPI nd)
      random_element(MTX m)
      random_element(PROCS mpi)
      set(c enable)
      random_element(RTOL t)
      ## TODO pick these at random!!
      set(a 1e-10)
      set(d 16)
      set(deltad 8)
      set(sep 25)
      set(l 8)
      set(test_name "SPARSE_mpi_P${mpi}_Th${th}_M${m}_ND${nd}_C${c}_L${l}_T${t}_A${a}_D0${d}_DD${deltad}_SEP${sep}")
      list(FIND ${test_names} ${test_name} tindex)
      if ("${tindex}" STREQUAL "-1")
        list(APPEND ${test_names} ${test_name})
        add_test(${test_name}
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${mpi}
          ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
          ${MPIEXEC_POSTFLAGS} m ${m}/${m}.mtx --sp_${c}_hss
          --hss_leaf_size ${l} --hss_rel_tol ${t} --hss_abs_tol ${a}
          --hss_d0 ${d} --hss_dd ${deltad} --sp_reordering_method ${nd}
          --sp_hss_min_sep_size ${sep})
        set_property(TEST ${test_name} PROPERTY
          ENVIRONMENT "OMP_NUM_THREADS=${th}")
      endif()
    endforeach()
  endif()
endif()
