include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/test)

add_executable(test_HSS_seq test_HSS_seq)
add_executable(test_HSS_mpi test_HSS_mpi)
add_executable(test_sparse_seq test_sparse_seq)
add_executable(test_sparse_mpi test_sparse_mpi)

target_link_libraries(test_HSS_seq strumpack ${LIB})
target_link_libraries(test_HSS_mpi strumpack ${LIB})
target_link_libraries(test_sparse_seq strumpack ${LIB})
target_link_libraries(test_sparse_mpi strumpack ${LIB})

set(THREADS 1 3 8)
set(PROCS   2 4 9 13 16 17 19)
set(MATRIX  T U L)
set(SIZES   1 10 200 500)
set(LEAF    1 3 16 128)
set(RTOL    1 1e-1 1e-5 1e-10)
set(ATOL    1e-10 1e-13)
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
    set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
    add_test(
      NAME ${test_name}
      COMMAND "${CMAKE_COMMAND}"
      -DTEST=${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq -DARG1=${m} -DARG2=${n}
      -DARG3=--hss_leaf_size -DARG4=${l}
      -DARG5=--hss_rel_tol   -DARG6=${t}
      -DARG7=--hss_abs_tol   -DARG8=${a}
      -DARG9=--hss_${s}_sync
      -DARG10=--hss_compression_algo -DARG11=${c}
      -DARG12=--hss_d0 -DARG13=${d}
      -DARG14=--hss_dd -DARG15=${deltad}
      -DTH=${th}
      -DOUTPUT=${single_output}
      -DALL_OUTPUT=${test_output}
      -DHEADING=Dense\ matrix\ --\ threads=${th},\ matrix=${m},\ size=${n},\ leaf=${l},\ rtol=${t},\ atol=${a},\ sync=${s},\ comp=${c},\ d0=${d},\ dd=${deltad}
      -P "${CMAKE_SOURCE_DIR}/test/run_seq.cmake"
      )
  endif()
endforeach()


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
    set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
    add_test(
      NAME ${test_name}
      COMMAND "${CMAKE_COMMAND}"
      -DTEST=${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
      -DARG1=${m} -DARG2=${n} -DARG3=--hss_leaf_size -DARG4=${l}
      -DARG5=--hss_rel_tol   -DARG6=${t} -DARG7=--hss_abs_tol   -DARG8=${a}
      -DARG9=--hss_${s}_sync -DARG10=--hss_compression_algo -DARG11=${c}
      -DARG12=--hss_d0 -DARG13=${d} -DARG14=--hss_dd -DARG15=${deltad}
      -DTH=${th} -DMPIPROCS=${mpi} -DOUTPUT=${single_output} -DALL_OUTPUT=${test_output}
      -DHEADING=Dense\ matrix\ --\ procs=${mpi},\ threads=${th},\ matrix=${m},\ size=${n},\ leaf=${l},\ rtol=${t},\ atol=${a},\ sync=${s},\ comp=${c},\ d0=${d},\ dd=${deltad}
      -P "${CMAKE_SOURCE_DIR}/test/run_mpi.cmake"
      )
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
    set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
    add_test(
      NAME ${test_name}
      COMMAND "${CMAKE_COMMAND}"
      -DTEST=${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
      -DARG1=${m} -DARG2=${n}
      -DARG3=--hss_leaf_size -DARG4=${l}
      -DARG5=--hss_rel_tol   -DARG6=${t}
      -DARG7=--hss_abs_tol   -DARG8=${a}
      -DARG9=--hss_${s}_sync
      -DARG10=--hss_compression_algo -DARG11=${c}
      -DARG12=--hss_d0 -DARG13=${d}
      -DARG14=--hss_dd -DARG15=${deltad}
      -DTH=${th} -DMPIPROCS=${mpi} -DOUTPUT=${single_output} -DALL_OUTPUT=${test_output}
      -DHEADING=Dense\ matrix\ --\ procs=${mpi},\ threads=${th},\ matrix=${m},\ size=${n},\ leaf=${l},\ rtol=${t},\ atol=${a},\ sync=${s},\ comp=${c},\ d0=${d},\ dd=${deltad}
      -P "${CMAKE_SOURCE_DIR}/test/run_mpi.cmake"
      )
  endif()
endforeach()



add_test(
  NAME "Download_sparse_test_matrices"
  COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/test/download_mtx.sh
  )

## Sequential sparse test
set(test_output ${CMAKE_BINARY_DIR}/test/SPARSE_seq.out)
set(MTX utm300 mesh3e1 lnsp3937 t2dal bcsstk28 cavity16 bcsstm08 sherman4 rdb968)
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
      set(l 4)
      set(test_name "SPARSE_seq_Th${th}_M${m}_ND${nd}_C${c}_L${l}_T${t}_A${a}_D0${d}_DD${deltad}_SEP${sep}")
      set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
      add_test(
	NAME ${test_name}
	COMMAND "${CMAKE_COMMAND}"
	-DTEST=${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
	-DARG1=${m}/${m}.mtx -DARG2=--sp_${c}_hss
	-DARG3=--hss_leaf_size -DARG4=${l}
	-DARG5=--hss_rel_tol   -DARG6=${t}
	-DARG7=--hss_abs_tol   -DARG8=${a}
	-DARG9=--hss_d0 -DARG10=${d}
	-DARG11=--hss_dd -DARG12=${deltad}
	-DARG13=--sp_reordering_method -DARG14=${nd}
	-DARG15=--sp_hss_min_sep_size -DARG16=${sep}
	-DTH=${th} -DMPIPROCS=${mpi} -DOUTPUT=${single_output} -DALL_OUTPUT=${test_output}
	-DHEADING=Sparse\ matrix\ --\ threads=${th},\ matrix=${m},\ nd=${nd},\ leaf=${l},\ hssrtol=${t},\ hssatol=${a},\ comp=${c},\ d0=${d},\ dd=${deltad}
	-P "${CMAKE_SOURCE_DIR}/test/run_seq.cmake"
	)
    endforeach()
  endforeach()
endforeach()

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
  set(a 1e-10)
  set(d 16)
  set(deltad 8)
  set(sep 25)
  set(l 4)
  set(test_name "SPARSE_mpi_P${mpi}_Th${th}_M${m}_ND${nd}_C${c}_L${l}_T${t}_A${a}_D0${d}_DD${deltad}_SEP${sep}")
  list(FIND ${test_names} ${test_name} tindex)
  if ("${tindex}" STREQUAL "-1")
    list(APPEND ${test_names} ${test_name})
    set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
    set(single_output "${CMAKE_BINARY_DIR}/test/${test_name}.out")
    add_test(
      NAME ${test_name}
      COMMAND "${CMAKE_COMMAND}"
      -DTEST=${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      -DARG1=m -DARG2=${m}/${m}.mtx -DARG3=--sp_${c}_hss
      -DARG4=--hss_leaf_size -DARG5=${l}
      -DARG6=--hss_rel_tol   -DARG7=${t}
      -DARG8=--hss_abs_tol   -DARG9=${a}
      -DARG10=--hss_d0 -DARG11=${d}
      -DARG12=--hss_dd -DARG13=${deltad}
      -DARG14=--sp_reordering_method -DARG15=${nd}
      -DARG16=--sp_hss_min_sep_size -DARG17=${sep}
      -DTH=${th} -DMPIPROCS=${mpi} -DOUTPUT=${single_output} -DALL_OUTPUT=${test_output}
      -DHEADING=Sparse\ matrix\ --\ threads=${th},\ matrix=${m},\ nd=${nd},\ leaf=${l},\ hssrtol=${t},\ hssatol=${a},\ comp=${c},\ d0=${d},\ dd=${deltad}
      -P "${CMAKE_SOURCE_DIR}/test/run_mpi.cmake"
      )
  endif()
endforeach()
