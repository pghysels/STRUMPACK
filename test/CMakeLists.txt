include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/test)

add_executable(test_HSS_seq test_HSS_seq)
add_executable(test_HSS_mpi test_HSS_mpi)

target_link_libraries(test_HSS_seq strumpack ${LIB})
target_link_libraries(test_HSS_mpi strumpack ${LIB})

set(THREADS 1 3 8)
set(MATRIX  T U)
set(SIZES   1 10 500)
set(LEAF    1 3 16 128)
set(RTOL    1 1e-1 1e-5 1e-13 1e-20)
set(ATOL    1e-1 1e-13 1e-20)
set(SYNC    enable disable)
set(COMP    original stable)
set(D0      1 2 4 8 64 128 300)
set(DD      1 2 4 8 64)

function(random_element list element)
  list(LENGTH ${list} len)
  string(RANDOM LENGTH 4 ALPHABET "0123456789" rn)
  math(EXPR index '${rn}%${len}')
  list(GET ${list} ${index} el)
  set(${element} ${el} PARENT_SCOPE)
endfunction(random_element)


set(test_output ${CMAKE_BINARY_DIR}/test/test_HSS_seq.out)
set(test_names "test")
foreach(ntests RANGE 99)
  random_element(RTOL r)
  random_element(THREADS p)
  random_element(MATRIX m)
  random_element(SIZES n)
  random_element(LEAF l)
  random_element(RTOL t)
  random_element(ATOL a)
  random_element(SYNC s)
  random_element(COMP c)
  random_element(D0 d)
  random_element(DD deltad)
  set(test_name "test_HSS_seq_P${p}_M${m}_N${n}_L${l}_T${t}_A${a}_S${s}_C${c}_D0${d}_DD${deltad}")
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
      -DTH=${p}
      -DOUTPUT=${single_output}
      -DALL_OUTPUT=${test_output}
      -DHEADING=Dense\ matrix\ --\ threads=${p},\ matrix=${m},\ size=${n},\ leaf=${l},\ rtol=${t},\ atol=${a},\ sync=${s}
      -P "${CMAKE_SOURCE_DIR}/test/run_seq.cmake"
      )
  endif()
endforeach()


## Sequential sparse test
# TODO

## Distributed HSS test
# TODO

## Distributed sparse test
# TODO

