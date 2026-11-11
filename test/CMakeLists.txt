add_executable(test_HSS_seq    test_HSS_seq.cpp)
add_executable(test_sparse_seq test_sparse_seq.cpp)
add_executable(test_BLR_seq    test_BLR_seq.cpp)
add_executable(test_BLR_pivot  test_BLR_pivot.cpp)
add_executable(test_matrix_IO  test_matrix_IO.cpp)
add_executable(test_SPD_seq test_SPD_seq.cpp)
add_executable(test_SPD_mixedPrecision test_SPD_mixedPrecision.cpp)
add_executable(test_schur test_schur.cpp)

target_link_libraries(test_HSS_seq strumpack)
target_link_libraries(test_sparse_seq strumpack)
target_link_libraries(test_BLR_seq strumpack)
target_link_libraries(test_BLR_pivot strumpack)
target_link_libraries(test_matrix_IO strumpack)
target_link_libraries(test_SPD_seq strumpack)
target_link_libraries(test_SPD_mixedPrecision strumpack)
target_link_libraries(test_schur strumpack)

# Sparse test matrices are downloaded (into the build/test directory)
# by the Download_sparse_test_matrices test, which is a fixture setup
# for all tests that use one of these matrices. Downloads can fail
# (network, server down). In that case download_mtx.sh exits with 77,
# and the tests that need a missing matrix exit with 77 as well, via
# require_mtx.sh. ctest reports both as "skipped" instead of "failed".
set(STRUMPACK_DOWNLOADED_MTX
  utm300 mesh3e1 t2dal bcsstk28 cavity16 bcsstm08 sherman4 rdb968 gemat11)
list(JOIN STRUMPACK_DOWNLOADED_MTX "|" STRUMPACK_DOWNLOADED_MTX_REGEX)

add_test(NAME "Download_sparse_test_matrices"
  COMMAND /bin/sh ${CMAKE_CURRENT_SOURCE_DIR}/download_mtx.sh)
set_tests_properties("Download_sparse_test_matrices" PROPERTIES
  FIXTURES_SETUP strumpack_test_matrices
  SKIP_RETURN_CODE 77
  TIMEOUT 1200)

# Same as add_test(<name> <command> [args...]), but if one of the
# arguments is a downloaded test matrix (<name>/<name>.mtx), the test
# depends on the download fixture and is skipped when the matrix is
# not available.
function(strumpack_add_test name)
  set(mtx "")
  foreach(arg IN LISTS ARGN)
    if(arg MATCHES "^(${STRUMPACK_DOWNLOADED_MTX_REGEX})/[^/]+\\.mtx$")
      set(mtx ${arg})
      break()
    endif()
  endforeach()
  if(mtx)
    add_test(NAME ${name} COMMAND /bin/sh
      ${CMAKE_CURRENT_SOURCE_DIR}/require_mtx.sh ${mtx} ${ARGN})
    set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      FIXTURES_REQUIRED strumpack_test_matrices)
  else()
    add_test(${name} ${ARGN})
  endif()
endfunction()

strumpack_add_test("user_test_HSS_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 100)
strumpack_add_test("user_test_sparse_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
  ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
strumpack_add_test("user_matrix_IO" ${CMAKE_CURRENT_BINARY_DIR}/test_matrix_IO T 1000)
strumpack_add_test("user_test_BLR_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300)
strumpack_add_test("user_test_SPD_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_SPD_seq bcsstm08/bcsstm08.mtx)
strumpack_add_test("user_test_SPD_mixedPrecision" ${CMAKE_CURRENT_BINARY_DIR}/test_SPD_mixedPrecision bcsstm08/bcsstm08.mtx)

if(STRUMPACK_USE_MPI)
  if(STRUMPACK_USE_BPACK)
    add_executable(test_HODLR_process_tree_mpi test_HODLR_process_tree_mpi.cpp)
    target_link_libraries(test_HODLR_process_tree_mpi strumpack)
    add_test(NAME HODLR_process_tree_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_HODLR_process_tree_mpi>
      ${MPIEXEC_POSTFLAGS})
    set_tests_properties(HODLR_process_tree_mpi PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endif()
  add_executable(test_HSS_mpi             test_HSS_mpi.cpp)
  add_executable(test_sparse_mpi          test_sparse_mpi.cpp)
  add_executable(test_structure_reuse_mpi test_structure_reuse_mpi.cpp)
  add_executable(test_BLR_mpi             test_BLR_mpi.cpp)
  add_executable(test_trsm_narrow_mpi     test_trsm_narrow_mpi.cpp)

  target_link_libraries(test_HSS_mpi strumpack)
  target_link_libraries(test_sparse_mpi strumpack)
  target_link_libraries(test_structure_reuse_mpi strumpack)
  target_link_libraries(test_BLR_mpi strumpack)
  target_link_libraries(test_trsm_narrow_mpi strumpack)

  execute_process(COMMAND ${MPIEXEC} --oversubscribe --version RESULT_VARIABLE oversubscribe_supported)
  if(${oversubscribe_supported} EQUAL 0)
    set(OVERSUBSCRIBEFLAG "--oversubscribe")
  endif()

  strumpack_add_test("user_test_HSS_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi T 100)
  strumpack_add_test("user_test_sparse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
  strumpack_add_test("user_structure_reuse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
  # add_test("user_test_BLR_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
  #   ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
  #   ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi 1000)
  strumpack_add_test("user_test_trsm_narrow_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_trsm_narrow_mpi)
  set_property(TEST "user_test_trsm_narrow_mpi" PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  foreach(np 1 3 4)
    strumpack_add_test("user_test_schur_mpi_${np}" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
      ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
      ${CMAKE_CURRENT_BINARY_DIR}/test_schur)
  endforeach()
else()
  strumpack_add_test("user_test_schur" ${CMAKE_CURRENT_BINARY_DIR}/test_schur)
endif()

set(test_name "HSS_seq_1")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_2")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 200 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_3")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_4")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_5")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_6")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 1 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_7")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_8")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 200 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_9")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 10 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_10")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_11")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_12")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_13")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 500 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_14")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_15")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_16")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_17")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 10 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_18")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_19")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_20")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_21")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_22")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch Gaussian)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_23")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo chunk --hss_nnz0 2 --hss_nnz 2)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_24")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo chunk --hss_nnz0 4 --hss_nnz 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_25")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo perm --hss_nnz0 2 --hss_nnz 2)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_26")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo perm --hss_nnz0 4 --hss_nnz 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")


set(test_name "BLR_seq_1")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm RL)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_2")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm LL)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_3")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Star --blr_compression_kernel full)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_4")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Star --blr_compression_kernel half)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_5")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Comb --blr_compression_kernel full)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_6")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Comb --blr_compression_kernel half)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

# BLR LU with actual row interchanges (random matrix), repeated
# multiple times with several threads, to catch errors in the
# pivoting and in the OpenMP task dependencies.
foreach(alg RL LL Comb Star)
  set(test_name "BLR_pivot_${alg}")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_pivot 1000 10 --blr_factor_algorithm ${alg})
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
endforeach()


if(STRUMPACK_USE_MPI)
  set(test_name "HSS_mpi_1")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_2")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_3")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_4")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_6")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_7")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_8")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_9")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 10 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_10")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_11")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_12")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_13")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)

  set(test_name "HSS_mpi_14")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_15")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_16")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_17")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_18")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_19")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_20")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 200 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_21")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_22")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_23")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_24")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_25")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_26")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_27")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 )
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_28")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_29")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_30")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_31")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_32")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_33")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_34")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 200 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_35")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_36")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_38")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_39")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_40")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_41")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_42")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_43")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_44")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_45")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_46")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_47")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_48")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_49")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_50")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_51")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_52")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 10 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_53")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_54")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_55")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # set(test_name "BLR_mpi_1")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm RL)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "BLR_mpi_2")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm LL)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "BLR_mpi_3")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm Star --blr_compression_kernel full)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "BLR_mpi_4")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm Star --blr_compression_kernel half)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "BLR_mpi_5")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm Comb --blr_compression_kernel full)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "BLR_mpi_6")
  # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi
  #   ${MPIEXEC_POSTFLAGS} 1000 --blr_factor_algorithm Comb --blr_compression_kernel half)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

endif()


set(test_name "SPARSE_seq_3")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_9")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_15")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_21")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_27")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_30")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_33")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_39")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_44")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_50")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25 --sp_rel_tol 1e-12)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_56")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

# set(test_name "SPARSE_seq_62")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
# set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_66")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 32 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

# set(test_name "SPARSE_seq_67")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
# set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

# set(test_name "SPARSE_seq_68")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
# set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

# set(test_name "SPARSE_seq_71")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
# set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_74")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression BLR --sp_enable_replace_tiny_pivots --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_77")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression BLR --sp_enable_replace_tiny_pivots --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25 --sp_rel_tol 1e-12)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_80")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_86")
strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

# Column-wise (COLWISE) BLR factorization with (near) exact
# compression, for nonsymmetric matrices that require pivoting. The
# factorization should be exact, so the solve must converge in 1-2
# iterations.
foreach(mtx cavity16 rdb968)
  set(test_name "SPARSE_BLR_seq_pivot_COLWISE_${mtx}")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq ${mtx}/${mtx}.mtx --sp_compression BLR --blr_factor_algorithm COLWISE --blr_leaf_size 4 --blr_rel_tol 1e-14 --blr_abs_tol 1e-14 --sp_compression_min_sep_size 25 --sp_enable_replace_tiny_pivots --sp_maxit 2 --sp_disable_gpu)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")
endforeach()


if(STRUMPACK_USE_SCOTCH)
  set(test_name "SPARSE_seq_scotch_1")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_2")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_3")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_4")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_5")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_6")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_7")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "SPARSE_seq_scotch_8")
  # add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --blr_admissibility strong --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_9")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_10")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_11")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_12")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_13")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
endif()

if(STRUMPACK_USE_ZFP)
  set(test_name "SPARSE_seq_lossy")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx --sp_compression LOSSY --sp_lossy_precision 16 --sp_maxit 10)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_lossless")
  strumpack_add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx --sp_compression LOSSLESS --sp_maxit 1)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
endif()


if(STRUMPACK_USE_MPI)
  set(test_name "SPARSE_HSS_mpi_1")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-2 --hss_abs_tol 1e-2 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_5")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_7")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_8")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_10")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_14")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_19")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_21")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # BLR with (near) exact compression: the factorization should be
  # exact, so the solve must converge in 1-2 iterations. This catches
  # errors in the distributed BLR factorization (e.g. pivoting) that
  # GMRES would otherwise hide.
  foreach(np 2 4 9)
    set(test_name "SPARSE_BLR_mpi_exact_np${np}")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-14 --blr_abs_tol 1e-14 --sp_reordering_method metis --sp_compression_min_sep_size 25 --sp_maxit 2)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endforeach()

  # Same, but for a nonsymmetric matrix that requires (partial)
  # pivoting, with multiple OpenMP threads per rank, and the
  # right-looking distributed BLR factorization. GPU is disabled so
  # the CPU code path is tested also in GPU-enabled builds.
  foreach(np 4 9)
    set(test_name "SPARSE_BLR_mpi_pivot_RL_np${np}")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression BLR --blr_factor_algorithm RL --blr_leaf_size 4 --blr_rel_tol 1e-14 --blr_abs_tol 1e-14 --sp_compression_min_sep_size 25 --sp_enable_replace_tiny_pivots --sp_maxit 2 --sp_disable_gpu)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")
    # same for the column-wise (COLWISE) distributed BLR factorization
    foreach(mtx cavity16 rdb968)
      set(test_name "SPARSE_BLR_mpi_pivot_COLWISE_${mtx}_np${np}")
      strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
        ${MPIEXEC_POSTFLAGS} ${mtx}/${mtx}.mtx --sp_compression BLR --blr_factor_algorithm COLWISE --blr_leaf_size 4 --blr_rel_tol 1e-14 --blr_abs_tol 1e-14 --sp_compression_min_sep_size 25 --sp_enable_replace_tiny_pivots --sp_maxit 2 --sp_disable_gpu)
      set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=4")
    endforeach()
  endforeach()

  if(STRUMPACK_USE_BPACK)
    set(test_name "SPARSE_HODLR_mpi_1")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HODLR --sp_enable_replace_tiny_pivots --hodlr_leaf_size 4 --hodlr_rel_tol 1e-2 --hodlr_abs_tol 1e-2 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_5")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_7")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_8")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25 --sp_rel_tol 1e-8)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_10")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 1 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    # set(test_name "SPARSE_HODBF_mpi_14")
    # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    #   ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 2 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_19")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 100 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_21")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 100 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_SCOTCH)
    set(test_name "SPARSE_mpi_13")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_16")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_18")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_22")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_24")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_PTSCOTCH)
    set(test_name "SPARSE_mpi_26")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_27")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_28")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_29")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_30")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-6 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_31")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_PARMETIS)
    set(test_name "SPARSE_mpi_9")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_6")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_11")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_12")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_15")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_17")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_20")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_23")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_25")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_2")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_3")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_4")
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()


  # test structure reuse with different matching jobs
  set(test_name "structure_reuse_1")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 0)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_2")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 2)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_3")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 3)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_4")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_5")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 5)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # test structure reuse with HSS compression enabled
  set(test_name "structure_reuse_HSS_1")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 0 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_2")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 2 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_3")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 3 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_4")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 4 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_5")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 5 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # with compression, update_matrix_values reorders the separators
  # again: the solves with the new factorization should converge as
  # fast as the first ones
  set(test_name "structure_reuse_HSS_maxit")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 0 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25 --sp_maxit 10)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_BLR_maxit")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression BLR --sp_matching 0 --blr_leaf_size 4 --blr_rel_tol 1e-3 --sp_compression_min_sep_size 25 --sp_maxit 10)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")


  # test matching, using MC64, and CombBLAS below, if available
  set(test_name "SPARSE_mpi_matching_2")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 2)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_3")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 3)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_4")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_5")
  strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 5)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # test CombBLAS
  if(CombBLAS_FOUND)
    set(test_name "SPARSE_mpi_CombBLAS")
    # number of processes needs to be a power of 2
    strumpack_add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 6)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

endif()
