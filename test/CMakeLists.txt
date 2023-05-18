add_executable(test_HSS_seq    test_HSS_seq.cpp)
add_executable(test_sparse_seq test_sparse_seq.cpp)
add_executable(test_BLR_seq    test_BLR_seq.cpp)
add_executable(test_matrix_IO  test_matrix_IO.cpp)

target_link_libraries(test_HSS_seq strumpack)
target_link_libraries(test_sparse_seq strumpack)
target_link_libraries(test_BLR_seq strumpack)
target_link_libraries(test_matrix_IO strumpack)

add_test("user_test_HSS_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 100)
add_test("user_test_sparse_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
  ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
add_test("user_matrix_IO" ${CMAKE_CURRENT_BINARY_DIR}/test_matrix_IO T 1000)
add_test("user_test_BLR_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300)

if(STRUMPACK_USE_MPI)
  add_executable(test_HSS_mpi             test_HSS_mpi.cpp)
  add_executable(test_sparse_mpi          test_sparse_mpi.cpp)
  add_executable(test_structure_reuse_mpi test_structure_reuse_mpi.cpp)
  add_executable(test_BLR_mpi             test_BLR_mpi.cpp)

  target_link_libraries(test_HSS_mpi strumpack)
  target_link_libraries(test_sparse_mpi strumpack)
  target_link_libraries(test_structure_reuse_mpi strumpack)
  target_link_libraries(test_BLR_mpi strumpack)

  # TODO check whether this is supported?
  set(OVERSUBSCRIBEFLAG "--oversubscribe")

  add_test("user_test_HSS_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi T 100)
  add_test("user_test_sparse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
  add_test("user_structure_reuse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx)
  # add_test("user_test_BLR_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
  #   ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
  #   ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_mpi 1000)
endif()

set(test_name "HSS_seq_1")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_2")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 200 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_3")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_4")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_5")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_6")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 1 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_7")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_8")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 200 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_9")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 10 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_10")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_11")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_12")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_13")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 500 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_14")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_15")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "HSS_seq_16")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_17")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 10 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_18")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq L 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_19")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_20")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "HSS_seq_21")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_22")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch Gaussian)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_23")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo chunk --hss_nnz0 2 --hss_nnz 2)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_24")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo chunk --hss_nnz0 4 --hss_nnz 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_25")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo perm --hss_nnz0 2 --hss_nnz 2)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "HSS_seq_26")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 1000 --hss_leaf_size 32 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 8 --hss_dd 8 --hss_compression_sketch SJLT --hss_SJLT_algo perm --hss_nnz0 4 --hss_nnz 4)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")


set(test_name "BLR_seq_1")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm RL)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_2")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm LL)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_3")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Star --blr_compression_kernel full)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_4")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Star --blr_compression_kernel half)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_5")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Comb --blr_compression_kernel full)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "BLR_seq_6")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_BLR_seq 300 --blr_factor_algorithm Comb --blr_compression_kernel half)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")


if(STRUMPACK_USE_MPI)
  set(test_name "HSS_mpi_1")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_2")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_3")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_4")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_6")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_7")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_8")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_9")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 10 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_10")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_11")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_12")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_13")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)

  set(test_name "HSS_mpi_14")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_15")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_16")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_17")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_18")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_19")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_20")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 200 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_21")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_22")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_23")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_24")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_25")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_26")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_27")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 )
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_28")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_29")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_30")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_31")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_32")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_33")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_34")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 200 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_35")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_36")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_38")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_39")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_40")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_41")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_42")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 1 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_43")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_44")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 500 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_45")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_46")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_47")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_48")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_49")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_50")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_51")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_52")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 10 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_53")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} U 500 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_54")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 1 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "HSS_mpi_55")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
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


add_test(NAME "Download_sparse_test_matrices" COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/test/download_mtx.sh)


set(test_name "SPARSE_seq_3")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_9")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_15")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_21")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_27")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_30")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_33")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_39")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_44")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_50")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_56")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

# set(test_name "SPARSE_seq_62")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
# set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_66")
# add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 32 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-4 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25)
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
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression BLR --sp_enable_replace_tiny_pivots --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_77")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression BLR --sp_enable_replace_tiny_pivots --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_enable_METIS_NodeND --sp_compression_min_sep_size 25 --sp_rel_tol 1e-10)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_80")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_86")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method metis --sp_compression_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")


if(STRUMPACK_USE_SCOTCH)
  set(test_name "SPARSE_seq_scotch_1")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_2")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_3")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_4")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_5")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_6")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_7")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  # set(test_name "SPARSE_seq_scotch_8")
  # add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --blr_admissibility strong --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_9")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_10")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_11")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_12")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_scotch_13")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression BLR --blr_leaf_size 4 --blr_rel_tol 1e-3 --blr_abs_tol 1e-10 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
endif()

if(STRUMPACK_USE_ZFP)
  set(test_name "SPARSE_seq_lossy")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx --sp_compression LOSSY --sp_lossy_precision 16 --sp_maxit 10)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

  set(test_name "SPARSE_seq_lossless")
  add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx --sp_compression LOSSLESS --sp_maxit 1)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")
endif()


if(STRUMPACK_USE_MPI)
  set(test_name "SPARSE_HSS_mpi_1")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-2 --hss_abs_tol 1e-2 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_7")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_8")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_10")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_14")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_19")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_HSS_mpi_21")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  if(STRUMPACK_USE_BPACK)
    set(test_name "SPARSE_HODLR_mpi_1")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HODLR --sp_enable_replace_tiny_pivots --hodlr_leaf_size 4 --hodlr_rel_tol 1e-2 --hodlr_abs_tol 1e-2 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_5")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_7")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODLR_mpi_8")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --sp_reordering_method metis --sp_compression_min_sep_size 25 --sp_rel_tol 1e-8)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_10")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-10 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 1 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    # set(test_name "SPARSE_HODBF_mpi_14")
    # add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    #   ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 2 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    # set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_19")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 100 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_HODBF_mpi_21")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HODLR --hodlr_leaf_size 4 --hodlr_rel_tol 1e-1 --hodlr_abs_tol 1e-10 --hodlr_rank_guess 16 --hodlr_butterfly_levels 100 --sp_reordering_method metis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_SCOTCH)
    set(test_name "SPARSE_mpi_13")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_16")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_18")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_22")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_24")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_PTSCOTCH)
    set(test_name "SPARSE_mpi_26")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_27")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_28")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_29")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_30")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-6 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_31")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

  if(STRUMPACK_USE_PARMETIS)
    set(test_name "SPARSE_mpi_9")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_6")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_11")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_12")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_15")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_17")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_20")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_23")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_25")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_2")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_3")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    set(test_name "SPARSE_mpi_4")
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()


  # test structure reuse with different matching jobs
  set(test_name "structure_reuse_1")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 0)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_2")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 2)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_3")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 3)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_4")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} utm300/utm300.mtx --sp_compression NONE --sp_matching 5)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # test structure reuse with HSS compression enabled
  set(test_name "structure_reuse_HSS_1")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 0 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_2")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 2 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_3")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 3 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_4")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 4 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "structure_reuse_HSS_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_structure_reuse_mpi
    ${MPIEXEC_POSTFLAGS} ${PROJECT_SOURCE_DIR}/examples/sparse/data/pde900.mtx
    --sp_compression HSS --sp_matching 5 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")


  # test matching, using MC64, and CombBLAS below, if available
  set(test_name "SPARSE_mpi_matching_2")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 2)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_3")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 3)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_4")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 4)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  set(test_name "SPARSE_mpi_matching_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 5)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  # test CombBLAS
  if(CombBLAS_FOUND)
    set(test_name "SPARSE_mpi_CombBLAS")
    # number of processes needs to be a power of 2
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
      ${MPIEXEC_POSTFLAGS} gemat11/gemat11.mtx --sp_matching 6)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
  endif()

endif()
