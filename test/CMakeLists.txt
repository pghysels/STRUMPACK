add_custom_target(tests)

add_executable(test_HSS_seq    EXCLUDE_FROM_ALL test_HSS_seq.cpp)
add_executable(test_sparse_seq EXCLUDE_FROM_ALL test_sparse_seq.cpp)
add_executable(test_BLR_seq    EXCLUDE_FROM_ALL test_BLR_seq.cpp)

target_link_libraries(test_HSS_seq strumpack)
target_link_libraries(test_sparse_seq strumpack)
target_link_libraries(test_BLR_seq strumpack)

add_dependencies(tests
  test_HSS_seq
  test_sparse_seq
  test_BLR_seq)


add_test("user_test_HSS_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_seq T 100)
add_test("user_test_sparse_seq" ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq
  ${PROJECT_SOURCE_DIR}/examples/data/pde900.mtx)

if(STRUMPACK_USE_MPI)
  add_executable(test_HSS_mpi    EXCLUDE_FROM_ALL test_HSS_mpi.cpp)
  add_executable(test_sparse_mpi EXCLUDE_FROM_ALL test_sparse_mpi.cpp)

  target_link_libraries(test_HSS_mpi strumpack)
  target_link_libraries(test_sparse_mpi strumpack)

  add_dependencies(tests
    test_HSS_mpi
    test_sparse_mpi)

  # TODO check whether this is supported?
  set(OVERSUBSCRIBEFLAG "--oversubscribe")

  add_test("user_test_HSS_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi T 100)
  add_test("user_test_sparse_mpi" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG}
    ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi m
    ${PROJECT_SOURCE_DIR}/examples/data/pde900.mtx)
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

  set(test_name "HSS_mpi_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} L 10 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4)
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

  set(test_name "HSS_mpi_37")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
    ${MPIEXEC_POSTFLAGS} T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4)
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

endif()


add_test(NAME "Download_sparse_test_matrices" COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/test/download_mtx.sh)


set(test_name "SPARSE_seq_1")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_2")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_3")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_4")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_5")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_6")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_7")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_8")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_9")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_10")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_11")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_12")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_13")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_14")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_15")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_16")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_17")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_18")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_19")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_20")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_21")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_22")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_23")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_24")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_25")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_26")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_27")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_28")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_29")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_30")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_31")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_32")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_33")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_34")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_35")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_36")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_37")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_38")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_39")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_40")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_41")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_42")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_42")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_43")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_44")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")

set(test_name "SPARSE_seq_45")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

set(test_name "SPARSE_seq_46")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

set(test_name "SPARSE_seq_47")
add_test(${test_name} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=8")


if(STRUMPACK_USE_MPI)

  set(test_name "SPARSE_mpi_1")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-2 --hss_abs_tol 1e-2 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_2")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_3")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_4")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_5")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_6")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_7")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_8")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_9")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_10")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_11")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_12")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_13")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_14")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_15")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_16")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_17")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 17 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_18")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_19")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_20")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_21")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_22")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_23")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m cavity16/cavity16.mtx --sp_compression HSS --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_24")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_25")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_26")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_27")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_28")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_29")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_30")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

  set(test_name "SPARSE_mpi_31")
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 19 ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
    ${MPIEXEC_POSTFLAGS} m bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

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

  #set(test_output ${CMAKE_BINARY_DIR}/test/HSS_seq.out)
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
      #set(CTEST_ENVIRONMENT "OMP_NUM_THREADS=${th}")
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
    #set(test_output ${CMAKE_BINARY_DIR}/test/HSS_mpi.out)
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
          ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
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
          ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_HSS_mpi
          ${MPIEXEC_POSTFLAGS} ${m} ${n} --hss_leaf_size ${l}
          --hss_rel_tol ${t} --hss_abs_tol ${a} --hss_${s}_sync
          --hss_compression_algorithm ${c} --hss_d0 ${d}
          --hss_dd ${deltad})
        set_property(TEST ${test_name} PROPERTY
          ENVIRONMENT "OMP_NUM_THREADS=${th}")
      endif()
    endforeach()
  endif()

  ## Sequential sparse test
  #set(test_output ${CMAKE_BINARY_DIR}/test/SPARSE_seq.out)
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
    #set(test_output ${CMAKE_BINARY_DIR}/test/SPARSE_mpi.out)
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
          ${MPIEXEC_PREFLAGS} ${OVERSUBSCRIBEFLAG} ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_mpi
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
