#!/bin/sh
set -e

export GREEN="\033[32;1m"
export TESTS_FOLDER=$TRAVIS_BUILD_DIR/build/test
cd $TESTS_FOLDER

if [ $TEST_NUMBER -eq 3 ] || [ $TEST_NUMBER -eq 4 ]
then
	printf "${GREEN} ###GC: Downloading sparse test matrices\n\n\n"

	wget http://portal.nersc.gov/project/sparse/strumpack/test_matrices.tar.gz
	tar -xvzf test_matrices.tar.gz
	rm test_matrices.tar.gz
fi

if [ $TEST_NUMBER -eq 1 ]
then
	printf "{GREEN} Running: test_HSS_seq"
	OMP_NUM_THREADS=3 ./test_HSS_seq L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=3 ./test_HSS_seq L 200 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=8 ./test_HSS_seq L 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=8 ./test_HSS_seq T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 ./test_HSS_seq U 1 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=8 ./test_HSS_seq U 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 200 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 ./test_HSS_seq L 10 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=3 ./test_HSS_seq L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=8 ./test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=8 ./test_HSS_seq L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=3 ./test_HSS_seq U 500 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=1 ./test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=8 ./test_HSS_seq U 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 10 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 ./test_HSS_seq L 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=1 ./test_HSS_seq U 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=3 ./test_HSS_seq T 500 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
fi

if [ $TEST_NUMBER -eq 2 ]
then
	printf "{GREEN} Running: test_HSS_mpi"
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi L 10 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi U 10 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi L 1 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi T 200 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi T 10 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi T 10 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi U 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi T 1 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi L 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi U 200 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi L 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi T 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi L 10 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_HSS_mpi T 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_HSS_mpi L 10 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi T 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi L 200 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi U 1 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi U 500 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi T 500 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi T 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi T 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_HSS_mpi U 500 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi T 1 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_HSS_mpi U 10 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_HSS_mpi U 500 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_HSS_mpi L 1 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8 
fi

if [ $TEST_NUMBER -eq 3 ]
then
	printf "{GREEN} Running: test_sparse_seq"
	OMP_NUM_THREADS=1 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=1 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=3 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
	OMP_NUM_THREADS=8 ./test_sparse_seq rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25
fi

if [ $TEST_NUMBER -eq 4 ]
then
	printf "${GREEN} Running: test_sparse_mpi"
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-2 --hss_abs_tol 1e-2 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_sparse_mpi m mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_sparse_mpi m cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m bcsstm08/bcsstm08.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_sparse_mpi m sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
#	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_sparse_mpi m sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_sparse_mpi m sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_sparse_mpi m cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 17 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_sparse_mpi m sherman4/sherman4.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 9 ./test_sparse_mpi m mesh3e1/mesh3e1.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m cavity16/cavity16.mtx --sp_enable_hss --sp_enable_replace_tiny_pivots --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 16 ./test_sparse_mpi m t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method scotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 13 ./test_sparse_mpi m utm300/utm300.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m rdb968/rdb968.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 4 ./test_sparse_mpi m t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 2 ./test_sparse_mpi m t2dal/t2dal.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
	OMP_NUM_THREADS=1 mpirun -n 19 ./test_sparse_mpi m bcsstk28/bcsstk28.mtx --sp_enable_hss --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method ptscotch --sp_hss_min_sep_size 25 
fi
