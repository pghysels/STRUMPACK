#!/bin/sh
set -e

export ROOT_DIR="$PWD"
export DATA_FOLDER=$ROOT_DIR/examples/sparse/data
export TESTS_FOLDER=$ROOT_DIR/build/test
cd $TESTS_FOLDER

export GREEN="\033[32;1m"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

if [ $TEST_NUMBER -eq 3 ] || [ $TEST_NUMBER -eq 4 ] || [ $TEST_NUMBER -eq 5 ]
then
        printf "${GREEN} ###GC: Preparing sparse test matrices\n\n\n"

        # This archive predates the SuiteSparse download outage and contains
        # every matrix used below. Keep a verified copy outside the build tree.
        matrix_cache="${XDG_CACHE_HOME:-$HOME/.cache}/strumpack"
        matrix_archive="$matrix_cache/test_matrices.tar.gz"
        matrix_sha256=b4c383f1d99a3837aaac43ed110e47aad8fa90771163e3f7dad910ae66b99d67
        if [ ! -f "$matrix_archive" ]; then
            mkdir -p "$matrix_cache"
            wget --timeout=30 --tries=3 -O "$matrix_archive.tmp" \
                https://portal.nersc.gov/project/sparse/strumpack/test_matrices.tar.gz
            printf '%s  %s\n' "$matrix_sha256" "$matrix_archive.tmp" | sha256sum -c -
            mv "$matrix_archive.tmp" "$matrix_archive"
        fi
        printf '%s  %s\n' "$matrix_sha256" "$matrix_archive" | sha256sum -c -
        tar -xzf "$matrix_archive" \
            utm300/utm300.mtx mesh3e1/mesh3e1.mtx t2dal/t2dal.mtx \
            bcsstk28/bcsstk28.mtx bcsstm08/bcsstm08.mtx \
            sherman4/sherman4.mtx rdb968/rdb968.mtx
fi

if [ $TEST_NUMBER -eq 0 ]
then
        printf "{GREEN} Running: basic tests"
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_HSS_mpi T 100
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_sparse_mpi $DATA_FOLDER/pde900.mtx
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx
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
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi L 500 --hss_leaf_size 8 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi L 10 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi T 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi U 10 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi L 1 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi T 200 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi T 10 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi T 10 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi U 200 --hss_leaf_size 1 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi T 1 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi L 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi U 200 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi L 1 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi T 1 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi T 200 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi L 10 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_HSS_mpi T 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 13  ./test_HSS_mpi L 10 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi T 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 64 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi L 200 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi L 500 --hss_leaf_size 16 --hss_rel_tol 1e-10 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi T 500 --hss_leaf_size 16 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi T 1 --hss_leaf_size 128 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi T 200 --hss_leaf_size 1 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi U 1 --hss_leaf_size 128 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi L 10 --hss_leaf_size 3 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi U 1 --hss_leaf_size 3 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi U 500 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi T 500 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi L 1 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi T 10 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi T 200 --hss_leaf_size 16 --hss_rel_tol 1e-1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 32 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 500 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm stable --hss_d0 32 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_HSS_mpi U 500 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi T 1 --hss_leaf_size 16 --hss_rel_tol 1e-5 --hss_abs_tol 1e-13 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 500 --hss_leaf_size 128 --hss_rel_tol 1 --hss_abs_tol 1e-13 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_HSS_mpi U 10 --hss_leaf_size 1 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_HSS_mpi U 500 --hss_leaf_size 3 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm stable --hss_d0 64 --hss_dd 8
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_HSS_mpi L 1 --hss_leaf_size 1 --hss_rel_tol 1e-5 --hss_abs_tol 1e-10 --hss_enable_sync --hss_compression_algorithm original --hss_d0 128 --hss_dd 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_HSS_mpi T 200 --hss_leaf_size 3 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_disable_sync --hss_compression_algorithm original --hss_d0 16 --hss_dd 8
fi

if [ $TEST_NUMBER -eq 3 ]
then
        printf "{GREEN} Running: test_sparse_seq"
        OMP_NUM_THREADS=1 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=3 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=8 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
fi

if [ $TEST_NUMBER -eq 4 ]
then
        printf "${GREEN} Running: test_sparse_mpi"
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_sparse_mpi sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_sparse_mpi sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_sparse_mpi utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
fi

if [ $TEST_NUMBER -eq 5 ]
then
        printf "${GREEN} Running: test_structure_reuse_mpi"
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 0
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 2
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 3
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 4
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 5
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 0 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 2 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 3 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 4 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 5 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
fi
