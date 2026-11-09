#!/bin/sh
set -e

export ROOT_DIR="$PWD"
export DATA_FOLDER=$ROOT_DIR/examples/sparse/data
export TESTS_FOLDER=$ROOT_DIR/build/test
cd $TESTS_FOLDER

export GREEN="\033[32;1m"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Matrices downloaded by test/download_mtx.sh (into $TESTS_FOLDER).
DOWNLOADED_MTX="utm300 mesh3e1 t2dal bcsstk28 cavity16 bcsstm08 sherman4 rdb968 gemat11"
SKIPPED=0
SKIPPED_MTX=""

# run_mtx [VAR=value ...] <command> [args...]
# Runs the command (through env, so it can be prefixed with environment
# variable assignments), unless one of its arguments is a downloaded
# test matrix that is not available: then the command is skipped and
# counted. A failing command still fails this script (set -e).
run_mtx() {
    for arg in "$@"; do
        for m in $DOWNLOADED_MTX; do
            if [ "$arg" = "$m/$m.mtx" ] && [ ! -s "$arg" ]; then
                printf "SKIP (matrix %s not available): %s\n" "$arg" "$*"
                SKIPPED=$((SKIPPED + 1))
                case " $SKIPPED_MTX " in
                    *" $m "*) ;;
                    *) SKIPPED_MTX="$SKIPPED_MTX $m" ;;
                esac
                return 0
            fi
        done
    done
    env "$@"
}

if [ $TEST_NUMBER -eq 3 ] || [ $TEST_NUMBER -eq 4 ] || [ $TEST_NUMBER -eq 5 ]
then
        printf "${GREEN} ###GC: Downloading sparse test matrices\n\n\n"
        # never fails the job: tests whose matrix is missing are skipped
        sh "$ROOT_DIR/test/download_mtx.sh" || true
fi

if [ $TEST_NUMBER -eq 0 ]
then
        printf "${GREEN} Running: basic tests"
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_HSS_mpi T 100
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_sparse_mpi $DATA_FOLDER/pde900.mtx
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2 ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 8 ./test_trsm_narrow_mpi
fi

if [ $TEST_NUMBER -eq 1 ]
then
        printf "${GREEN} Running: test_HSS_seq"
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
        printf "${GREEN} Running: test_HSS_mpi"
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
        printf "${GREEN} Running: test_sparse_seq"
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq t2dal/t2dal.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq bcsstk28/bcsstk28.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-4 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=3 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=8 ./test_sparse_seq rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
fi

if [ $TEST_NUMBER -eq 4 ]
then
        printf "${GREEN} Running: test_sparse_mpi"
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi bcsstm08/bcsstm08.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 2   ./test_sparse_mpi sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 16  ./test_sparse_mpi sherman4/sherman4.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 17  ./test_sparse_mpi utm300/utm300.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 4   ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method metis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 9   ./test_sparse_mpi mesh3e1/mesh3e1.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_sparse_mpi rdb968/rdb968.mtx --sp_compression HSS --hss_leaf_size 4 --hss_rel_tol 1e-10 --hss_abs_tol 1e-10 --hss_d0 16 --hss_dd 8 --sp_reordering_method parmetis --sp_compression_min_sep_size 25
fi

if [ $TEST_NUMBER -eq 5 ]
then
        printf "${GREEN} Running: test_structure_reuse_mpi"
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 0
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 2
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 3
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 4
        run_mtx OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi utm300/utm300.mtx --sp_compression NONE --sp_matching 5
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 0 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 2 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 3 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 4 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
        OMP_NUM_THREADS=1 mpirun --oversubscribe -n 19  ./test_structure_reuse_mpi $DATA_FOLDER/pde900.mtx --sp_compression HSS --sp_matching 5 --hss_leaf_size 4 --hss_rel_tol 1e-3 --hss_abs_tol 1e-10 --sp_compression_min_sep_size 25
fi

if [ $SKIPPED -gt 0 ]
then
        msg="Skipped $SKIPPED test(s): test matrices could not be downloaded:$SKIPPED_MTX"
        printf "\n%s\n" "$msg"
        # shown as an annotation on the GitHub Actions run
        if [ -n "$GITHUB_ACTIONS" ]; then
            echo "::warning title=Test matrices missing::$msg"
        fi
fi
