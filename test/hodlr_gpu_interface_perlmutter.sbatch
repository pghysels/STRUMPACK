#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:05:00
#SBATCH -J hodlr_gpu_api
#SBATCH -o gpu_interface_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_AWARE_MPI=0 MPICH_GPU_SUPPORT_ENABLED=0
cd "$SLURM_SUBMIT_DIR"
out="gpu_interface_${SLURM_JOB_ID}"
mkdir -p "$out"
exe="$PWD/build_gpu_hodlr/test/test_HODLR_gpu_interface_mpi"
sha256sum "$exe" build_cpu_interface/test/test_HODLR_gpu_interface_mpi > "$out/executables.sha256"
git rev-parse HEAD > "$out/source_commit.txt"
for mode in 0 1 2; do
  BPACK_CHECK=hodlr timeout 90s srun -n 2 -c 4 --gpus-per-task=1 --gpu-bind=single:1 \
    "$exe" --hodlr_use_gpu "$mode" --hodlr_verbose > "$out/mode${mode}.log" 2>&1
  rg -q 'HODLR hybrid interface: PASS' "$out/mode${mode}.log"
done
BPACK_CHECK=hodlr timeout 90s srun -n 2 -c 4 --gpus-per-task=1 --gpu-bind=single:1 \
  "$exe" --hodlr_use_gpu 1 --cold-subcomm --hodlr_verbose > "$out/cold.log" 2>&1
rg -q 'HODLR hybrid interface: PASS' "$out/cold.log"
for ranks in 1 2; do
  srun -n "$ranks" -c 2 build_cpu_interface/test/test_HODLR_gpu_interface_mpi > "$out/cpu${ranks}.log" 2>&1
  rg -q 'HODLR hybrid interface: PASS' "$out/cpu${ranks}.log"
done
set +e
srun -n 1 -c 2 build_cpu_interface/test/test_HODLR_gpu_interface_mpi --hodlr_use_gpu 1 > "$out/cpu_rejects_gpu.log" 2>&1
status=$?
set -e
test "$status" -ne 0
rg -q 'GPU-enabled ButterflyPACK' "$out/cpu_rejects_gpu.log"
python3 test/check_hodlr_gpu_interface.py "$out"
