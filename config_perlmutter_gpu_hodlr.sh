#!/bin/bash
# Build ButterflyPACK's HODLR GPU backend and link it into STRUMPACK.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
module load PrgEnv-gnu cmake/3.30.2 cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export CRAYPE_LINK_TYPE=dynamic

libroot="${CFS}/m2957/lib/lib/PrgEnv-gnu"
bpack_install="${HODLR_BPACK_INSTALL:-$PWD/../ButterflyPACK-strumpack-gpu-support/install_hodlr_gpu}"
strumpack_build="${HODLR_STRUMPACK_BUILD:-$PWD/build_gpu_hodlr}"
build_jobs="${BUILD_JOBS:-8}"
export METIS_DIR="$libroot/parmetis-4.0.3/install"
export ParMETIS_DIR="$METIS_DIR"

# The dedicated builder preserves tracked source bytes around ButterflyPACK's
# precision preprocessing and installs only the library and its interfaces.
# Set HODLR_SKIP_BPACK=1 to use an existing HODLR_BPACK_INSTALL instead.
if [[ "${HODLR_SKIP_BPACK:-0}" != 1 ]]; then
  HODLR_MAGMA_DIR="${HODLR_MAGMA_DIR:-${HODLR_MAGMA_ROOT:-${CFS}/m2957/lib/magma_v2.10.0}}" \
    BUILD_JOBS="$build_jobs" bash ./build_butterflypack_hodlr_gpu.sh
fi

cmake -S . -B "$strumpack_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
  -DSTRUMPACK_USE_MPI=ON -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_CUDA=OFF \
  -DSTRUMPACK_USE_HIP=OFF -DSTRUMPACK_USE_SYCL=OFF \
  -DSTRUMPACK_COUNT_FLOPS=ON \
  -DTPL_ENABLE_BPACK=ON -DSTRUMPACK_REQUIRE_BPACK_GPU=ON \
  -DButterflyPACK_DIR="$bpack_install/lib64/cmake/ButterflyPACK" \
  -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_MAGMA=OFF -DTPL_ENABLE_KBLAS=OFF \
  -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF \
  -DTPL_ENABLE_PARMETIS=ON \
  -DMETIS_DIR="$METIS_DIR" -DParMETIS_DIR="$ParMETIS_DIR" \
  -DTPL_BLAS_LIBRARIES="$libroot/OpenBLAS_sequential/build/install/lib/libopenblas.so" \
  -DTPL_LAPACK_LIBRARIES="$libroot/OpenBLAS_sequential/build/install/lib/libopenblas.so" \
  -DTPL_SCALAPACK_LIBRARIES="$libroot/scalapack-2.2.0/build/install/lib/libscalapack.so"
cmake --build "$strumpack_build" --target testPoisson3dMPIDist \
  test_HODLR_gpu_interface_mpi test_HODLR_process_tree_mpi -j "$build_jobs"
