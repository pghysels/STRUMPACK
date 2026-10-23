#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
module load PrgEnv-gnu cmake/3.30.2 cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export CRAYPE_LINK_TYPE=dynamic
export HODLR_MAGMA_DIR="${HODLR_MAGMA_DIR:-/global/cfs/cdirs/m2957/lib/magma_v2.10.0}"
python3 - <<'PY'
from pathlib import Path
import os, subprocess
source=(Path.cwd()/'../ButterflyPACK-strumpack-gpu-support').resolve()
build=source/'build_hodlr_gpu'
install=source/'install_hodlr_gpu'
libroot=Path(os.environ['CFS'])/'m2957/lib/lib/PrgEnv-gnu'
magma=Path(os.environ['HODLR_MAGMA_DIR'])
# The upstream precision generator also strips whitespace in source files.
# Preserve all tracked source bytes, including the user's existing changes.
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=str(source)).split(b'\0')
snapshot={source/os.fsdecode(p):(source/os.fsdecode(p)).read_bytes()
          for p in tracked if p and (source/os.fsdecode(p)).is_file()}
args=['cmake','-S',str(source),'-B',str(build),
      '-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON',
      '-DCMAKE_Fortran_COMPILER=ftn','-DCMAKE_CXX_COMPILER=CC','-DCMAKE_C_COMPILER=cc',
      '-DCMAKE_Fortran_FLAGS=-DMPIMODULE','-DCMAKE_INSTALL_PREFIX='+str(install),
      '-Denable_mpi=ON','-Denable_openmp=ON','-Denable_toplevel_openmp=OFF',
      '-Denable_h2_gpu=ON','-DCMAKE_CUDA_ARCHITECTURES=80',
      '-Denable_python=OFF','-Denable_parsec=OFF','-Denable_fftw=OFF',
      '-Denable_examples=OFF',
      '-DTPL_H2_MAGMA_INCLUDE_DIRS='+str(magma/'include'),
      '-DTPL_H2_MAGMA_LIBRARIES='+str(magma/'lib/libmagma.so'),
      '-DTPL_BLAS_LIBRARIES='+str(libroot/'OpenBLAS_sequential/build/install/lib/libopenblas.so'),
      '-DTPL_LAPACK_LIBRARIES='+str(libroot/'OpenBLAS_sequential/build/install/lib/libopenblas.so'),
      '-DTPL_SCALAPACK_LIBRARIES='+str(libroot/'scalapack-2.2.0/build/install/lib/libscalapack.so')]
try:
    subprocess.check_call(args)
finally:
    restored=0
    for p,data in snapshot.items():
        if p.read_bytes()!=data:
            p.write_bytes(data)
            restored+=1
    print('Restored configure-time changes to',restored,'tracked source files.')
subprocess.check_call(['cmake','--build',str(build),'--target','butterflypack','-j',os.environ.get('BUILD_JOBS','8')])
subprocess.check_call(['cmake','--install',str(build)])
PY
