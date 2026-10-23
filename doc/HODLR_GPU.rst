Hybrid ButterflyPACK HODLR GPU interface
======================================

This feature branch is based on the shared-process-tree change (217972ad)
and retains upstream's multi-RHS packing correction (8937587f).

Build and dependencies
----------------------

Use ButterflyPACK's gpu_hodlr branch with enable_h2_gpu=ON, CUDA and MAGMA.
The companion checkout ../ButterflyPACK-strumpack-gpu-support, branch
fix/strumpack-gpu-integration (8d8b38b), contains the noncollective device-selection
fix and the CMake package exports needed by a STRUMPACK build with its own
CUDA dense-front backend disabled. Independent fronts must never trigger a
hidden collective on MPI_COMM_WORLD. The existing validated installation in
../ButterflyPACK_gpu_hodlr/install_hodlr_gpu already contains these changes.

For Perlmutter, run bash config_perlmutter_gpu_hodlr.sh. To reuse that existing
installation instead of rebuilding ButterflyPACK::

  HODLR_SKIP_BPACK=1 \
  HODLR_BPACK_INSTALL="$PWD/../ButterflyPACK_gpu_hodlr/install_hodlr_gpu" \
  bash config_perlmutter_gpu_hodlr.sh

CMake supports both the combined ButterflyPACK::butterflypack target and
older precision-specific targets. It compiles AND links a probe referencing
all six d_/z_ availability, device-sharing and exchange-initialization
symbols. STRUMPACK_REQUIRE_BPACK_GPU=ON makes a missing GPU ABI a configure
error. Exported CPU stubs are distinguished from a GPU-enabled library at
runtime. Older CPU installations continue to work without GPU symbols.

Selection and lifetime
----------------------

Set --sp_compression HODLR --hodlr_use_gpu 1 --hodlr_butterfly_levels 0.
Mode 0 is the default CPU path; mode 2 selects FP64 tensor-core GEMMs.
Double and double-complex arithmetic are supported. This initial interface
requires OMP_NUM_THREADS=1 and element-extraction construction. Randomized
matvec construction is rejected rather than silently factoring on CPU.
ButterflyPACK's HODLR dense-block ZFP compression is not enabled here.

Before numerical front traversal, every solver rank collectively initializes
device sharing and the exchange arena on the solver communicator. Explicit
standalone HODLRMatrix users must call HODLR_initialize_gpu<T>(comm) before
starting independent matrix work. Rank/GPU binding should initially use one
MPI rank per GPU. Multiple ranks sharing a GPU and expanding solver groups
are not claimed as validated by this interface: the backend's sharing cache
is process-wide and initialized on first use.

F11/F22 and rectangular F12/F21 retain shared process-tree ownership within
each front; independent fronts retain independent communication contexts.
The host callbacks assemble original sparse entries and child Schur
contributions. ButterflyPACK compresses them on CPU and uploads the result,
then owns HODLR GPU factors, multiplies, extraction and inverse applications.
Rectangular Butterfly operations stay on CPU. Existing packed-RHS conversion
and distributed 2D/1D redistribution remain in place.

Execution information
---------------------

--hodlr_verbose prints per-operation API wall times and backend selection.
HODLRMatrix::backend_statistics() exposes cumulative local-rank construction,
factor, multiply and inverse times and RHS boundary bytes. The construction
time includes host compression and its upload, not a separately measured GPU
construction phase. RHS bytes count the input/output payload crossing the
host API on GPU inverse/multiply calls; they exclude internal MPI, metadata,
allocation and factor transfers. They are not a full PCIe traffic counter.

GPU reports include resident matrix factor bytes from ButterflyPACK and
CUDA device-wide used/total memory. The latter includes all processes and
pool reservations; it is not a per-matrix or per-process peak. The report is
emitted on rank zero of each matrix communicator, with local statistics.
Pure kernel time and separate upload/download timing require additional
ButterflyPACK instrumentation; unavailable counters are not reported as zero.

Next extension: GPU frontal evaluation
-------------------------------------

No device evaluator is registered by the hybrid sparse-front path. Device
construction is a separate extension, not implemented by copying a Poisson
stencil. It must register a block evaluator after construct_init and before
construct_element_compute. That evaluator must map ButterflyPACK's zero-based
front ids through separator/update permutations, assemble original entries
AND child Schur complements, and maintain child/device data until all queued
work completes. It must use the supplied CUDA stream and scratch allocator.

Construction also needs compatible BACA settings (RecLR_leaf 4 or 5,
LR_BLK_NUM=1, forwardN15flag=0). The later work should separately address
front-data residency, distributed child extraction, ownership and stream
ordering before enabling device construction. The hybrid path remains the
correctness reference.

Focused regression
------------------

The new test_HODLR_gpu_interface_mpi checks real/complex 1/3-RHS strided
inverse applications, unchanged padding, timings and RHS accounting.
--cold-subcomm with nonzero --hodlr_use_gpu makes only rank zero perform
GPU work on MPI_SELF while other ranks wait at a world barrier; invoke it
under a timeout to catch hidden world collectives.

Previous Poisson GPU numerical evidence is preserved in the original
STRUMPACK checkout's hodlr_rhs_gpu_fix_report.txt (jobs 60043836/60043857).
Large Poisson validation is not repeated for this consolidation.
