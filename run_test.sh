#!/bin/bash
#SBATCH -N 64
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 00:30:00


module swap PrgEnv-intel PrgEnv-gnu
module unload cmake
module load cmake
module rm darshan



#OpenMP settings:
export OMP_PLACES=threads
export OMP_PROC_BIND=spread



# blas=openblas
blas=libsci
# blas=netlibblas


# module unload darshan-runtime

# exe=/project/projectdirs/m2957/liuyangz/my_research/STRUMPACK_thai_3D/build_$blas/examples/sparse/testMMdouble

# ##### MPI code shows wrong rank for openblas?? 
exe=/project/projectdirs/m2957/liuyangz/my_research/STRUMPACK_thai_3D/build_$blas/examples/sparse/testMMdoubleMPIDist




# DIM=2
# inputdir=/project/projectdirs/m2957/liuyangz/my_research/matlab_direct_solvers_${DIM}DRD/SPDE${DIM}D
# geo=Uniform_4096

DIM=3
inputdir=/project/projectdirs/m2957/liuyangz/my_research/matlab_direct_solvers_${DIM}DRD/SPDE${DIM}D
geo=Uniform_350



out=Jun8
mkdir -p $out


tol=1e-2


knnhodlrbf=64
knnlrbf=128
sample_param=2.0
hodlr_butterfly_levels=100

# ulimit -c unlimited
nmpi=128
nthread=16
THREADS_PER_RANK=`expr 2 \* $nthread`

#for n in 100 125 150 175 200; do
# for eps in 0.0001; do
# for eps in 1 0.01 0.0001 1e-06 1e-08; do
# for eps in 0.01; do
for eps in 0.0001; do

inputmat=${inputdir}/${geo}_${eps}/${geo}_${eps}.mtx.bin
inputrhs=${inputdir}/${geo}_${eps}/${geo}_${eps}_b.mtx
inputxexact=${inputdir}/${geo}_${eps}/${geo}_${eps}_x.mtx



    export OMP_NUM_THREADS=$nthread

    # srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression NONE --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 --sp_print_root_front_stats --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_NONE_fastmath_${blas}.log
    
    # sep=100
    # front=300
    # comp=LOSSY
    # srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp \
    # --sp_print_root_front_stats --sp_enable_METIS_NodeNDP \
    # --sp_compression_min_front_size ${front} --sp_compression_min_sep_size ${sep} --blr_rel_tol ${tol} --help \
    # | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sep}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}.log


    # sep=100
    # front=300
    # comp=HSS
    # srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp \
    # --sp_print_root_front_stats --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 --hss_leaf_size 64 \
    # --sp_compression_min_front_size ${front} --sp_compression_min_sep_size ${sep} --hss_rel_tol ${tol} \
    # --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sep}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}.log

    # blr_cb=COLWISE
    # blr_leaf_size=256 
    # sep=100
    # front=300
    # comp=BLR
    # srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp \
    # --blr_factor_algorithm $blr_cb --blr_leaf_size $blr_leaf_size --blr_seq_CB_Compression --sp_proportional_mapping FACTOR_MEMORY --sp_print_root_front_stats --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 \
    # --sp_compression_min_front_size ${front} --sp_compression_min_sep_size ${sep} --blr_rel_tol ${tol} --help \
    # | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sep}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}.log 


    # sep=600
    # front=5000
    # comp=HODLR
    # srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp \
    # --sp_print_root_front_stats --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 --hodlr_rel_tol ${tol} \
    # --sp_compression_min_front_size ${front} --sp_compression_min_sep_size ${sep} \
    # --hodlr_butterfly_levels ${hodlr_butterfly_levels} --sp_maxit 1000 --hodlr_quiet \
    # --hodlr_knn_hodlrbf ${knnhodlrbf} --hodlr_knn_lrbf ${knnlrbf} --hodlr_BF_sampling_parameter ${sample_param} \
    # --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sep}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}_hodlr_butterfly_levels_${hodlr_butterfly_levels}.log


    sepBLR=100
    sepHODLR=50000000
    # sepHODLR=5000
    frontHODLR=50000000
    front=300
    comp=ZFP_BLR_HODLR
    blr_cb=COLWISE 
    blr_leaf_size=256

    knnhodlrbf=256
    knnlrbf=64

    # knnhodlrbf=0
    # knnlrbf=0
    sample_param=1.2
    tol=1e-2
    tol2=1e-2
    rankrate=1.2
    srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp\
    --blr_factor_algorithm $blr_cb --blr_seq_CB_Compression --sp_proportional_mapping FACTOR_MEMORY \
    --sp_print_root_front_stats --hodlr_rel_tol ${tol} \
    --blr_leaf_size $blr_leaf_size \
    --sp_compression_min_front_size ${front}\
    --sp_blr_min_sep_size ${sepBLR} --blr_rel_tol ${tol2}\
    --sp_hodlr_min_sep_size ${sepHODLR} --sp_hodlr_min_front_size ${frontHODLR}  --hodlr_quiet --hodlr_rank_rate ${rankrate} --hodlr_max_rank 1000\
    --hodlr_butterfly_levels 100 --sp_maxit 1000 \
    --hodlr_knn_hodlrbf ${knnhodlrbf} --hodlr_knn_lrbf ${knnlrbf} --hodlr_BF_sampling_parameter ${sample_param} --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 \
    --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sepBLR}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}_sepHODLR${sepHODLR}_frontHODLR_${frontHODLR}.log 




    sepBLR=100
    # sepHODLR=50000000
    sepHODLR=5000
    frontHODLR=50000000
    front=300
    comp=ZFP_BLR_HODLR
    blr_cb=COLWISE 
    blr_leaf_size=256

    knnhodlrbf=256
    knnlrbf=64

    # knnhodlrbf=0
    # knnlrbf=0
    sample_param=1.2
    tol=1e-2
    tol2=1e-2
    rankrate=1.2
    srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp\
    --blr_factor_algorithm $blr_cb --blr_seq_CB_Compression --sp_proportional_mapping FACTOR_MEMORY \
    --sp_print_root_front_stats --hodlr_rel_tol ${tol} \
    --blr_leaf_size $blr_leaf_size \
    --sp_compression_min_front_size ${front}\
    --sp_blr_min_sep_size ${sepBLR} --blr_rel_tol ${tol2}\
    --sp_hodlr_min_sep_size ${sepHODLR} --sp_hodlr_min_front_size ${frontHODLR}  --hodlr_quiet --hodlr_rank_rate ${rankrate} --hodlr_max_rank 1000\
    --hodlr_butterfly_levels 100 --sp_maxit 1000 \
    --hodlr_knn_hodlrbf ${knnhodlrbf} --hodlr_knn_lrbf ${knnlrbf} --hodlr_BF_sampling_parameter ${sample_param} --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 \
    --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sepBLR}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}_sepHODLR${sepHODLR}_frontHODLR_${frontHODLR}.log 





    sepBLR=100
    # sepHODLR=50000000
    sepHODLR=5000
    frontHODLR=50000000
    front=300
    comp=BLR_HODLR
    blr_cb=COLWISE 
    blr_leaf_size=256

    knnhodlrbf=256
    knnlrbf=64

    # knnhodlrbf=0
    # knnlrbf=0
    sample_param=1.2
    tol=1e-2
    tol2=1e-2
    rankrate=1.2
    srun -n $nmpi -c $THREADS_PER_RANK --cpu_bind=cores $exe $inputmat $inputrhs $inputxexact --sp_compression $comp\
    --blr_factor_algorithm $blr_cb --blr_seq_CB_Compression --sp_proportional_mapping FACTOR_MEMORY \
    --sp_print_root_front_stats --hodlr_rel_tol ${tol} \
    --blr_leaf_size $blr_leaf_size \
    --sp_compression_min_front_size ${front}\
    --sp_blr_min_sep_size ${sepBLR} --blr_rel_tol ${tol2}\
    --sp_hodlr_min_sep_size ${sepHODLR} --sp_hodlr_min_front_size ${frontHODLR}  --hodlr_quiet --hodlr_rank_rate ${rankrate} --hodlr_max_rank 1000\
    --hodlr_butterfly_levels 100 --sp_maxit 1000 \
    --hodlr_knn_hodlrbf ${knnhodlrbf} --hodlr_knn_lrbf ${knnlrbf} --hodlr_BF_sampling_parameter ${sample_param} --sp_enable_METIS_NodeNDP --sp_rel_tol 1e-10 --sp_abs_tol 1e-15 \
    --help | tee ${out}/Thai_${DIM}D_geo_${geo}_eps${eps}_sep${sepBLR}_front${front}_t${tol}_nmpi_${nmpi}_nthread_${nthread}_${comp}_fastmath_${blas}_sepHODLR${sepHODLR}_frontHODLR_${frontHODLR}.log 






done
