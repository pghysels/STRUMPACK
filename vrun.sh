# !/bin/bash

# module switch PrgEnv-intel PrgEnv-gnu
# module load python/2.7-anaconda-5.2
# python2 script_openTuner.py --test-limit 100

export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

FOLDER="/home/administrator//Desktop/research/ButterflyPACK_noarpack/EXAMPLE/KRR_DATA"
dim=8
# FOLDER="/global/homes/g/gichavez/cori/mats/ml_data/SUSY/SUSY_d18_1M_100k_100k"
# # FOLDER="/global/homes/g/gichavez/cori/mats/ml_data/SUSY/SUSY_d18_4M_500k_500k"
# dim=18

if [ ! -d "$FOLDER" ]; then
  echo "FOLDER directory not found!"
  exit
fi
DATA=$FOLDER/susy_10Kn

EXEC="/home/administrator/Desktop/research/STRUMPACK_master/build/examples/KernelRegressionMPI"
if [ ! -f ${EXEC} ]; then
    echo "EXEC file not found!"
    exit
fi

# Kernel and dimension
# kernel=Gauss
kernel=ANOVA

# Compression parameters
LEAF=128
k_ann=128

# Running parameters
hval=1.0
lval=3.0
# kernel degree
p=1

MAX_RANK=10001
MPIRUN="mpirun -n 2 "

RTOL=1e-2
${MPIRUN} ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${p} ${kernel} test \
--hss_rel_tol ${RTOL} \
--hss_abs_tol 1e-20 \
--hss_max_rank ${MAX_RANK} \
--hss_clustering_algorithm cobble \
--hss_leaf_size ${LEAF} \
--hss_dd ${k_ann} \
--hss_approximate_neighbors ${k_ann} \
--hss_ann_iterations 10 \
--hss_p 1 \
--hss_d0 1 \
--hss_scratch_folder ${FOLDER} \
--hss_verbose \
--hodlr_rel_tol ${RTOL} \
--hodlr_geo 0 \
--hodlr_knn_hodlrbf ${k_ann} \
2>&1 | tee z_log_run_1e-2.out


# exit

# for RTOL in 1e-2 
# do
  # echo ${EXEC}
  # echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}
  # ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${p} ${kernel} test \
  # --hss_rel_tol ${RTOL} \
  # --hss_abs_tol 1e-20 \
  # --hss_max_rank ${MAX_RANK} \
  # --hss_clustering_algorithm cobble \
  # --hss_leaf_size ${LEAF} \
  # --hss_dd ${k_ann} \
  # --hss_approximate_neighbors ${k_ann} \
  # --hss_ann_iterations 10 \
  # --hss_p 1 \
  # --hss_d0 1 \
  # --hss_scratch_folder ${FOLDER} \
  # --hss_verbose \
  # 2>&1 | tee z_log_run_all.out
# done
