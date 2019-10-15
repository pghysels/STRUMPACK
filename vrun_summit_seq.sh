# !/bin/bash

# module switch PrgEnv-intel PrgEnv-gnu
# module load python/2.7-anaconda-5.2
# python2 script_openTuner.py --test-limit 100

RANK_PER_RS=1
GPU_PER_RANK=0	




# FOLDER="/gpfs/alpine/scratch/liuyangz/csc289/ButterflyPACK/EXAMPLE/KRR_DATA"
# dim=8
FOLDER="/gpfs/alpine/csc289/scratch/liuyangz/ML/mats/checkCERR/scikit"
#FOLDER="/gpfs/alpine/csc289/scratch/liuyangz/ML/mats/SUSY/"


 dim=8
 DATA0=susy_d8_10k_1k/susy
 hval=1.3
 lval=3.11


# dim=16
# DATA0=letter_d16_9K_1K_1K/letter
# hval=0.60
# lval=4.83


# dim=16
# DATA0=pendigits_d16_9K_1K_1K/pendigits
# hval=0.94
# lval=0.732

# dim=27
# DATA0=hepmass_d27_10K_1K_1K/hepmass
# hval=3.54
# lval=4.28

# dim=54
# DATA0=covtype_d54_10K_1K_1K/covtype
# hval=1.89
# lval=5.85

#dim=128
#DATA0=gas_d128_10K_1K_1K/gas
#hval=1.25
#lval=2.24

# dim=784
# DATA0=mnist_d784_10K_1K_1K/mnist
# hval=5.77
# lval=0.211


DATA=$FOLDER/$DATA0
mkdir -p $DATA0

EXEC="/gpfs/alpine/scratch/liuyangz/csc289/STRUMPACK_master/build/examples/KernelRegression"
if [ ! -f ${EXEC} ]; then
    echo "EXEC file not found!"
    exit
fi



# Kernel and dimension
kernel=Gauss
#kernel=ANOVA

# Compression parameters
LEAF=128
k_ann=128
lr_leaf=5

# Running parameters
# hval=1.0
# lval=3.0
# kernel degree
p=1

MAX_RANK=10001

RTOL=1e-2



# for CORE_VAL in 16 18 32 50 64 98 128 200 256 512 1024
#for CORE_VAL in  512 
for CORE_VAL in  1
# for CORE_VAL in  32 64
do

NTH=40
RS_VAL=`expr $CORE_VAL / $RANK_PER_RS`
MOD_VAL=`expr $CORE_VAL % $RANK_PER_RS`
if [[ $MOD_VAL -ne 0 ]]
then
  RS_VAL=`expr $RS_VAL + 1`
fi
OMP_NUM_THREADS=$NTH
TH_PER_RS=`expr $NTH \* $RANK_PER_RS`
GPU_PER_RS=`expr $RANK_PER_RS \* $GPU_PER_RANK`


export OMP_NUM_THREADS=$OMP_NUM_THREADS

jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${p} ${kernel} test \
--hss_rel_tol ${RTOL} \
--hss_abs_tol 1e-20 \
--hss_max_rank ${MAX_RANK} \
--hss_clustering_algorithm cobble \
--hss_leaf_size ${LEAF} \
--hss_dd ${k_ann} \
--hss_approximate_neighbors ${k_ann} \
--hss_ann_iterations 50 \
--hss_scratch_folder ${FOLDER} \
--hss_verbose \
--hodlr_rel_tol ${RTOL} \
--hodlr_geo 0 \
--hodlr_knn_hodlrbf ${k_ann} \
--hodlr_lr_leaf ${lr_leaf} | tee $DATA0/log_run_1e-2_ann_history.out


done
