#!/usr/bin/env bash

# DATA="/Users/gichavez/Documents/mats/SUSY1/susy_train_10K_test_1K/susy_10Kn"
# DATA="/Users/gichavez/Documents/mats/SUSY1/susy_train_1K_test_1K/susy_1Kn"
DATA="/Users/gichavez/Documents/mats/SUSY1/susy_train_256_test_256/susy_256n"

# Kernel type, Gaussian=1, Laplacian=2
kernel=1

# Data dimension
dim=8

# Gaussian parameters h and lambda:
hval=1
lval=4

RTOL=1e-1
DD=64

# Remove binary files
EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann"
rm -rf ${EXEC}
EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann_MPI"
rm -rf ${EXEC}

# Build
sh /Users/gichavez/Documents/Github/code_strann/gcbuild.sh

# # Sequential
# EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann"
# time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} natural test \
#       --hss_rel_tol=${RTOL} --hss_leaf_size=64 --hss_max_rank=2000 \
#       --hss_dd=${DD} --hss_verbose

# # Distributed
EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann_MPI"
time mpirun -n 4 ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} natural test \
      --hss_rel_tol=${RTOL} --hss_leaf_size=64 --hss_max_rank=2000 \
      --hss_dd=${DD} --hss_verbose


# Scratch-parallel
# EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/scratch"
# rm -rf ${EXEC}
# sh /Users/gichavez/Documents/Github/code_strann/gcbuild.sh
# mpirun -n 4 ${EXEC}