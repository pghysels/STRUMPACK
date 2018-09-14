#!/usr/bin/env bash

DATA="/Users/gichavez/Documents/mats/SUSY1/susy_train_10K_test_1K/susy_10Kn"
EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann"

# Kernel type, Gaussian=1, Laplacian=2
kernel=1

# Data dimension
dim=8

# Gaussian parameters h and lambda:
hval=1
lval=4


RTOL=1e-1
DD=128
time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
      --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
      --hss_dd=${DD} --hss_verbose

RTOL=1e-2
DD=128
time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
      --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
      --hss_dd=${DD} --hss_verbose

RTOL=1e-3
DD=512
time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
      --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
      --hss_dd=${DD} --hss_verbose
