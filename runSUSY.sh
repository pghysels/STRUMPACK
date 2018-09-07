#!/usr/bin/env bash

DATA="/Users/gichavez/Documents/Github/mats/SUSY1K_latest/susy_1Kn"
DATA="/Users/gichavez/Documents/Github/mats/SUSY1/susy_train_10K_test_1K/susy_10Kn"

EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann"

# Only valid and test for 10k
# DATA="/Users/gichavez/Documents/Github/mats/SUSY3/susy_N"
# DATA="/Users/gichavez/Documents/Github/mats/SUSY3/susy_N"

kernel=1
dim=8
hval=1
lval=4

RTOL=1e-3
time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
      --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
      --hss_d0=1024 --hss_dd=512 --hss_log_ranks --hss_verbose