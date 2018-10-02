#!/usr/bin/env bash

EXEC="/Users/gichavez/Documents/Github/code_strann/build/examples/KernelRegression_ann"
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
time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
      --hss_rel_tol=${RTOL} --hss_leaf_size=64 --hss_max_rank=2000 \
      --hss_dd=${DD} --hss_verbose

# RTOL=1e-2
# DD=128
# time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
#       --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
#       --hss_dd=${DD} --hss_verbose

# RTOL=1e-3
# DD=512
# time ${EXEC} ${DATA} ${dim} ${hval} ${lval} ${kernel} 2means test \
#       --hss_rel_tol=${RTOL} -hss_leaf_size=128 --hss_max_rank=2000 \
#       --hss_dd=${DD} --hss_verbose



# 

# # usage: ./KernelRegression_mf file d h lambda kern(1=Gau,2=Lapl) reorder(nat, 2means, kd, pca) mode(valid, test)

# # data dimension    = 8
# # kernel h          = 1
# # lambda            = 4
# # kernel type       = Gauss
# # reordering/clust  = 2means
# # validation/test   = test

# # hss_opts.d0       = 128
# # hss_opts.dd       = 64
# # hss_opts.rel_t    = 0.1
# # hss_opts.abs_t    = 1e-08
# # hss_opts.leaf     = 64

# # Reading data...
# # Reading took 0.002141
# # matrix size = 256 x 8

# # Starting HSS compression...
# # ANN time = 0.125052 sec
# ---> USING COMPRESS_ANN <---
# NLEF(256,256)
# NLEF(120,120)
# NLEF(103,103)
# LEAF(59,59)
# LEAF(44,44)
# LEAF(17,17)
# NLEF(136,136)
# LEAF(62,62)
# NLEF(74,74)
# LEAF(40,40)
# LEAF(34,34)
# ### compression time = 0.043532 ###
# # created K matrix of dimension 256 x 256 with 4 levels
# # compression succeeded!
# # rank(K) = 47
# # HSS memory(K) = 0.312528 MB 
# # HSS matrix is 59.61% of dense
# # compression error = ||Kdense-K*I||_F/||Kdense||_F = 0.00832226

# # Factorization start
# # factorization time = 0.020699

# # Solution start...
# # solve time = 0.000315
# # solution error = 0.00834432

# # Prediction start...
# # Prediction took 0.015063
# # prediction score: 75%


# real  0m0.265s
# user  0m0.257s
# sys 0m0.006s