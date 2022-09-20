#!/bin/bash

T=8

dim=3
cor=.2
tol=1e-4

mkdir out_cov

leaf=64

export OMP_NUM_THREADS=$T
for comp in stable original; do
    for k in 14 19 24 29; do
        echo $k
        sampling=sjlt
        for nnz in 2 4 8; do
            echo $nnz
            ../build/examples/dense/testCovariance \
                $dim $k $cor --hss_rel_tol $tol \
                --hss_compression_algorithm $comp \
                --hss_nnz0 $nnz --hss_nnz $nnz \
                --hss_compression_sketch $sampling \
                --hss_leaf_size $leaf --help \
                > out_cov/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}_nnz${nnz}
        done
        sampling=gaussian
        ../build/examples/dense/testCovariance \
            $dim $k $cor --hss_rel_tol $tol \
            --hss_compression_algorithm $comp \
            --hss_compression_sketch $sampling \
            --hss_leaf_size $leaf --help \
            > out_cov/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}
    done
done
