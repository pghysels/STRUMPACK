#!/bin/bash

T=8

dim=3
cor=.2

out=out_cov
mkdir $out

leaf=256
comp=stable

export OMP_NUM_THREADS=$T
for tol in 1e-2 1e-4 1e-6; do
    echo $tol
    for k in 9 19 29; do
        echo $k
        sampling=SJLT
        for nnz in 1 2 4 8; do
            echo $nnz
            ../../build/examples/dense/testCovariance \
                $dim $k $cor --hss_rel_tol $tol \
                --hss_compression_algorithm $comp \
                --hss_nnz0 $nnz --hss_nnz $nnz \
                --hss_compression_sketch $sampling \
                --hss_leaf_size $leaf --help \
                > ${out}/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}_nnz${nnz}
        done
        sampling=Gaussian
        ../../build/examples/dense/testCovariance \
            $dim $k $cor --hss_rel_tol $tol \
            --hss_compression_algorithm $comp \
            --hss_compression_sketch $sampling \
            --hss_leaf_size $leaf --help \
            > ${out}/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}
        sampling=SRHT
        ../../build/examples/dense/testCovariance \
            $dim $k $cor --hss_rel_tol $tol \
            --hss_compression_algorithm $comp \
            --hss_compression_sketch $sampling \
            --hss_leaf_size $leaf --help \
            > ${out}/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}
    done
done