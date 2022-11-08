#!/bin/bash

T=8

dim=3
cor=.2
tol=1e-2

out=out_cov_in_place_l256
mkdir $out

leaf=256

export OMP_NUM_THREADS=$T
# for comp in stable original; do
for tol in 1e-2 1e-4 1e-6; do
    for comp in stable; do
        # for k in 14 19 24 29; do
        for k in 9 19 29; do
            echo $k
            sampling=sjlt
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
            sampling=gaussian
            ../../build/examples/dense/testCovariance \
                $dim $k $cor --hss_rel_tol $tol \
                --hss_compression_algorithm $comp \
                --hss_compression_sketch $sampling \
                --hss_leaf_size $leaf --help \
                > ${out}/out_dim${dim}_k${k}_cor${cor}_T${T}_tol${tol}_leaf${leaf}_${comp}_${sampling}
        done
    done
done
