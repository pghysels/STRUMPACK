#!/bin/bash

out=out_h2opus
mkdir $out

tol=1e-3

leaf=64

export OMP_NUM_THREADS=4

for k in 25 50 75 100 125 150 200 225 250 275 300 325 350; do
    sep=$((k*k-1))
    echo $k

    comp=H2
    ./build/examples/sparse/testPoisson3d \
           $k \
           --sp_nx $k --sp_ny $k --sp_nz $k \
           --sp_disable_gpu \
           --sp_disable_openmp_tree \
           --sp_compression $comp \
           --hodlr_rel_tol $tol \
           --hodlr_abs_tol $tol \
           --hodlr_leaf_size $leaf \
           --sp_compression_min_sep_size $sep \
           --sp_compression_min_front_size 100000000 \
           --sp_print_compressed_front_stats \
           > ${out}/P3_${k}_${comp}_tol${tol}_leaf${leaf}.log

    comp=BLR
    ./build/examples/sparse/testPoisson3d \
           $k \
           --hodlr_leaf_size $leaf \
           --blr_leaf_size $leaf \
           --sp_disable_gpu \
           --sp_disable_openmp_tree \
           --sp_compression $comp \
           --blr_rel_tol $tol \
           --blr_abs_tol $tol \
           --hodlr_leaf_size $leaf \
           --blr_leaf_size $leaf \
           --sp_compression_min_sep_size $sep \
           --sp_compression_min_front_size 100000000 \
           --sp_print_compressed_front_stats \
           > ${out}/P3_${k}_${comp}_tol${tol}_leaf${leaf}.log

    comp=HODLR
    lvl=0
    mpirun -n 1 ./build/examples/sparse/testPoisson3dMPIDist \
           $k \
           --sp_disable_gpu \
           --sp_disable_openmp_tree \
           --sp_compression $comp \
           --hodlr_rel_tol $tol \
           --hodlr_abs_tol $tol \
           --hodlr_leaf_size $leaf \
           --sp_compression_min_sep_size $sep \
           --sp_compression_min_front_size 100000000 \
           --hodlr_butterfly_levels $lvl \
           --sp_print_compressed_front_stats \
           > ${out}/P3_${k}_${comp}_tol${tol}_leaf${leaf}.log

    compression=HODBF
    comp=HODLR
    lvl=100
    mpirun -n 1 ./build/examples/sparse/testPoisson3dMPIDist \
           $k \
           --sp_disable_gpu \
           --sp_disable_openmp_tree \
           --sp_compression $comp \
           --hodlr_rel_tol $tol \
           --hodlr_abs_tol $tol \
           --hodlr_leaf_size $leaf \
           --sp_compression_min_sep_size $sep \
           --sp_compression_min_front_size 100000000 \
           --hodlr_butterfly_levels $lvl \
           --sp_print_compressed_front_stats \
           > ${out}/P3_${k}_${compression}_tol${tol}_leaf${leaf}.log

done
