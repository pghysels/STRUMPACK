#!/bin/bash

comp=stable
tol=1e-4

for nnz in 2 4 8; do
    for k in 14 19 24 29 34; do
        grep "Number of unknowns:" out_cov/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $4}' >> tmp_N_nnz${nnz}
        grep "rank(H)" out_cov/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $4}' >> tmp_rank_nnz${nnz}
        grep "A\*S time\|AT\*S time" out_cov/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$5} END {print sum}' >> tmp_sample0_time_nnz${nnz}
        grep "A\*S appended cols time\|AT\*S appended cols time" out_cov/out_dim3_k${k}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$7} END {print sum}' >> tmp_sample_append_time_nnz${nnz}
        grep "A\*S and AT\*S gemm" out_cov/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$6} END {print sum}' >> tmp_sample_gauss_time_nnz${nnz}
        grep "relative error" out_cov/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $7}' >> tmp_rel_err_nnz${nnz}
    done
    paste tmp_N_nnz${nnz} tmp_rank_nnz${nnz} > tmp_N_rank_nnz${nnz}
    paste tmp_N_nnz${nnz} tmp_sample0_time_nnz${nnz} tmp_sample_append_time_nnz${nnz} tmp_sample_gauss_time_nnz${nnz} > tmp_N_sample_append_nnz${nnz}
    paste tmp_N_nnz${nnz} tmp_rel_err_nnz${nnz} > tmp_N_rel_err_nnz${nnz}
done
for k in 14 19 24 29 34; do
    grep "Number of unknowns:" out_cov/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $4}' >> tmp_N_gaussian
    grep "rank(H)" out_cov/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $4}' >> tmp_rank_gaussian
    grep "relative error" out_cov/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $7}' >> tmp_rel_err_gaussian
done
paste tmp_N_gaussian tmp_rank_gaussian > tmp_N_rank_gaussian
paste tmp_N_gaussian tmp_rel_err_gaussian > tmp_N_rel_err_gaussian
cat tmp_N_rel_err_gaussian

gnuplot plot_cov.gpl

rm tmp*
