#!/bin/bash

# out=out_cov_old
out=out_cov_in_place #_l256
comp=stable
for tol in 1e-2 1e-4 1e-6; do
    rm tmp*
    echo "${tol}"
    for nnz in 1 2 4 8; do
        for k in 9 19 29; do
            grep "Number of unknowns:" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $4}' >> tmp_N_nnz${nnz}
            grep "rank(H)" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $4}' >> tmp_rank_nnz${nnz}
            grep "A\*S time\|AT\*S time" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$5} END {print sum}' >> tmp_sample0_time_nnz${nnz}
            grep "A\*S appended cols time\|AT\*S appended cols time" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$7} END {print sum}' >> tmp_sample_append_time_nnz${nnz}
            grep "A\*S and AT\*S gemm" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{sum+=$6} END {print sum}' >> tmp_sample_gauss_time_nnz${nnz}
            grep "relative error" ${out}/out_dim3_k${k}*tol${tol}*${comp}_sjlt_nnz${nnz} | awk '{print $7}' >> tmp_rel_err_nnz${nnz}
        done
        paste tmp_N_nnz${nnz} tmp_rank_nnz${nnz} > tmp_N_rank_nnz${nnz}
        paste tmp_N_nnz${nnz} tmp_sample0_time_nnz${nnz} tmp_sample_append_time_nnz${nnz} tmp_sample_gauss_time_nnz${nnz} > tmp_N_sample_append_nnz${nnz}
        cat tmp_N_sample_append_nnz${nnz} | awk '{print $2+$3}' > tmp_constr_nnz${nnz}
        paste tmp_N_nnz${nnz} tmp_rel_err_nnz${nnz} > tmp_N_rel_err_nnz${nnz}
    done
    for k in 9 19 29; do
        grep "Number of unknowns:" ${out}/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $4}' >> tmp_N_gaussian
        grep "rank(H)" ${out}/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $4}' >> tmp_rank_gaussian
        grep "relative error" ${out}/out_dim3_k${k}*tol${tol}*${comp}_gaussian | awk '{print $7}' >> tmp_rel_err_gaussian
    done
    paste tmp_N_gaussian tmp_rank_gaussian > tmp_N_rank_gaussian
    paste tmp_N_gaussian tmp_rel_err_gaussian > tmp_N_rel_err_gaussian

    paste tmp_N_gaussian tmp_sample_gauss_time_nnz2 tmp_constr_nnz1 tmp_constr_nnz2 tmp_constr_nnz4 tmp_constr_nnz8 tmp_rank_gaussian tmp_rank_nnz1 tmp_rank_nnz2 tmp_rank_nnz4 tmp_rank_nnz8 tmp_rel_err_gaussian tmp_rel_err_nnz1 tmp_rel_err_nnz2 tmp_rel_err_nnz4 tmp_rel_err_nnz8 > latex_table
    cat latex_table
done

gnuplot plot_cov.gpl

rm tmp*
