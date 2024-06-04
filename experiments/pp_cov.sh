#!/bin/bash

out=out_cov
comp=stable

dim=3

for k in 9 19 29; do
    for tol in 1e-2 1e-4 1e-6; do
        rm -rf tmp*
        for nnz in 1 2 4 8; do
            grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $2}' > tmp_err_med_nnz${nnz}
            grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $3}' > tmp_err_min_nnz${nnz}
            grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $4}' > tmp_err_max_nnz${nnz}
        done
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $2}' >> tmp_err_med_gaussian
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $3}' >> tmp_err_min_gaussian
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $4}' >> tmp_err_max_gaussian
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $2}' >> tmp_err_med_srht
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $3}' >> tmp_err_min_srht
        grep "errors:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $4}' >> tmp_err_max_srht
        case $tol in
            1e-2)
                color='red'
                mark='';;
            1e-4)
                color='green'
                mark='square';;
            1e-6)
                color='blue'
                mark='triangle';;
        esac
        printf '\t\\addplot[color=%s, mark=%s*, only marks, mark size=2pt, error bars/.cd, y dir=both,y explicit]\n' $color $mark
        printf '\tcoordinates { %% %s, %s\n' $tol $k
        printf '\t\t(G,  %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_gaussian` `cat tmp_err_min_gaussian` `cat tmp_err_max_gaussian`
        printf '\t\t(S1, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_nnz1` `cat tmp_err_min_nnz1` `cat tmp_err_max_nnz1`
        printf '\t\t(S2, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_nnz2` `cat tmp_err_min_nnz2` `cat tmp_err_max_nnz2`
        printf '\t\t(S4, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_nnz4` `cat tmp_err_min_nnz4` `cat tmp_err_max_nnz4`
        printf '\t\t(S8, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_nnz8` `cat tmp_err_min_nnz8` `cat tmp_err_max_nnz8`
        printf '\t\t(H,  %s)  -= (0, %s) += (0, %s)\n' `cat tmp_err_med_srht` `cat tmp_err_min_srht` `cat tmp_err_max_srht`
        printf '\t};\n'
        echo ""
    done
done


for k in 9 19 29; do
    for tol in 1e-2 1e-4 1e-6; do
        rm -rf tmp*
        for nnz in 1 2 4 8; do
            grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $2}' > tmp_ranks_med_nnz${nnz}
            grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $3}' > tmp_ranks_min_nnz${nnz}
            grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $4}' > tmp_ranks_max_nnz${nnz}
        done
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $2}' >> tmp_ranks_med_gaussian
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $3}' >> tmp_ranks_min_gaussian
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $4}' >> tmp_ranks_max_gaussian
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $2}' >> tmp_ranks_med_srht
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $3}' >> tmp_ranks_min_srht
        grep "ranks:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $4}' >> tmp_ranks_max_srht
        case $tol in
            1e-2)
                color='red'
                mark='';;
            1e-4)
                color='green'
                mark='square';;
            1e-6)
                color='blue'
                mark='triangle';;
        esac
        # case $k in
        #     10000) mark='';;
        #     20000) mark='square';;
        #     40000) mark='triangle';;
        # esac
        # case $tol in
        #     1e-2) color='red';;
        #     1e-4) color='green';;
        #     1e-6) color='blue';;
        # esac
        printf '\t\\addplot[color=%s, mark=%s*, only marks, mark size=2pt, error bars/.cd, y dir=both,y explicit]\n' $color $mark
        printf '\tcoordinates { %% %s, %s\n' $tol $k
        printf '\t\t(G,  %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_gaussian` `cat tmp_ranks_min_gaussian` `cat tmp_ranks_max_gaussian`
        printf '\t\t(S1, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_nnz1` `cat tmp_ranks_min_nnz1` `cat tmp_ranks_max_nnz1`
        printf '\t\t(S2, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_nnz2` `cat tmp_ranks_min_nnz2` `cat tmp_ranks_max_nnz2`
        printf '\t\t(S4, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_nnz4` `cat tmp_ranks_min_nnz4` `cat tmp_ranks_max_nnz4`
        printf '\t\t(S8, %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_nnz8` `cat tmp_ranks_min_nnz8` `cat tmp_ranks_max_nnz8`
        printf '\t\t(H,  %s)  -= (0, %s) += (0, %s)\n' `cat tmp_ranks_med_srht` `cat tmp_ranks_min_srht` `cat tmp_ranks_max_srht`
        printf '\t};\n'
        echo ""

    done
done

for tol in 1e-2 1e-4 1e-6; do
    echo $tol
    for k in 9 19 29; do
        rm -rf tmp*
        for nnz in 1 2 4 8; do
            grep "A\*S time\|AT\*S time" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{sum+=$5} END {print sum/3.0/1000.0}' > tmp_sample_nnz${nnz}
            grep "times:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SJLT_nnz${nnz} | awk '{print $2/1000.0}' > tmp_constr_nnz${nnz}
        done

        grep "A\*S time\|AT\*S time" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{sum+=$5} END {print sum/3.0/1000.0}' >> tmp_sample_gaussian
        grep "times:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $2/1000.0}' >> tmp_constr_gaussian

        grep "A\*S time\|AT\*S time" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{sum+=$5} END {print sum/3.0/1000.0}' >> tmp_sample_srht
        grep "times:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_SRHT | awk '{print $2/1000.0}' >> tmp_constr_srht

        grep "Number of unknowns:" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $4/1000.0}' >> tmp_N_gaussian
        grep "\% of dense" ${out}/out_dim${dim}_k${k}*tol${tol}*${comp}_Gaussian | awk '{print $6}' | tail -n 1 | sed 's/%//'>> tmp_compr
#        printf '& & %s & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.0f & %5.1f \\\\ \n' $k `cat tmp_sample_gaussian` `cat tmp_sample_nnz1` `cat tmp_sample_nnz2` `cat tmp_sample_nnz4` `cat tmp_sample_nnz8` `cat tmp_sample_srht` `cat tmp_constr_gaussian` `cat tmp_constr_nnz1` `cat tmp_constr_nnz2` `cat tmp_constr_nnz4` `cat tmp_constr_nnz8` `cat tmp_constr_srht` `cat tmp_compr`

        # printf '& & %s & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.1f \\\\ \n' $k `cat tmp_sample_gaussian` `cat tmp_sample_nnz1` `cat tmp_sample_nnz2` `cat tmp_sample_nnz4` `cat tmp_sample_nnz8` `cat tmp_sample_srht` `cat tmp_constr_gaussian` `cat tmp_constr_nnz1` `cat tmp_constr_nnz2` `cat tmp_constr_nnz4` `cat tmp_constr_nnz8` `cat tmp_constr_srht` `cat tmp_compr`
        printf '& & %s & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %3.3g & %5.1f \\\\ \n' $k `cat tmp_sample_gaussian` `cat tmp_sample_nnz1` `cat tmp_sample_nnz2` `cat tmp_sample_nnz4` `cat tmp_sample_nnz8` `cat tmp_sample_srht` `cat tmp_constr_gaussian` `cat tmp_constr_nnz1` `cat tmp_constr_nnz2` `cat tmp_constr_nnz4` `cat tmp_constr_nnz8` `cat tmp_constr_srht` `cat tmp_compr`
    done
    echo ""
done

rm tmp*
