#!/bin/bash

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export EXEC=/global/cscratch1/sd/gichavez/intel17/STR_DenseExamples/build/examples/DenseCovariance

echo srun -N 1 -n 32 -c 2 --cpu_bind=threads ${EXEC} 3 29 0.2 --hss_d0 1500 --hss_dd 128  2>&1 | tee -a log_runCov.txt
time srun -N 1 -n 32 -c 2 --cpu_bind=threads ${EXEC} 3 29 0.2 --hss_d0 1500 --hss_dd 128  2>&1 | tee -a log_runCov.txt

