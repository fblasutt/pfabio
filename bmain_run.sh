#!/bin/bash

#SBATCH -N 1                 ## number of nodes
#SBATCH --cpus-per-task=1                 ## number of nodes
#SBATCH --mem-per-cpu=1500MB            ## number of nodes
#SBATCH -t 00:00:10          ## walltime
#SBATCH	--job-name="py_ivf"    ## name of job




source ~/pension_E/bin/activate
python -u estimation.py
deactivate
exit