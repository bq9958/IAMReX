#!/bin/bash

#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -J monoCPU
#SBATCH -o monoCPU.out
#SBATCH -A m3766

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
srun -n 64 -c 4 --cpu-bind=cores,verbose ../../amr3d.gnu.x86-milan.MPI.ex inputs.random_particle
