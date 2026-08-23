#!/bin/bash

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J Re400
#SBATCH -o Re400.out
#SBATCH -t 02:00:00
#SBATCH -A m3766

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
srun -n 4 --gpu-bind=none -G 4 ../../../amr3d.gnu.MPI.CUDA.ex inputs.3d.flow_past_sphere
