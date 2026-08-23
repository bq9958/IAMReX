#!/bin/bash

#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -J fps_Re200
#SBATCH -o %x_%j.out
##SBATCH -A m3766

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
srun -n 4 ../../../amr3d.gnu.x86-milan.MPI.ex inputs.3d.flow_past_sphere
