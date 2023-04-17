#!/bin/bash
#SBATCH --partition=romeo 
#SBATCH --mem=16000M 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --error=e.out
#SBATCH --output=output.out



omp_threads=64
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job
module load OpenMPI/4.0.5-GCC-10.2.0

export OMP_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=true
export OMP_PLACES="{0}:128:2"

cd /scratch/ws/0/seya960b-paper/romeo/paper_europar-1/benchmarks/teams_openmp
./a.out

