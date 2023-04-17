#!/bin/bash
#SBATCH --partition=ml-all 
#SBATCH --mem=16000M 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --sockets-per-node=1 
#SBATCH --cores-per-socket=22 
#SBATCH --threads-per-core=1
#SBATCH --time=00:30:00
#SBATCH --error=power.err
#SBATCH --output=power.out


omp_threads=22
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job
module load GCCcore/10.2.0

export OMP_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=true
export OMP_PLACES="{0}:22:1"

cd /scratch/ws/0/seya960b-paper/power/paper_europar-1/benchmarks/teams_openmp
./a.out

