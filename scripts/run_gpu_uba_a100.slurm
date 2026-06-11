#!/bin/bash
#SBATCH --job-name=hw4_gpu_a100
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=hw4_gpu_a100_%j.out

set -euo pipefail

source /etc/profile.d/z00_lmod.sh
module purge
module load CUDA/11.1.1-GCC-10.3.0

cd "$SLURM_SUBMIT_DIR"

export ARCH=sm_80
bash scripts/run_benchmarks_cluster.sh
