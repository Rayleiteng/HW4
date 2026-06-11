#!/usr/bin/env bash
#SBATCH --job-name=hw4_gpu_heat
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

module load cuda || true
bash scripts/run_benchmarks.sh
