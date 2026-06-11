#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p cluster_results

: "${ARCH:=sm_80}"
export ARCH

make clean
make -j"${SLURM_CPUS_PER_TASK:-$(nproc)}"

./cuda_device_info | tee cluster_results/device_info.txt

: > cluster_results/benchmark.log
: > cluster_results/validation.log

run_bench() {
    echo "COMMAND $*" | tee -a cluster_results/benchmark.log
    "$@" | tee -a cluster_results/benchmark.log
}

run_validation() {
    echo "COMMAND $*" | tee -a cluster_results/validation.log
    "$@" | tee -a cluster_results/validation.log
}

echo "Validating CUDA kernels against the CPU reference..."
run_validation ./heat_cuda --n 256 --steps 500 --block-x 16 --block-y 16 \
    --validate
run_validation ./heat_cuda_tile --n 256 --steps 500 --tile-x 16 --tile-y 16 \
    --validate

echo "Serial baselines for speedup table..."
for n in 512 1024 2048; do
    run_bench ./heat_serial --n "$n" --steps 2000
done

echo "Part A: naive block-size exploration..."
for b in 8 16 32; do
    run_bench ./heat_cuda --n 1024 --steps 2000 --block-x "$b" --block-y "$b"
done

echo "Part B: shared-memory tile-size exploration..."
for t in 8 16 32; do
    run_bench ./heat_cuda_tile --n 1024 --steps 2000 --tile-x "$t" --tile-y "$t"
done

echo "Throughput versus N for both CUDA kernels..."
for n in 256 512 1024 2048 4096; do
    run_bench ./heat_cuda --n "$n" --steps 2000 --block-x 16 --block-y 16
    run_bench ./heat_cuda_tile --n "$n" --steps 2000 --tile-x 16 --tile-y 16
done
