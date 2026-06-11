#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results plots

make -j"$(nproc)"

./cuda_device_info | tee results/device_info.txt

: > results/benchmark.log
: > results/validation.log

run_bench() {
    echo "COMMAND $*" | tee -a results/benchmark.log
    "$@" | tee -a results/benchmark.log
}

run_validation() {
    echo "COMMAND $*" | tee -a results/validation.log
    "$@" | tee -a results/validation.log
}

echo "Generating serial snapshots for plots..."
rm -f results/serial_snapshot_step_*.csv results/serial_final.csv
run_bench ./heat_serial --n 256 --steps 100000 --eps 1e-4 \
    --snapshot-prefix results/serial_snapshot \
    --snapshot-steps 0,100,500,2000,20000,100000 \
    --output results/serial_final.csv

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
    run_bench ./heat_cuda --n "$n" --steps 2000 --block-x 8 --block-y 8
    run_bench ./heat_cuda_tile --n "$n" --steps 2000 --tile-x 16 --tile-y 16
done

python3 scripts/plot_results.py results/benchmark.log
