#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_result_line(line):
    line = line.strip()
    if not line.startswith("RESULT,"):
        return None
    record = {}
    for field in line.split(",")[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            record[key] = value
    for key in ("N", "steps", "converged"):
        if key in record:
            record[key] = int(record[key])
    for key in ("elapsed_ms", "throughput_Gpt_s", "max_delta"):
        if key in record:
            record[key] = float(record[key])
    if "validation_max_abs_error" in record:
        record["validation_max_abs_error"] = float(
            record["validation_max_abs_error"]
        )
    if "stencil_shared_kb" in record:
        record["stencil_shared_kb"] = float(record["stencil_shared_kb"])
    return record


def load_records(log_path):
    records = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_result_line(line)
            if parsed:
                records.append(parsed)
    return records


def write_parsed_csv(records, path):
    keys = [
        "mode",
        "N",
        "steps",
        "elapsed_ms",
        "throughput_Gpt_s",
        "max_delta",
        "converged",
        "block",
        "tile",
        "stencil_shared_kb",
        "validation_max_abs_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def plot_snapshots():
    files = sorted(
        glob.glob("results/serial_snapshot_step_*.csv"),
        key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]),
    )
    if not files:
        return

    cols = 3
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.4 * cols, 3.8 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    vmax = 100.0
    for ax, path in zip(axes, files):
        grid = np.loadtxt(path, delimiter=",")
        step = int(os.path.splitext(path)[0].split("_")[-1])
        im = ax.imshow(grid, origin="upper", cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(f"step {step}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    for ax in axes[len(files) :]:
        ax.axis("off")
    fig.colorbar(
        im,
        ax=axes[: len(files)].tolist(),
        shrink=0.82,
        pad=0.02,
        label="temperature",
    )
    fig.suptitle("2D heat equation snapshots")
    fig.savefig("plots/serial_snapshots.png", dpi=180)
    plt.close(fig)

    final_path = "results/serial_final.csv"
    if os.path.exists(final_path):
        grid = np.loadtxt(final_path, delimiter=",")
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(grid, origin="upper", cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title("stationary solution approximation")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, label="temperature")
        fig.tight_layout()
        fig.savefig("plots/stationary_solution.png", dpi=180)
        plt.close(fig)


def best_last_by_key(records, mode, config_key, config_value):
    selected = {}
    for record in records:
        if record.get("mode") != mode:
            continue
        if record.get(config_key) != config_value:
            continue
        selected[record["N"]] = record
    return [selected[n] for n in sorted(selected)]


def plot_throughput(
    records,
    output_path="plots/throughput_vs_n.png",
    naive_block="8x8",
    shared_tile="16x16",
    title="CUDA throughput versus grid size",
):
    naive = best_last_by_key(records, "cuda_naive", "block", naive_block)
    shared = best_last_by_key(records, "cuda_shared", "tile", shared_tile)
    if not naive or not shared:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(
        [r["N"] for r in naive],
        [r["throughput_Gpt_s"] for r in naive],
        marker="o",
        label="naive global memory",
    )
    ax.plot(
        [r["N"] for r in shared],
        [r["throughput_Gpt_s"] for r in shared],
        marker="s",
        label="shared-memory tiled",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("throughput (Gpoints/s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", nargs="?", default="results/benchmark.log")
    parser.add_argument("--parsed-csv", default="results/parsed_results.csv")
    parser.add_argument("--throughput-plot", default="plots/throughput_vs_n.png")
    parser.add_argument("--naive-block", default="8x8")
    parser.add_argument("--shared-tile", default="16x16")
    parser.add_argument("--title", default="CUDA throughput versus grid size")
    parser.add_argument("--skip-snapshots", action="store_true")
    args = parser.parse_args()

    records = load_records(args.log_path)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.dirname(args.parsed_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.throughput_plot), exist_ok=True)
    write_parsed_csv(records, args.parsed_csv)
    if not args.skip_snapshots:
        plot_snapshots()
    plot_throughput(
        records,
        output_path=args.throughput_plot,
        naive_block=args.naive_block,
        shared_tile=args.shared_tile,
        title=args.title,
    )


if __name__ == "__main__":
    main()
