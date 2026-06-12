#include "cuda_common.cuh"
#include "heat_common.hpp"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

__global__ void heat_naive_update(float* __restrict__ t_new,
                                  const float* __restrict__ t_old,
                                  int n,
                                  float r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        const int k = i * n + j;
        t_new[k] = t_old[k] +
                   r * (t_old[k - n] + t_old[k + n] + t_old[k - 1] +
                        t_old[k + 1] - 4.0f * t_old[k]);
    }
}

__global__ void heat_naive_update_diff(float* __restrict__ t_new,
                                       const float* __restrict__ t_old,
                                       float* __restrict__ diff_grid,
                                       int n,
                                       float r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        const int k = i * n + j;
        float local_diff = 0.0f;
        if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
            const float updated =
                t_old[k] +
                r * (t_old[k - n] + t_old[k + n] + t_old[k - 1] +
                     t_old[k + 1] - 4.0f * t_old[k]);
            t_new[k] = updated;
            local_diff = fabsf(updated - t_old[k]);
        }
        diff_grid[k] = local_diff;
    }
}

struct Options {
    int n = 1024;
    int steps = 2000;
    int block_x = 16;
    int block_y = 16;
    int check_interval = 50;
    float r = 0.24f;
    float eps = 0.0f;
    bool validate = false;
    std::string output;
};

Options parse_options(int argc, char** argv) {
    Options opt;
    opt.n = heat::get_int_arg(argc, argv, "--n", opt.n);
    opt.steps = heat::get_int_arg(argc, argv, "--steps", opt.steps);
    opt.block_x = heat::get_int_arg(argc, argv, "--block-x", opt.block_x);
    opt.block_y = heat::get_int_arg(argc, argv, "--block-y", opt.block_y);
    opt.check_interval =
        heat::get_int_arg(argc, argv, "--check-interval", opt.check_interval);
    opt.r = heat::get_float_arg(argc, argv, "--r", opt.r);
    opt.eps = heat::get_float_arg(argc, argv, "--eps", opt.eps);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--validate") {
            opt.validate = true;
        }
    }
    opt.output = heat::get_arg(argc, argv, "--output", std::string());

    if (opt.n < 3) {
        throw std::invalid_argument("N must be at least 3");
    }
    if (opt.steps < 1) {
        throw std::invalid_argument("steps must be positive");
    }
    if (opt.block_x < 1 || opt.block_y < 1) {
        throw std::invalid_argument("block dimensions must be positive");
    }
    if (opt.r <= 0.0f || opt.r > 0.25f) {
        throw std::invalid_argument("r must be in (0, 0.25] for stability");
    }
    if (opt.check_interval < 1) {
        throw std::invalid_argument("check interval must be positive");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
        const cudaDeviceProp prop = active_device_properties();
        const int threads_per_block = opt.block_x * opt.block_y;
        if (threads_per_block > prop.maxThreadsPerBlock) {
            throw std::invalid_argument("block has more threads than the device allows");
        }

        std::vector<float> host_grid;
        heat::initialize_grid(host_grid, opt.n);

        float* d_old = nullptr;
        float* d_new = nullptr;
        float* d_diff = nullptr;
        const std::size_t bytes = host_grid.size() * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_old, bytes));
        CHECK_CUDA(cudaMalloc(&d_new, bytes));
        CHECK_CUDA(cudaMemcpy(d_old, host_grid.data(), bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_new, host_grid.data(), bytes,
                              cudaMemcpyHostToDevice));

        const dim3 block(opt.block_x, opt.block_y);
        const dim3 grid((opt.n + block.x - 1) / block.x,
                        (opt.n + block.y - 1) / block.y);
        CHECK_CUDA(cudaMalloc(&d_diff, bytes));

        int steps_run = 0;
        double final_delta = 0.0;
        bool converged = false;

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < opt.steps; ++step) {
            const bool should_check =
                (step + 1 == opt.steps) ||
                (opt.eps > 0.0f && ((step + 1) % opt.check_interval == 0));

            if (should_check) {
                heat_naive_update_diff<<<grid, block>>>(
                    d_new, d_old, d_diff, opt.n, opt.r);
            } else {
                heat_naive_update<<<grid, block>>>(d_new, d_old, opt.n, opt.r);
            }
            CHECK_CUDA(cudaGetLastError());

            if (should_check) {
                thrust::device_ptr<float> diffs(d_diff);
                final_delta = static_cast<double>(thrust::reduce(
                    diffs, diffs + host_grid.size(), 0.0f, thrust::maximum<float>()));
            }

            std::swap(d_old, d_new);
            steps_run = step + 1;

            if (should_check && opt.eps > 0.0f &&
                final_delta < static_cast<double>(opt.eps)) {
                converged = true;
                break;
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        CHECK_CUDA(cudaMemcpy(host_grid.data(), d_old, bytes,
                              cudaMemcpyDeviceToHost));

        double validation_error = std::numeric_limits<double>::quiet_NaN();
        if (opt.validate) {
            std::vector<float> reference;
            heat::initialize_grid(reference, opt.n);
            heat::solve_cpu(reference, opt.n, steps_run, opt.r, 0.0f);
            validation_error = heat::max_abs_difference(host_grid, reference);
        }

        if (!opt.output.empty()) {
            heat::write_csv_grid(opt.output, host_grid, opt.n);
        }

        std::ostringstream extra;
        extra << "block=" << opt.block_x << 'x' << opt.block_y;
        if (opt.validate) {
            extra << ",validation_max_abs_error=" << std::scientific
                  << std::setprecision(6) << validation_error;
        }

        heat::print_result_line("cuda_naive",
                                opt.n,
                                steps_run,
                                elapsed_ms,
                                final_delta,
                                converged,
                                extra.str());

        CHECK_CUDA(cudaFree(d_diff));
        CHECK_CUDA(cudaFree(d_new));
        CHECK_CUDA(cudaFree(d_old));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}





