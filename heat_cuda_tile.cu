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
#include <sstream>
#include <utility>
#include <vector>

namespace {

__device__ __forceinline__ int clamp_index(int x, int lo, int hi) {
    return max(lo, min(x, hi));
}

__device__ __forceinline__ float read_grid(const float* grid,
                                           int n,
                                           int row,
                                           int col) {
    row = clamp_index(row, 0, n - 1);
    col = clamp_index(col, 0, n - 1);
    return grid[row * n + col];
}

__device__ void load_tile_with_halo(float* tile,
                                    const float* old_grid,
                                    int n) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * blockDim.x + tx;
    const int row = blockIdx.y * blockDim.y + ty;

    const int sw = blockDim.x + 2;
    const int s_col = tx + 1;
    const int s_row = ty + 1;

    tile[s_row * sw + s_col] =
        (row < n && col < n) ? old_grid[row * n + col] : 0.0f;

    if (tx == 0) {
        tile[s_row * sw] =
            (row < n) ? read_grid(old_grid, n, row, col - 1) : 0.0f;
    }
    if (tx == blockDim.x - 1) {
        tile[s_row * sw + blockDim.x + 1] =
            (row < n) ? read_grid(old_grid, n, row, col + 1) : 0.0f;
    }
    if (ty == 0) {
        tile[s_col] =
            (col < n) ? read_grid(old_grid, n, row - 1, col) : 0.0f;
    }
    if (ty == blockDim.y - 1) {
        tile[(blockDim.y + 1) * sw + s_col] =
            (col < n) ? read_grid(old_grid, n, row + 1, col) : 0.0f;
    }

    if (tx == 0 && ty == 0) {
        tile[0] = (row < n && col < n)
                      ? read_grid(old_grid, n, row - 1, col - 1)
                      : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        tile[blockDim.x + 1] =
            (row < n && col < n)
                ? read_grid(old_grid, n, row - 1, col + 1)
                : 0.0f;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        tile[(blockDim.y + 1) * sw] =
            (row < n && col < n)
                ? read_grid(old_grid, n, row + 1, col - 1)
                : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        tile[(blockDim.y + 1) * sw + blockDim.x + 1] =
            (row < n && col < n)
                ? read_grid(old_grid, n, row + 1, col + 1)
                : 0.0f;
    }

    __syncthreads();
}

__device__ __forceinline__ float stencil_from_tile(const float* tile,
                                                   int sw,
                                                   int s_row,
                                                   int s_col,
                                                   float r) {
    const float center = tile[s_row * sw + s_col];
    return center +
           r * (tile[(s_row - 1) * sw + s_col] +
                tile[(s_row + 1) * sw + s_col] +
                tile[s_row * sw + s_col - 1] +
                tile[s_row * sw + s_col + 1] - 4.0f * center);
}

__global__ void heat_shared_step(float* __restrict__ new_grid,
                                 const float* __restrict__ old_grid,
                                 int n,
                                 float r) {
    extern __shared__ float tile[];
    load_tile_with_halo(tile, old_grid, n);

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
        const int sw = blockDim.x + 2;
        const int s_row = threadIdx.y + 1;
        const int s_col = threadIdx.x + 1;
        new_grid[row * n + col] =
            stencil_from_tile(tile, sw, s_row, s_col, r);
    }
}

__global__ void heat_shared_step_with_diff(float* __restrict__ new_grid,
                                           const float* __restrict__ old_grid,
                                           float* __restrict__ diff_grid,
                                           int n,
                                           float r) {
    extern __shared__ float tile[];
    load_tile_with_halo(tile, old_grid, n);

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n) {
        return;
    }

    const int k = row * n + col;
    float diff = 0.0f;

    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
        const int sw = blockDim.x + 2;
        const int s_row = threadIdx.y + 1;
        const int s_col = threadIdx.x + 1;
        const float old_value = tile[s_row * sw + s_col];
        const float updated = stencil_from_tile(tile, sw, s_row, s_col, r);

        new_grid[k] = updated;
        diff = fabsf(updated - old_value);
    }

    diff_grid[k] = diff;
}

struct Options {
    int n = 1024;
    int steps = 2000;
    int tile_x = 16;
    int tile_y = 16;
    int check_interval = 50;
    float r = 0.24f;
    float eps = 0.0f;
    bool validate = false;
};

Options read_options(int argc, char** argv) {
    Options opt;
    opt.n = heat::get_int_arg(argc, argv, "--n", opt.n);
    opt.steps = heat::get_int_arg(argc, argv, "--steps", opt.steps);
    opt.tile_x = heat::get_int_arg(argc, argv, "--tile-x", opt.tile_x);
    opt.tile_y = heat::get_int_arg(argc, argv, "--tile-y", opt.tile_y);
    opt.check_interval =
        heat::get_int_arg(argc, argv, "--check-interval", opt.check_interval);
    opt.r = heat::get_float_arg(argc, argv, "--r", opt.r);
    opt.eps = heat::get_float_arg(argc, argv, "--eps", opt.eps);
    opt.validate = heat::has_arg(argc, argv, "--validate");

    if (opt.n < 3) {
        throw std::invalid_argument("N must be at least 3");
    }
    if (opt.steps < 1) {
        throw std::invalid_argument("steps must be positive");
    }
    if (opt.tile_x < 1 || opt.tile_y < 1) {
        throw std::invalid_argument("tile dimensions must be positive");
    }
    if (opt.r <= 0.0f || opt.r > 0.25f) {
        throw std::invalid_argument("r must be in (0, 0.25] for stability");
    }
    if (opt.check_interval < 1) {
        throw std::invalid_argument("check interval must be positive");
    }
    return opt;
}

double reduce_max_on_gpu(float* device_values, std::size_t count) {
    thrust::device_ptr<float> values(device_values);
    return static_cast<double>(
        thrust::reduce(values, values + count, 0.0f, thrust::maximum<float>()));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = read_options(argc, argv);
        const cudaDeviceProp prop = active_device_properties();
        const int threads_per_block = opt.tile_x * opt.tile_y;

        if (threads_per_block > prop.maxThreadsPerBlock) {
            throw std::invalid_argument(
                "tile has more threads than the device allows");
        }

        const std::size_t shared_bytes =
            static_cast<std::size_t>(opt.tile_y + 2) *
            static_cast<std::size_t>(opt.tile_x + 2) * sizeof(float);
        if (shared_bytes > static_cast<std::size_t>(prop.sharedMemPerBlock)) {
            throw std::invalid_argument("requested tile uses too much shared memory");
        }

        std::vector<float> host_grid;
        heat::initialize_grid(host_grid, opt.n);

        const std::size_t value_count = host_grid.size();
        const std::size_t bytes = value_count * sizeof(float);

        float* old_grid = nullptr;
        float* new_grid = nullptr;
        float* diff_grid = nullptr;

        CHECK_CUDA(cudaMalloc(&old_grid, bytes));
        CHECK_CUDA(cudaMalloc(&new_grid, bytes));
        CHECK_CUDA(cudaMalloc(&diff_grid, bytes));
        CHECK_CUDA(
            cudaMemcpy(old_grid, host_grid.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(
            cudaMemcpy(new_grid, host_grid.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 block(opt.tile_x, opt.tile_y);
        const dim3 grid((opt.n + block.x - 1) / block.x,
                        (opt.n + block.y - 1) / block.y);

        int steps_run = 0;
        double final_delta = 0.0;
        bool converged = false;

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();

        for (int step = 0; step < opt.steps; ++step) {
            const bool check_now =
                (step + 1 == opt.steps) ||
                (opt.eps > 0.0f && (step + 1) % opt.check_interval == 0);

            if (check_now) {
                heat_shared_step_with_diff<<<grid, block, shared_bytes>>>(
                    new_grid, old_grid, diff_grid, opt.n, opt.r);
            } else {
                heat_shared_step<<<grid, block, shared_bytes>>>(
                    new_grid, old_grid, opt.n, opt.r);
            }
            CHECK_CUDA(cudaGetLastError());

            if (check_now) {
                final_delta = reduce_max_on_gpu(diff_grid, value_count);
            }

            std::swap(old_grid, new_grid);
            steps_run = step + 1;

            if (check_now && opt.eps > 0.0f && final_delta < opt.eps) {
                converged = true;
                break;
            }
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();

        CHECK_CUDA(
            cudaMemcpy(host_grid.data(), old_grid, bytes, cudaMemcpyDeviceToHost));

        double validation_error = std::numeric_limits<double>::quiet_NaN();
        if (opt.validate) {
            std::vector<float> reference;
            heat::initialize_grid(reference, opt.n);
            heat::solve_cpu(reference, opt.n, steps_run, opt.r, 0.0f);
            validation_error = heat::max_abs_difference(host_grid, reference);
        }

        std::ostringstream extra;
        extra << "tile=" << opt.tile_x << 'x' << opt.tile_y
              << ",stencil_shared_kb=" << std::fixed << std::setprecision(3)
              << static_cast<double>(shared_bytes) / 1024.0;
        if (opt.validate) {
            extra << ",validation_max_abs_error=" << std::scientific
                  << std::setprecision(6) << validation_error;
        }

        heat::print_result_line("cuda_shared", opt.n, steps_run, elapsed_ms,
                                final_delta, converged, extra.str());

        CHECK_CUDA(cudaFree(diff_grid));
        CHECK_CUDA(cudaFree(new_grid));
        CHECK_CUDA(cudaFree(old_grid));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
