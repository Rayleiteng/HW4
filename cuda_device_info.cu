#include "cuda_common.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

int main() {
    try {
        int count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&count));
        std::cout << "CUDA device count: " << count << '\n';
        for (int dev = 0; dev < count; ++dev) {
            cudaDeviceProp prop{};
            CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
            std::cout << "Device " << dev << ": " << prop.name << '\n'
                      << "  Compute capability: " << prop.major << '.'
                      << prop.minor << '\n'
                      << "  SM count: " << prop.multiProcessorCount << '\n'
                      << "  Max threads per block: "
                      << prop.maxThreadsPerBlock << '\n'
                      << "  Shared memory per block: "
                      << (prop.sharedMemPerBlock / 1024.0) << " KB\n"
                      << "  Shared memory per SM: "
                      << (prop.sharedMemPerMultiprocessor / 1024.0)
                      << " KB\n"
                      << "  Memory clock: " << prop.memoryClockRate
                      << " kHz\n"
                      << "  Memory bus width: " << prop.memoryBusWidth
                      << " bits\n"
                      << "  Core clock: " << prop.clockRate << " kHz\n"
                      << std::fixed << std::setprecision(2)
                      << "  Approx peak FP32: "
                      << theoretical_fp32_gflops(prop) << " GFLOP/s\n"
                      << "  Approx peak bandwidth: "
                      << theoretical_memory_bandwidth_gbs(prop) << " GB/s\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
