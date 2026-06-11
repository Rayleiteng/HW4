#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t status__ = (call);                                        \
        if (status__ != cudaSuccess) {                                        \
            std::ostringstream oss__;                                        \
            oss__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " \
                  << cudaGetErrorString(status__);                           \
            throw std::runtime_error(oss__.str());                           \
        }                                                                    \
    } while (0)

inline int cuda_core_count_per_sm(int major, int minor) {
    if (major == 8 && minor == 9) {
        return 128;
    }
    if (major == 8 && minor == 6) {
        return 128;
    }
    if (major == 8 && minor == 0) {
        return 64;
    }
    if (major == 9) {
        return 128;
    }
    if (major == 7) {
        return 64;
    }
    if (major == 6 && minor == 1) {
        return 128;
    }
    if (major == 6) {
        return 64;
    }
    if (major == 3) {
        return 192;
    }
    return 64;
}

inline cudaDeviceProp active_device_properties() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    return prop;
}

inline double theoretical_fp32_gflops(const cudaDeviceProp& prop) {
    const int cores_per_sm =
        cuda_core_count_per_sm(prop.major, prop.minor);
    const double clock_hz = static_cast<double>(prop.clockRate) * 1000.0;
    const double flops = static_cast<double>(prop.multiProcessorCount) *
                         static_cast<double>(cores_per_sm) * 2.0 * clock_hz;
    return flops / 1.0e9;
}

inline double theoretical_memory_bandwidth_gbs(const cudaDeviceProp& prop) {
    const double clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    const double bus_bytes = static_cast<double>(prop.memoryBusWidth) / 8.0;
    return 2.0 * clock_hz * bus_bytes / 1.0e9;
}
