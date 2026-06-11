ARCH ?= sm_89
CXX ?= g++
NVCC ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)

.PHONY: all clean

all: heat_serial heat_cuda heat_cuda_tile cuda_device_info

heat_serial: heat_serial.cpp heat_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ heat_serial.cpp

heat_cuda: heat_cuda.cu heat_common.hpp cuda_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ heat_cuda.cu

heat_cuda_tile: heat_cuda_tile.cu heat_common.hpp cuda_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ heat_cuda_tile.cu

cuda_device_info: cuda_device_info.cu cuda_common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ cuda_device_info.cu

clean:
	rm -f heat_serial heat_cuda heat_cuda_tile cuda_device_info
