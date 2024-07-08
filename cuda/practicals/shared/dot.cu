#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// implement dot product kernel
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, const int n) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ double values[];


    // 1. multiply all the corresponding elements to obtain the element wise products
    if(i < n) {
        values[i] = x[i] * y[i];
    }

    // sync all the threads
    __syncthreads();

    // 2. compute the sum of all the elements.
    for(int n2 = n; i < n2 / 2; n2 = (n2 + 1) / 2) {
        values[i] += values[i + (n2 + 1) / 2];
        __syncthreads();
    }

    if(i == 0) {
        *result = values[i];
    }
}

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
    // call dot product kernel
    const unsigned block_dim = 1024;
    const unsigned grid_dim = (n + block_dim - 1) / block_dim;
    dot_gpu_kernel<<<grid_dim, block_dim, n * sizeof(double)>>>(x, y, result, n);

    cudaDeviceSynchronize();
    return *result;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

