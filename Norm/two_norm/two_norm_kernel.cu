#include "two_norm_kernel.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void two_norm_kernel(const float *a, const float *b, float *c, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < m; j++) {
        float diff = a[row * m + j] - b[row * m + j];
        sum += diff * diff;
    }
    c[row] = sqrtf(sum);
}

void launch_two_norm(const float *a, const float *b, float *c, int n, int m) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    two_norm_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
}