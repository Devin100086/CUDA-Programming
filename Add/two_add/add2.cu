#include"add2.h"
#include<cuda_runtime.h>

__global__ void add2_kernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n){
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add2_kernel<<<numBlocks, blockSize>>>(a, b, c, n);
                 }