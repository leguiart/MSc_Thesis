#include "stdio.h"
#include <halloc.h>

__global__ void kernel() {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 0; i < 10; i++) {
        int *m = (int *)hamalloc(sizeof(int));
        if (m == NULL) {
            printf("Thread %d: iter %d: Out of memory..\n", idx, i);
            return;
        }
        *m = 0;
        atomicCAS(m, 0, 1); // make sure it is shared or global memory.
        atomicExch(m, 0);
        printf("Thread %d: iter %d: m: %d.\n", idx, i, *m);
        hafree(m);
    }
}

int main() {
    ha_init(halloc_opts_t());
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("demo.\n");
}