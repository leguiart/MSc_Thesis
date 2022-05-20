#ifndef COLLISIONDETECTION_UTILS_CUH
#define COLLISIONDETECTION_UTILS_CUH

#ifndef CUDA_ERROR_CHECK
__device__ __host__ inline void CUDA_ERROR_CHECK_OUTPUT(cudaError_t code, const char *file, int line, bool abort=false) {
    if (code != cudaSuccess) {
        printf("%s(%d): CUDA Function Error: %s \n", file, line, cudaGetErrorString(code));
        if (abort) assert(0);
    }
}
#define CUDA_ERROR_CHECK(ans) { CUDA_ERROR_CHECK_OUTPUT((ans), __FILE__, __LINE__); }
#endif

#define CUDA_CHECK_AFTER_CALL() {CUDA_ERROR_CHECK(cudaGetLastError());}
#define VcudaDeviceSynchronize() {CUDA_ERROR_CHECK(cudaDeviceSynchronize())}


// // A simple allocator for caching cudaMalloc allocations.
// struct cached_allocator
// {

//     cached_allocator() {}

//     ~cached_allocator()
//     {
//         cudaFree((void*)bytes_allocated);
//     }

//     __device__
//     void *allocate(size_t num_bytes)
//     {
//         assert(!currently_allocated);
//         if (num_bytes > num_bytes_allocated) {
//             cudaFree(bytes_allocated);
//             cudaMalloc(&bytes_allocated, num_bytes);
//         }
//         return bytes_allocated;
//     }

//     __device__
//     void deallocate(void *ptr, size_t)
//     {
//         currently_allocated = false;
//     }

// private:
    
//     void* bytes_allocated;
//     size_t num_bytes_allocated;
//     bool currently_allocated;
// };


#endif // COLLISIONDETECTION_UTILS_CUH