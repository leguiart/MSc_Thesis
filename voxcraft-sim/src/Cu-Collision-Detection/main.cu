#include <bitset>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "include/CollisionSystem.cuh"

int main() {
    cudaEvent_t real_start_t, start_t, stop_t;
    cudaEventCreate(&real_start_t);
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    float milliseconds = 0;
    unsigned int levels = 14;
    unsigned int N =  1<<levels;
    unsigned int MAX_COLS_PER_OBJECT = 64;

    std::cout << "Running on 1<<" << levels << " (" << N << ") points" << std::endl;
    auto colSystem = CollisionSystem(N, MAX_COLS_PER_OBJECT, true);
    CollisionSystem *colSys_d;
    cudaMalloc((void**)&colSys_d, sizeof(CollisionSystem));
    cudaMemcpy(colSys_d, &colSystem, sizeof(CollisionSystem), cudaMemcpyHostToDevice);

    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    auto x_pos_h = colSystem.get_x_pos_host();
    auto y_pos_h = colSystem.get_y_pos_host();
    auto z_pos_h = colSystem.get_z_pos_host();
    auto r_h = colSystem.get_radius_host();

    // init positions
    for (int i = 0; i < N; i++) {
        x_pos_h[i] = u01(rng);
        y_pos_h[i] = u01(rng);
        z_pos_h[i] = u01(rng);
        r_h[i] = 0.01;
    }

    // Sync from host.
    {
        cudaEventRecord(start_t);
        colSystem.update_all_from_host();
        cudaEventRecord(stop_t);
        cudaEventSynchronize(start_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Sync from host took: " << milliseconds << " ms" << std::endl;
    }

    cudaEventRecord(real_start_t);

    // sort ids by pos
    {
        cudaEventRecord(start_t);
        colSystem.update_x_pos_ranks();
        colSystem.update_y_pos_ranks();
        colSystem.update_z_pos_ranks();
        cudaEventRecord(stop_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Sort by pos took: " << milliseconds << " ms" << std::endl;
    }

    // generate morton numbers.
    {
        cudaEventRecord(start_t);
        colSystem.update_mortons();
    //  colSystem.update_mortons_fast({0, 1}, {0, 1}, {0, 1});
        cudaEventRecord(stop_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Compute morton numbers took: " << milliseconds << " ms" << std::endl;
    }


    // build bvh tree
    {
        cudaEventRecord(start_t);
        colSystem.build_tree();
        cudaEventRecord(stop_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Build bvh tree took: " << milliseconds << " ms" << std::endl;

    }

    // fill bvh tree with bounding boxes.
    {
        cudaEventRecord(start_t);
        colSystem.update_bounding_boxes();
        cudaEventRecord(stop_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Fill BVH Tree with Bounding Boxes took: " << milliseconds << " ms" << std::endl;
    }

    // compute list of collisions.
    {
        cudaEventRecord(start_t);
        int numColsGPU = colSystem.find_collisions();
        cudaEventRecord(stop_t);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&milliseconds, start_t, stop_t);
        std::cout << "Tree Traversal Took: " << milliseconds << " ms" << std::endl;
        std::cout << "There were " << numColsGPU << " collisions." << std::endl;
    }


    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&milliseconds, real_start_t, stop_t);
    std::cout << "Total took: " << milliseconds << " ms" << std::endl;

    return 0;
}
